"""Evalúa un modelo entrenado y ajusta el post-procesado.

Uso:
    uv run python scripts/evaluar.py --ejecucion base
    uv run python scripts/evaluar.py --ejecucion base --particion prueba

En validación se prueban los umbrales del post-procesado y se guardan los
mejores en modelos/<ejecucion>/postproceso.json. La partición de prueba solo se
evalúa con esos valores ya fijados: elegirlos mirando la prueba inflaría los
resultados, y el script se niega a hacerlo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dentalvision.datos import CLASE_A_FDI, DientesDataset, colacion
from dentalvision.dentex import NOMBRE_POSICION, Subconjunto
from dentalvision.evaluacion import (
    Resultado,
    emparejar,
    evaluar,
    evaluar_por_radiografia,
    intervalo,
)
from dentalvision.modelo import cargar, dispositivo, sin_mascaras
from dentalvision.particiones import particionar
from dentalvision.postproceso import MODOS, postprocesar

RAIZ = Path(__file__).resolve().parents[1]
UMBRALES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
KAPPAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
METRICAS = {
    "cobertura": lambda r: r.cobertura,
    "numeracion": lambda r: r.numeracion,
    "exactitud": lambda r: r.exactitud,
    "errores_por_radiografia": lambda r: r.errores_por_radiografia,
    "perfectas": lambda r: r.perfectas,
}


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--ejecucion", required=True, help="carpeta dentro de modelos/")
    p.add_argument("--particion", choices=("validacion", "prueba"), default="validacion")
    p.add_argument("--rehacer", action="store_true", help="repetir la inferencia aunque haya cache")
    return p.parse_args()


def predicciones(carpeta: Path, particion: str, rehacer: bool):
    """Salida cruda del detector para toda la partición, con caché en disco."""
    cache = carpeta / f"predicciones_{particion}.pt"
    pesos = carpeta / "mejor.pt"
    if cache.exists() and not rehacer and cache.stat().st_mtime > pesos.stat().st_mtime:
        datos = torch.load(cache, weights_only=True)
        return datos["preds"], datos["reales"], datos["ficheros"]

    dev = dispositivo()
    modelo, config = cargar(carpeta, dev=dev)
    sub = Subconjunto("enumeracion")
    radiografias = getattr(particionar(sub), particion)
    ds = DientesDataset(sub, radiografias, ancho=config["ancho"], con_mascaras=False)
    dl = DataLoader(ds, batch_size=2, collate_fn=colacion, num_workers=4)
    preds, reales = [], []
    with torch.no_grad(), sin_mascaras(modelo):
        for imgs, tgts in tqdm(dl, desc=f"inferencia ({particion})"):
            salida = modelo([x.to(dev) for x in imgs])
            preds.extend({k: v.cpu() for k, v in s.items()} for s in salida)
            reales.extend(tgts)
    ficheros = [r.fichero for r in radiografias]
    torch.save({"preds": preds, "reales": reales, "ficheros": ficheros}, cache)
    return preds, reales, ficheros


def aplicar(preds, modo, ajustes):
    return [postprocesar(p, modo, umbral=ajustes["umbral"], kappa=ajustes["kappa"] or 0.2)
            for p in preds]


def ajustar(preds, reales, carpeta: Path) -> None:
    mejores = {}
    for modo in MODOS:
        for umbral, kappa in product(UMBRALES, KAPPAS if modo == "orden" else (None,)):
            ajustes = {"umbral": umbral, "kappa": kappa}
            r = evaluar(aplicar(preds, modo, ajustes), reales, umbral_score=0.0)
            print(f"{modo:9} umbral {umbral:.1f}  kappa {kappa or '-':>4}   {r}")
            clave = (r.errores_por_radiografia, -r.exactitud)
            if modo not in mejores or clave < mejores[modo][0]:
                mejores[modo] = (clave, ajustes, r)

    print("\nMejor por modo en validación (menos errores por radiografía):")
    for modo, (_, ajustes, r) in mejores.items():
        print(f"  {modo:9} {ajustes}   {r}")
    ruta = carpeta / "postproceso.json"
    ruta.write_text(json.dumps({m: a for m, (_, a, _) in mejores.items()}, indent=2) + "\n")
    print(f"\nGuardado en {ruta}")


def pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def fallos_por_posicion(salida, reales):
    """Piezas mal numeradas, no encontradas y sobrantes por posición (1..8)."""
    mal, faltan, sobran = Counter(), Counter(), Counter()
    for p, v in zip(salida, reales, strict=True):
        orden = torch.argsort(p["scores"], descending=True)
        cajas, etiquetas = p["boxes"][orden], p["labels"][orden]
        pares = emparejar(cajas, v["boxes"])
        for j, i in pares:
            if etiquetas[j] != v["labels"][i]:
                mal[CLASE_A_FDI[int(v["labels"][i])] % 10] += 1
        encontradas = {i for _, i in pares}
        usadas = {j for j, _ in pares}
        for i, c in enumerate(v["labels"]):
            if i not in encontradas:
                faltan[CLASE_A_FDI[int(c)] % 10] += 1
        for j, c in enumerate(etiquetas):
            if j not in usadas:
                sobran[CLASE_A_FDI[int(c)] % 10] += 1
    return mal, faltan, sobran


def probar(preds, reales, ficheros, carpeta: Path) -> None:
    ruta = carpeta / "postproceso.json"
    if not ruta.exists():
        sys.exit("Falta postproceso.json: ajusta antes en validación "
                 f"(uv run python scripts/evaluar.py --ejecucion {carpeta.name}).")
    ajustes = json.loads(ruta.read_text())

    resultados = {}
    print("| Post-procesado | Cobertura | Numeración | Exactitud | Errores por radiografía | Odontogramas perfectos |")
    print("|---|---|---|---|---|---|")
    for modo in MODOS:
        por_rx = evaluar_por_radiografia(aplicar(preds, modo, ajustes[modo]), reales, umbral_score=0.0)
        total = sum(por_rx, Resultado())
        fila = {"ajustes": ajustes[modo], "totales": vars(total)}
        celdas = []
        for nombre, f in METRICAS.items():
            lo, hi = intervalo(por_rx, f)
            fila[nombre] = {"valor": f(total), "ic95": [lo, hi]}
            fmt = num if nombre == "errores_por_radiografia" else pct
            celdas.append(f"{fmt(f(total))} ({fmt(lo)}–{fmt(hi)})")
        fila["errores_por_imagen"] = {
            fichero: r.errores_por_radiografia for fichero, r in zip(ficheros, por_rx)
        }
        resultados[modo] = fila
        print(f"| {modo} | " + " | ".join(celdas) + " |")

    final = aplicar(preds, "orden", ajustes["orden"])
    errores = Counter(min(3, round(r.errores_por_radiografia))
                      for r in evaluar_por_radiografia(final, reales, umbral_score=0.0))
    print(f"\nRadiografías por número de errores (orden): 0 -> {errores[0]}, 1 -> {errores[1]}, "
          f"2 -> {errores[2]}, 3 o más -> {errores[3]}  (de {len(reales)})")

    mal, faltan, sobran = fallos_por_posicion(final, reales)
    print("\n| Posición | Mal numeradas | No encontradas | Sobrantes |")
    print("|---|---|---|---|")
    for pos, nombre in NOMBRE_POSICION.items():
        print(f"| {pos} ({nombre}) | {mal[pos]} | {faltan[pos]} | {sobran[pos]} |")
    resultados["orden"]["fallos_por_posicion"] = {
        "mal_numeradas": dict(mal), "no_encontradas": dict(faltan), "sobrantes": dict(sobran)}

    salida = carpeta / "resultados_prueba.json"
    salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False) + "\n")
    print(f"\nGuardado en {salida}")


def main() -> int:
    a = argumentos()
    carpeta = RAIZ / "modelos" / a.ejecucion
    preds, reales, ficheros = predicciones(carpeta, a.particion, a.rehacer)
    if a.particion == "validacion":
        ajustar(preds, reales, carpeta)
    else:
        probar(preds, reales, ficheros, carpeta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
