"""Elige cuándo marcar una pieza como dudosa y mide si sirve.

Uso:
    uv run python scripts/ajustar_duda.py --ejecucion base

Una pieza es dudosa si su arcada pierde poca afinidad cuando se le prohíbe su
número (ver postproceso.py). El umbral se elige en validación con una regla
fija, decidida antes de mirar la prueba: el mayor que marque como mucho DUDAS
piezas por radiografía de media. Más de dos por boca ya no es «revisa estas»,
es «revísalo todo». Después se mide una sola vez en prueba.

Usa las predicciones en caché de scripts/evaluar.py (las dos particiones) y
guarda el umbral y las cifras en modelos/<ejecucion>/duda.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dentalvision.datos import FDI_A_CLASE
from dentalvision.evaluacion import emparejar
from dentalvision.postproceso import opciones, postprocesar

RAIZ = Path(__file__).resolve().parents[1]
DUDAS = 2.0  # piezas marcadas por radiografía, de media, como mucho
REMUESTREOS = 2000


def piezas(carpeta: Path, particion: str) -> list[list[tuple[float, bool, bool]]]:
    """Por radiografía, cada pieza emparejada con una real:
    (pérdida, ¿bien numerada?, ¿la alternativa era la buena?)."""
    cache = carpeta / f"predicciones_{particion}.pt"
    if not cache.exists():
        sys.exit(f"Falta {cache}: pasa antes scripts/evaluar.py --particion {particion}.")
    datos = torch.load(cache, weights_only=True)
    ajustes = json.loads((carpeta / "postproceso.json").read_text())["orden"]
    config = json.loads((carpeta / "config.json").read_text())
    salida = []
    for p, v in zip(datos["preds"], datos["reales"], strict=True):
        f = postprocesar(p, "orden", **opciones(config, ajustes), dudas=True)
        orden = torch.argsort(f["scores"], descending=True)
        etiquetas, alternativas = f["labels"][orden], f["alternativas"][orden]
        perdidas = f["perdidas"][orden]
        filas = []
        for j, i in emparejar(f["boxes"][orden], v["boxes"]):
            real = int(v["labels"][i])
            alternativa = int(alternativas[j])
            filas.append((float(perdidas[j]), int(etiquetas[j]) == real,
                          alternativa > 0 and FDI_A_CLASE[alternativa] == real))
        salida.append(filas)
    return salida


def medir(rx: list[list[tuple[float, bool, bool]]], umbral: float) -> dict:
    marcadas = mal = cazados = alternativa_buena = 0
    for filas in rx:
        for perdida, bien, alt in filas:
            mal += not bien
            if perdida < umbral:
                marcadas += 1
                cazados += not bien
                alternativa_buena += (not bien) and alt
    return {
        "marcadas_por_radiografia": marcadas / len(rx),
        "fallos_cazados": cazados / max(mal, 1),
        "fallo_entre_marcadas": cazados / max(marcadas, 1),
        "alternativa_correcta_en_cazados": alternativa_buena / max(cazados, 1),
        "tasa_de_fallo_base": mal / max(sum(len(f) for f in rx), 1),
        "fallos": mal,
        "marcadas": marcadas,
    }


def intervalo(rx, umbral: float, clave: str, semilla: int = 0) -> list[float]:
    """IC 95% por bootstrap de radiografías: las piezas de una boca no son independientes."""
    gen = np.random.default_rng(semilla)
    valores = [medir([rx[k] for k in gen.integers(0, len(rx), len(rx))], umbral)[clave]
               for _ in range(REMUESTREOS)]
    return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]


def elegir(rx) -> float:
    candidatos = sorted({p for filas in rx for p, _, _ in filas})
    validos = [u for u in candidatos if medir(rx, u)["marcadas_por_radiografia"] <= DUDAS]
    return validos[-1] if validos else 0.0


def pct(x: float) -> str:
    return f"{x:.0%}"


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--ejecucion", required=True, help="carpeta dentro de modelos/")
    carpeta = RAIZ / "modelos" / a.parse_args().ejecucion

    validacion = piezas(carpeta, "validacion")
    umbral = elegir(validacion)
    resultado = {"umbral_perdida": umbral, "regla": f"el mayor con <= {DUDAS} marcadas por radiografía en validación"}
    for nombre, rx in (("validacion", validacion), ("prueba", piezas(carpeta, "prueba"))):
        m = medir(rx, umbral)
        for clave in ("marcadas_por_radiografia", "fallos_cazados", "fallo_entre_marcadas"):
            m[f"{clave}_ic95"] = intervalo(rx, umbral, clave)
        resultado[nombre] = m
        print(f"{nombre:10}  {m['marcadas_por_radiografia']:.1f} marcadas por radiografía · "
              f"cazan el {pct(m['fallos_cazados'])} de los fallos "
              f"({pct(m['fallos_cazados_ic95'][0])}–{pct(m['fallos_cazados_ic95'][1])}) · "
              f"{pct(m['fallo_entre_marcadas'])} de las marcadas está mal "
              f"({pct(m['fallo_entre_marcadas_ic95'][0])}–{pct(m['fallo_entre_marcadas_ic95'][1])}), "
              f"frente al {m['tasa_de_fallo_base']:.1%} de todas · "
              f"la alternativa acierta en el {pct(m['alternativa_correcta_en_cazados'])} de los cazados")

    ruta = carpeta / "duda.json"
    ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n")
    print(f"\nUmbral {umbral:.4f}, guardado en {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
