"""Genera el odontograma de una radiografía panorámica.

Uso:
    uv run python scripts/odontograma.py --ejecucion base --imagen panoramica.png
    uv run python scripts/odontograma.py --ejecucion base --particion prueba --indice 7

Con --particion la radiografía sale del dataset y se compara con su anotación:
los fallos se marcan en rojo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from dentalvision.dentex import Subconjunto
from dentalvision.evaluacion import emparejar
from dentalvision.inferencia import Odontografo
from dentalvision.particiones import particionar
from dentalvision.render import Pieza, apilar, dibujar_odontograma, dibujar_prediccion

RAIZ = Path(__file__).resolve().parents[1]


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--ejecucion", required=True, help="carpeta dentro de modelos/")
    origen = p.add_mutually_exclusive_group(required=True)
    origen.add_argument("--imagen", help="ruta a una panorámica")
    origen.add_argument("--particion", choices=("validacion", "prueba"))
    p.add_argument("--indice", type=int, default=0, help="radiografía dentro de la partición")
    p.add_argument("--fichero", help="radiografía de la partición por nombre de fichero")
    p.add_argument("--salida", help="PNG de salida (por defecto en salidas/)")
    p.add_argument("--patologia", help="modelo de hallazgos en modelos/ (opcional)")
    return p.parse_args()


def comparar(final: dict, piezas: list[Pieza], radiografia, escala: float) -> set[int]:
    """Marca cada pieza predicha como bien/mal/sobra y añade las que faltan.

    Devuelve las casillas del odontograma que no coinciden con la anotación.
    """
    reales = [d for d in radiografia.dientes if d.bbox[2] * escala >= 2 and d.bbox[3] * escala >= 2]
    cajas_r = torch.tensor(
        [[x * escala, y * escala, (x + w) * escala, (y + h) * escala] for x, y, w, h in (d.bbox for d in reales)]
    ).reshape(-1, 4)
    orden = torch.argsort(final["scores"], descending=True).tolist()
    pares = emparejar(final["boxes"][orden], cajas_r)
    emparejada = {orden[j]: i for j, i in pares}

    for k, p in enumerate(piezas):
        if k not in emparejada:
            p.estado = "sobra"
        else:
            real = reales[emparejada[k]].fdi
            p.estado, p.real = ("bien", None) if real == p.fdi else ("mal", real)
    encontradas = set(emparejada.values())
    for i, d in enumerate(reales):
        if i not in encontradas:
            piezas.append(Pieza(fdi=d.fdi, caja=tuple(cajas_r[i].tolist()), estado="falta"))

    predichas = {p.fdi for p in piezas if p.estado != "falta"}
    return predichas ^ {d.fdi for d in reales}


def main() -> int:
    a = argumentos()
    odontografo = Odontografo(
        RAIZ / "modelos" / a.ejecucion,
        patologia=RAIZ / "modelos" / a.patologia if a.patologia else None,
    )

    radiografia = None
    if a.imagen:
        ruta = Path(a.imagen)
    else:
        sub = Subconjunto("enumeracion")
        particion = getattr(particionar(sub), a.particion)
        if a.fichero:
            radiografia = next(r for r in particion if r.fichero == a.fichero)
        else:
            radiografia = particion[a.indice]
        ruta = sub.ruta(radiografia)

    pred = odontografo.predecir(ruta)
    img, escala, piezas, final = pred.imagen, pred.escala, pred.piezas, pred.final
    errores: set[int] = set()
    if radiografia is not None:
        errores = comparar(final, piezas, radiografia, escala)
        cuenta = {e: sum(p.estado == e for p in piezas) for e in ("bien", "mal", "sobra", "falta")}
        print(f"{radiografia.fichero}: {cuenta['bien']} bien, {cuenta['mal']} mal numeradas "
              f"{[f'{p.fdi}>{p.real}' for p in piezas if p.estado == 'mal']}, "
              f"{cuenta['sobra']} sobran, {cuenta['falta']} faltan")

    presentes = {p.fdi for p in piezas if p.estado != "falta"}
    figura = apilar(
        dibujar_prediccion(img, piezas, hallazgos=pred.hallazgos),
        dibujar_odontograma(presentes, errores=errores, hallazgos=pred.por_diente(),
                            titulo="Odontograma generado"),
    )
    destino = Path(a.salida) if a.salida else RAIZ / "salidas" / f"odontograma_{ruta.stem}.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.save(destino, optimize=True)
    print(f"Guardado en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
