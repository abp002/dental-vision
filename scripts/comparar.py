"""Compara ejecuciones en validación con bootstrap emparejado.

Uso:
    uv run python scripts/comparar.py base rois512 bn-congelada

La primera ejecución es la referencia. Cada una usa su propio post-procesado
(modo orden, ajustado antes con scripts/evaluar.py). Las diferencias se
calculan remuestreando las mismas radiografías para todos los modelos: la
dificultad de cada radiografía se cancela y queda solo la diferencia entre
modelos.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from dentalvision.evaluacion import (
    Resultado,
    evaluar_por_radiografia,
    intervalo_diferencia,
)
from dentalvision.postproceso import postprocesar

RAIZ = Path(__file__).resolve().parents[1]


def cargar_ejecucion(nombre: str):
    carpeta = RAIZ / "modelos" / nombre
    cache = torch.load(carpeta / "predicciones_validacion.pt", weights_only=True)
    ajustes = json.loads((carpeta / "postproceso.json").read_text())["orden"]
    salida = [postprocesar(p, "orden", umbral=ajustes["umbral"], kappa=ajustes["kappa"])
              for p in cache["preds"]]
    por_rx = evaluar_por_radiografia(salida, cache["reales"], umbral_score=0.0)
    historial = [json.loads(l) for l in (carpeta / "historial.jsonl").read_text().splitlines()]
    return {
        "ficheros": cache["ficheros"],
        "por_rx": por_rx,
        "total": sum(por_rx, Resultado()),
        # Mediana: robusta a una época ralentizada por otro proceso en la GPU.
        "s_paso": statistics.median(h["segundos_por_paso"] for h in historial),
        "memoria": max(h["memoria_gb"] for h in historial),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ejecuciones", nargs="+")
    a = p.parse_args()

    datos = {n: cargar_ejecucion(n) for n in a.ejecuciones}
    ref = a.ejecuciones[0]
    for n in a.ejecuciones[1:]:
        if datos[n]["ficheros"] != datos[ref]["ficheros"]:
            sys.exit(f"{n} y {ref} no se evaluaron sobre las mismas radiografías")

    print("| Ejecución | s/paso | VRAM | Exactitud | Errores por radiografía | "
          f"Diferencia con {ref} (IC 95%) | Odontogramas perfectos |")
    print("|---|---|---|---|---|---|---|")
    for n, d in datos.items():
        t = d["total"]
        if n == ref:
            dif = "—"
        else:
            lo, hi = intervalo_diferencia(datos[ref]["por_rx"], d["por_rx"],
                                          lambda r: r.errores_por_radiografia)
            delta = t.errores_por_radiografia - datos[ref]["total"].errores_por_radiografia
            dif = f"{delta:+.2f} ({lo:+.2f} a {hi:+.2f})".replace(".", ",")
        print(f"| {n} | {d['s_paso']:.3f} | {d['memoria']:.1f} GB | {t.exactitud:.1%} | "
              f"{t.errores_por_radiografia:.2f} | {dif} | {t.perfectas:.1%} |".replace(".", ","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
