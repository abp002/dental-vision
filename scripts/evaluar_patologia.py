"""Evalúa el detector de hallazgos (caries, caries profunda, lesión periapical,
diente impactado) y su unión con el odontograma.

Uso:
    uv run python scripts/evaluar_patologia.py --ejecucion patologia --numeracion base

1. Ajusta en la validación propia (102 radiografías) el umbral de confianza de
   la métrica clínica, y lo guarda en modelos/<ejecucion>/umbral.json.
2. Evalúa una vez en los conjuntos oficiales del concurso, que el detector no
   ha visto:
   - validación oficial (50 radiografías, las 4 clases oficiales);
   - prueba oficial (250 radiografías, solo las 3 clases cuya traducción desde
     las etiquetas originales no admite duda: ver dentex.TRADUCCION_PRUEBA).

Métricas:
   - AP al estilo COCO, la del concurso, por diagnóstico y, uniendo cada
     hallazgo a su diente con el modelo de numeración, por cuadrante y número.
   - Clínica: de los hallazgos reales, cuántos encuentra (sensibilidad), qué
     parte de lo que marca es real (precisión) y falsas alarmas por radiografía.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from dentalvision.datos import (
    CLASE_A_PATOLOGIA,
    PATOLOGIA_A_CLASE,
    a_tensor,
    cargar_imagen,
)
from dentalvision.dentex import TRADUCCION_PRUEBA, Subconjunto
from dentalvision.evaluacion import Hallazgos, ap_coco, hallazgos_por_clase, iou
from dentalvision.inferencia import Odontografo
from dentalvision.modelo import cargar, dispositivo, sin_mascaras
from dentalvision.particiones import particion_de

RAIZ = Path(__file__).resolve().parents[1]
UMBRALES = [round(0.05 * k, 2) for k in range(2, 19)]


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--ejecucion", required=True, help="modelo de patología en modelos/")
    p.add_argument("--numeracion", required=True, help="modelo de numeración en modelos/")
    p.add_argument("--repeticiones", type=int, default=500, help="bootstrap")
    return p.parse_args()


def asignar_diente(pred: dict, piezas) -> list[int | None]:
    """FDI de cada hallazgo: la pieza del odontograma que más solapa con él."""
    if not piezas or len(pred["boxes"]) == 0:
        return [None] * len(pred["boxes"])
    cajas = torch.tensor([p.caja for p in piezas])
    solape = iou(pred["boxes"], cajas)
    mejor = solape.argmax(dim=1)
    return [piezas[int(j)].fdi if solape[i, j] > 0.3 else None for i, j in enumerate(mejor)]


@torch.no_grad()
def predecir(modelo, odontografo, sub, radiografias, dev, ancho):
    """Por radiografía: hallazgos predichos (con su FDI) y reales, a la escala de trabajo."""
    preds, reales = [], []
    for r in tqdm(radiografias, desc=sub.nombre):
        img, escala = cargar_imagen(sub.ruta(r), ancho)
        with sin_mascaras(modelo):
            p = {k: v.cpu() for k, v in modelo([a_tensor(img).to(dev)])[0].items()}
        p["fdi"] = asignar_diente(p, odontografo.predecir(img).piezas)
        preds.append(p)
        dientes = [d for d in r.dientes if d.patologia in PATOLOGIA_A_CLASE]
        reales.append({
            "boxes": torch.tensor([[x * escala, y * escala, (x + w) * escala, (y + h) * escala]
                                   for x, y, w, h in (d.bbox for d in dientes)]).reshape(-1, 4),
            "labels": torch.tensor([PATOLOGIA_A_CLASE[d.patologia] for d in dientes], dtype=torch.int64),
            "fdi": [d.fdi for d in dientes],
        })
    return preds, reales


def por_nivel(preds, reales, nivel: str):
    """Reetiqueta predicciones y verdades por cuadrante o por número de pieza,
    como hace el evaluador oficial con sus tres niveles de etiqueta."""
    def clave(fdi):
        if fdi is None:
            return None
        return fdi // 10 if nivel == "cuadrante" else fdi % 10

    def reetiquetar(d):
        claves = [clave(f) for f in d["fdi"]]
        keep = torch.tensor([c is not None for c in claves], dtype=torch.bool)
        out = {"boxes": d["boxes"][keep],
               "labels": torch.tensor([c for c in claves if c is not None], dtype=torch.int64)}
        if "scores" in d:
            out["scores"] = d["scores"][keep]
        return out

    return [reetiquetar(p) for p in preds], [reetiquetar(v) for v in reales]


def con_intervalo(funcion, n: int, repeticiones: int) -> tuple[float, float, float]:
    """Valor y percentiles 2,5-97,5 remuestreando radiografías."""
    valor = funcion(list(range(n)))
    rng = random.Random(0)
    muestras = sorted(funcion([rng.randrange(n) for _ in range(n)]) for _ in range(repeticiones))
    return valor, muestras[int(0.025 * repeticiones)], muestras[int(0.975 * repeticiones) - 1]


def elegir_umbral(preds, reales, clases) -> float:
    """El que maximiza la F1 media por clase."""
    def f1_media(u):
        res = hallazgos_por_clase(preds, reales, clases, umbral_score=u)
        f1s = []
        for lista in res.values():
            h = sum(lista, Hallazgos())
            s, p = h.sensibilidad, h.precision
            f1s.append(2 * s * p / (s + p) if s + p else 0.0)
        return sum(f1s) / len(f1s)
    return max(UMBRALES, key=f1_media)


def informe(nombre, preds, reales, clases, umbral, repeticiones, vistas: set[int]):
    n = len(preds)
    print(f"\n## {nombre} ({n} radiografías)\n")
    resultado = {}

    def ap(indices, metrica, nivel=None, subconjunto=None):
        idx = [i for i in indices if subconjunto is None or i in subconjunto]
        p, v = [preds[i] for i in idx], [reales[i] for i in idx]
        if nivel:
            p, v = por_nivel(p, v, nivel)
            etiquetas = {k: str(k) for k in (range(1, 5) if nivel == "cuadrante" else range(1, 9))}
        else:
            etiquetas = clases
        return ap_coco(p, v, etiquetas)[metrica]

    print("| Métrica (estilo COCO) | Valor (IC 95%) |")
    print("|---|---|")
    for metrica in ["AP", "AP50", "AP75", "AR100"] + [f"AP {c}" for c in clases.values()]:
        v, lo, hi = con_intervalo(lambda idx: ap(idx, metrica), n, repeticiones)
        resultado[f"diagnostico {metrica}"] = [v, lo, hi]
        print(f"| Diagnóstico {metrica} | {v:.3f} ({lo:.3f}–{hi:.3f}) |")
    no_vistas = set(range(n)) - vistas
    for nivel in ("cuadrante", "numeracion"):
        v, lo, hi = con_intervalo(lambda idx: ap(idx, "AP", nivel, no_vistas), n, repeticiones)
        resultado[f"{nivel} AP"] = [v, lo, hi]
        print(f"| {nivel.capitalize()} AP ({len(no_vistas)} rx no vistas por el modelo de numeración) "
              f"| {v:.3f} ({lo:.3f}–{hi:.3f}) |")

    res = hallazgos_por_clase(preds, reales, clases, umbral_score=umbral)
    print(f"\n| Hallazgo (umbral {umbral:.2f}) | Reales | Sensibilidad | Precisión | Falsas alarmas por rx |")
    print("|---|---|---|---|---|")
    for clase, lista in res.items():
        h = sum(lista, Hallazgos())
        s = con_intervalo(lambda idx: sum((lista[i] for i in idx), Hallazgos()).sensibilidad, n, 2000)
        pr = con_intervalo(lambda idx: sum((lista[i] for i in idx), Hallazgos()).precision, n, 2000)
        resultado[f"clinica {clase}"] = {"reales": h.reales, "sensibilidad": s, "precision": pr,
                                         "falsas_por_rx": h.falsas_por_radiografia}
        print(f"| {clase} | {h.reales} | {s[0]:.0%} ({s[1]:.0%}–{s[2]:.0%}) | "
              f"{pr[0]:.0%} ({pr[1]:.0%}–{pr[2]:.0%}) | {h.falsas_por_radiografia:.2f} |")
    return resultado


def main() -> int:
    a = argumentos()
    dev = dispositivo()
    carpeta = RAIZ / "modelos" / a.ejecucion
    modelo, config = cargar(carpeta, dev=dev)
    odontografo = Odontografo(RAIZ / "modelos" / a.numeracion, dev=dev)
    ancho = config["ancho"]
    todas = dict(CLASE_A_PATOLOGIA)
    tres = {k: v for k, v in todas.items() if v in TRADUCCION_PRUEBA.values()}

    # Radiografías que el modelo de numeración vio al entrenar: su asignación
    # de número a cada hallazgo sería optimista y se excluye de esos AP.
    sub_num, part_num = particion_de("numeracion")
    huellas_num = {sub_num.huella(r) for r in part_num.entrenamiento + part_num.validacion}

    sub, part = particion_de("patologia")
    preds, reales = predecir(modelo, odontografo, sub, part.validacion, dev, ancho)
    umbral = elegir_umbral(preds, reales, todas)
    (carpeta / "umbral.json").write_text(json.dumps({"umbral": umbral}) + "\n")
    print(f"Umbral elegido en la validación propia: {umbral:.2f}")

    resultados = {"umbral": umbral}
    for nombre, clases in (("validacion_oficial", todas), ("prueba_oficial", tres)):
        s = Subconjunto(nombre)
        p, v = predecir(modelo, odontografo, s, s.radiografias, dev, ancho)
        vistas = {i for i, r in enumerate(s.radiografias) if s.huella(r) in huellas_num}
        if nombre == "prueba_oficial":
            # Solo las clases evaluables: fuera las predicciones de caries profunda.
            p = [{**x, **{k: x[k][torch.isin(x["labels"], torch.tensor(list(clases)))]
                          for k in ("boxes", "labels", "scores")},
                  "fdi": [f for f, c in zip(x["fdi"], x["labels"].tolist()) if c in clases]}
                 for x in p]
        resultados[nombre] = informe(nombre, p, v, clases, umbral, a.repeticiones, vistas)

    salida = carpeta / "resultados_oficiales.json"
    salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False) + "\n")
    print(f"\nGuardado en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
