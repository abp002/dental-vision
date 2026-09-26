"""Métricas del odontograma.

La métrica estándar en detección es mAP. Es útil para comparar con la
literatura y no le dice absolutamente nada a un dentista.

Aquí se miden cosas que sí significan algo en la consulta:

  cobertura    de las piezas realmente presentes, cuántas encuentra
  numeración   de las que encuentra, a cuántas les pone el número FDI correcto
  exactitud    de las piezas presentes, cuántas salen encontradas Y bien
               numeradas (cobertura x numeración)
  errores      correcciones por radiografía: piezas que faltan, números mal
               puestos y detecciones que sobran. Es lo que tiene que arreglar
               el profesional al revisar el borrador
  perfectos    radiografías cuyo odontograma sale sin un solo fallo

Un sistema con 97% de numeración correcta suena magnífico hasta que caes en
que, con 28 piezas por boca, eso son casi 1 error por radiografía. Por eso se
informa de los errores por radiografía y no solo de porcentajes por pieza.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Intersección sobre unión entre dos conjuntos de cajas xyxy."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    x1 = torch.max(a[:, None, 0], b[None, :, 0])
    y1 = torch.max(a[:, None, 1], b[None, :, 1])
    x2 = torch.min(a[:, None, 2], b[None, :, 2])
    y2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Resultado:
    piezas_reales: int = 0
    piezas_encontradas: int = 0
    numeracion_correcta: int = 0
    falsos_positivos: int = 0
    radiografias: int = 0
    radiografias_perfectas: int = 0

    def __add__(self, otro: Resultado) -> Resultado:
        return Resultado(*(a + b for a, b in zip(asdict(self).values(), asdict(otro).values())))

    @property
    def cobertura(self) -> float:
        return self.piezas_encontradas / max(1, self.piezas_reales)

    @property
    def numeracion(self) -> float:
        return self.numeracion_correcta / max(1, self.piezas_encontradas)

    @property
    def exactitud(self) -> float:
        return self.numeracion_correcta / max(1, self.piezas_reales)

    @property
    def perfectas(self) -> float:
        return self.radiografias_perfectas / max(1, self.radiografias)

    @property
    def errores_por_radiografia(self) -> float:
        fallos = (self.piezas_reales - self.numeracion_correcta) + self.falsos_positivos
        return fallos / max(1, self.radiografias)

    def metricas(self) -> dict[str, float]:
        return {
            "cobertura": self.cobertura,
            "numeracion": self.numeracion,
            "exactitud": self.exactitud,
            "errores_por_radiografia": self.errores_por_radiografia,
            "perfectas": self.perfectas,
        }

    def __str__(self) -> str:
        return (
            f"cobertura {self.cobertura:6.1%}   "
            f"numeracion {self.numeracion:6.1%}   "
            f"exactitud {self.exactitud:6.1%}   "
            f"errores/radiografia {self.errores_por_radiografia:.2f}   "
            f"perfectos {self.perfectas:6.1%}"
        )


def emparejar(
    cajas_p: torch.Tensor, cajas_r: torch.Tensor, umbral_iou: float = 0.5
) -> list[tuple[int, int]]:
    """Parejas (predicción, pieza real). Las predicciones deben venir ordenadas
    de más a menos confianza.

    Voraz por confianza, como en COCO: cada predicción se queda con la pieza
    real libre que más solapa. Si dos predicciones caen sobre el mismo diente
    gana la más segura, y la otra queda sin pareja (falso positivo).
    """
    matriz = iou(cajas_p, cajas_r)
    libres = torch.ones(len(cajas_r), dtype=torch.bool)
    pares = []
    for j in range(len(cajas_p)):
        if not libres.any():
            break
        solape = torch.where(libres, matriz[j], torch.tensor(-1.0))
        i = int(solape.argmax())
        if solape[i] < umbral_iou:
            continue
        libres[i] = False
        pares.append((j, i))
    return pares


def evaluar_radiografia(
    pred: dict,
    real: dict,
    *,
    umbral_iou: float = 0.5,
    umbral_score: float = 0.5,
) -> Resultado:
    r = Resultado(radiografias=1)
    keep = pred["scores"] >= umbral_score
    orden = torch.argsort(pred["scores"][keep], descending=True)
    cajas_p = pred["boxes"][keep][orden].cpu()
    etiq_p = pred["labels"][keep][orden].cpu()
    cajas_r = real["boxes"].cpu()
    etiq_r = real["labels"].cpu()

    r.piezas_reales = len(cajas_r)
    pares = emparejar(cajas_p, cajas_r, umbral_iou)
    r.piezas_encontradas = len(pares)
    r.numeracion_correcta = sum(int(etiq_p[j] == etiq_r[i]) for j, i in pares)
    r.falsos_positivos = len(cajas_p) - r.piezas_encontradas
    if r.numeracion_correcta == len(cajas_r) and len(cajas_p) == len(cajas_r):
        r.radiografias_perfectas = 1
    return r


def evaluar_por_radiografia(
    predicciones: list[dict], verdades: list[dict], **umbrales
) -> list[Resultado]:
    return [
        evaluar_radiografia(p, v, **umbrales)
        for p, v in zip(predicciones, verdades, strict=True)
    ]


def evaluar(predicciones: list[dict], verdades: list[dict], **umbrales) -> Resultado:
    return sum(evaluar_por_radiografia(predicciones, verdades, **umbrales), Resultado())


def intervalo(
    por_radiografia: list[Resultado],
    metrica: Callable[[Resultado], float],
    *,
    repeticiones: int = 2000,
    confianza: float = 0.95,
    semilla: int = 0,
) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap, remuestreando radiografías.

    Se remuestrea por radiografía y no por pieza: los dientes de una misma boca
    no son independientes (misma exposición, mismo paciente), y tratarlos como
    tales daría intervalos falsamente estrechos.
    """
    rng = random.Random(semilla)
    n = len(por_radiografia)
    valores = sorted(
        metrica(sum((por_radiografia[rng.randrange(n)] for _ in range(n)), Resultado()))
        for _ in range(repeticiones)
    )
    cola = (1 - confianza) / 2
    return valores[int(cola * repeticiones)], valores[int((1 - cola) * repeticiones) - 1]


def intervalo_diferencia(
    a: list[Resultado],
    b: list[Resultado],
    metrica: Callable[[Resultado], float],
    *,
    repeticiones: int = 2000,
    confianza: float = 0.95,
    semilla: int = 0,
) -> tuple[float, float]:
    """Intervalo de confianza de metrica(b) - metrica(a), con bootstrap emparejado.

    `a` y `b` son dos modelos evaluados sobre las mismas radiografías, en el
    mismo orden. En cada repetición se remuestrean las mismas radiografías para
    los dos: la dificultad de cada radiografía se cancela y queda solo la
    diferencia entre modelos. Si el intervalo contiene el 0, la diferencia no
    se distingue del azar del conjunto de validación.
    """
    if len(a) != len(b):
        raise ValueError("Los dos modelos deben evaluarse sobre las mismas radiografías")
    rng = random.Random(semilla)
    n = len(a)
    diferencias = []
    for _ in range(repeticiones):
        idx = [rng.randrange(n) for _ in range(n)]
        suma_a = sum((a[i] for i in idx), Resultado())
        suma_b = sum((b[i] for i in idx), Resultado())
        diferencias.append(metrica(suma_b) - metrica(suma_a))
    diferencias.sort()
    cola = (1 - confianza) / 2
    return diferencias[int(cola * repeticiones)], diferencias[int((1 - cola) * repeticiones) - 1]


# --- Hallazgos (patología) ---------------------------------------------------


def ap_coco(
    predicciones: list[dict], verdades: list[dict], clases: dict[int, str]
) -> dict[str, float]:
    """AP al estilo COCO, la métrica oficial del concurso DENTEX.

    Devuelve AP (IoU 0,50:0,95), AP50, AP75 y AR100 globales, y el AP de cada
    clase. Las cajas deben estar en la misma escala en predicción y verdad.
    """
    import contextlib
    import io

    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    imagenes, anotaciones, detecciones = [], [], []
    for i, (p, v) in enumerate(zip(predicciones, verdades, strict=True)):
        imagenes.append({"id": i})
        for caja, clase in zip(v["boxes"].tolist(), v["labels"].tolist()):
            x1, y1, x2, y2 = caja
            anotaciones.append({
                "id": len(anotaciones) + 1, "image_id": i, "category_id": clase,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })
        for caja, clase, score in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
            x1, y1, x2, y2 = caja
            detecciones.append({
                "image_id": i, "category_id": clase,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score,
            })

    silencio = contextlib.redirect_stdout(io.StringIO())
    with silencio:
        gt = COCO()
        gt.dataset = {"images": imagenes, "annotations": anotaciones,
                      "categories": [{"id": k, "name": n} for k, n in clases.items()]}
        gt.createIndex()
        dt = gt.loadRes(detecciones) if detecciones else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    resultado = {k: float(ev.stats[i]) for k, i in (("AP", 0), ("AP50", 1), ("AP75", 2), ("AR100", 8))}
    precision = ev.eval["precision"]  # [iou, recall, clase, área, maxdets]
    for k, (id_clase, nombre) in enumerate(clases.items()):
        p = precision[:, :, k, 0, -1]
        resultado[f"AP {nombre}"] = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
    return resultado


@dataclass
class Hallazgos:
    """Aciertos y fallos de un tipo de hallazgo, con un umbral de confianza."""

    reales: int = 0
    encontrados: int = 0  # reales con una predicción de su clase encima
    falsas_alarmas: int = 0
    radiografias: int = 0

    def __add__(self, otro: Hallazgos) -> Hallazgos:
        return Hallazgos(*(a + b for a, b in zip(asdict(self).values(), asdict(otro).values())))

    @property
    def sensibilidad(self) -> float:
        return self.encontrados / max(1, self.reales)

    @property
    def precision(self) -> float:
        return self.encontrados / max(1, self.encontrados + self.falsas_alarmas)

    @property
    def falsas_por_radiografia(self) -> float:
        return self.falsas_alarmas / max(1, self.radiografias)


def hallazgos_por_clase(
    predicciones: list[dict],
    verdades: list[dict],
    clases: dict[int, str],
    *,
    umbral_score: float = 0.5,
    umbral_iou: float = 0.5,
) -> dict[str, list[Hallazgos]]:
    """Por clase, una lista con el resultado de cada radiografía (para bootstrap)."""
    salida: dict[str, list[Hallazgos]] = {n: [] for n in clases.values()}
    for p, v in zip(predicciones, verdades, strict=True):
        for clase, nombre in clases.items():
            mp = (p["labels"] == clase) & (p["scores"] >= umbral_score)
            orden = torch.argsort(p["scores"][mp], descending=True)
            cajas_p = p["boxes"][mp][orden].cpu()
            cajas_r = v["boxes"][v["labels"] == clase].cpu()
            pares = emparejar(cajas_p, cajas_r, umbral_iou)
            salida[nombre].append(Hallazgos(
                reales=len(cajas_r), encontrados=len(pares),
                falsas_alarmas=len(cajas_p) - len(pares), radiografias=1,
            ))
    return salida
