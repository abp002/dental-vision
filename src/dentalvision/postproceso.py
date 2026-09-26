"""Post-procesado con restricciones anatómicas.

El detector decide cada pieza por separado: no sabe que está mirando una boca.
Dos reglas que cualquier dentista aplica sin pensar corrigen parte de sus
errores:

1. UNICIDAD. Cada número FDI aparece como mucho una vez. El NMS de torchvision
   es por clase, así que dos propuestas sobre la misma pieza pueden salir como
   16 y como 17 a la vez.

2. ORDEN. En cada arcada las piezas van en fila. En la imagen, de izquierda a
   derecha: arriba 18..11 21..28, abajo 48..41 31..38 (la derecha del paciente
   aparece a la izquierda). Si faltan piezas quedan huecos, pero el orden no
   cambia.

Primero se agrupan las detecciones que caen sobre la misma pieza (candidatos);
después se asigna un número a cada candidato. Tres modos, para poder medir qué
aporta cada regla:

    ninguno    la salida del detector tal cual, filtrada por confianza
    unicidad   asignación óptima (algoritmo húngaro): cada FDI una sola vez
    orden      alineamiento por programación dinámica en cada arcada: unicidad
               y orden a la vez
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .datos import CLASE_A_FDI, FDI_A_CLASE, FDIS
from .dentex import ARCADAS
from .evaluacion import iou

MODOS = ("ninguno", "unicidad", "orden")


def arcada(fdi: int) -> str:
    return "superior" if fdi // 10 in (1, 2) else "inferior"


@dataclass
class Candidato:
    """Una pieza física: el grupo de detecciones que caen sobre ella."""

    indice: int  # detección más segura del grupo: de ella salen caja y máscara
    caja: torch.Tensor  # xyxy
    fdis: dict[int, float] = field(default_factory=dict)  # mejor puntuación por FDI

    @property
    def fdi(self) -> int:
        return max(self.fdis, key=self.fdis.get)

    @property
    def presencia(self) -> float:
        """Confianza en que ahí hay una pieza, sea cual sea su número.

        Si el detector duda entre 16 (0,45) y 17 (0,40), ninguna hipótesis pasa
        un umbral de 0,5, pero está claro que hay un diente.
        """
        return min(1.0, sum(self.fdis.values()))

    @property
    def x(self) -> float:
        return float(self.caja[0] + self.caja[2]) / 2


def agrupar(pred: dict, *, umbral_iou: float = 0.5) -> list[Candidato]:
    """Junta las detecciones que se solapan mucho: son la misma pieza con
    distintas hipótesis de número. Voraz por confianza."""
    scores = pred["scores"].cpu()
    cajas = pred["boxes"].cpu()
    etiquetas = pred["labels"].cpu()
    orden = torch.argsort(scores, descending=True).tolist()
    solape = iou(cajas, cajas)
    usada = [False] * len(orden)
    grupos: list[Candidato] = []
    for s in orden:
        if usada[s]:
            continue
        c = Candidato(indice=s, caja=cajas[s])
        for j in orden:
            if not usada[j] and solape[s, j] >= umbral_iou:
                usada[j] = True
                f = CLASE_A_FDI[int(etiquetas[j])]
                c.fdis[f] = max(c.fdis.get(f, 0.0), float(scores[j]))
        grupos.append(c)
    return grupos


def _afinidad(c: Candidato, secuencia: list[int], kappa: float) -> list[float]:
    """Cuánto encaja el candidato en cada posición de la arcada.

    La puntuación que el detector da a un número se extiende a las posiciones
    vecinas, atenuada por `kappa` en cada paso: si dice 16 con 0,9, el 15 y el
    17 reciben 0,9·kappa. Cuando el orden obliga a cambiar un número, se
    prefiere el vecino más cercano a lo que vio el detector.
    """
    pos = {f: k for k, f in enumerate(secuencia)}
    propias = [(pos[f], s) for f, s in c.fdis.items() if f in pos]
    return [
        max((s * kappa ** abs(k - p) for p, s in propias), default=0.0)
        for k in range(len(secuencia))
    ]


def _alinear(
    candidatos: list[Candidato], secuencia: list[int], kappa: float
) -> list[tuple[Candidato, int]]:
    """Asigna posiciones de la arcada a los candidatos sin repetir ninguna y
    respetando el orden izquierda-derecha. Maximiza la afinidad total.

    Es un alineamiento de dos secuencias, como en bioinformática: candidatos
    ordenados por x contra posiciones de la arcada. Se puede saltar un
    candidato (se descarta) o una posición (pieza ausente).
    """
    cs = sorted(candidatos, key=lambda c: c.x)
    n, m = len(cs), len(secuencia)
    a = [_afinidad(c, secuencia, kappa) for c in cs]
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = max(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1] + a[i - 1][j - 1])

    pares = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1][j - 1] > 0 and dp[i, j] == dp[i - 1, j - 1] + a[i - 1][j - 1]:
            pares.append((cs[i - 1], secuencia[j - 1]))
            i, j = i - 1, j - 1
        elif dp[i, j] == dp[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return pares[::-1]


def _unicos(candidatos: list[Candidato]) -> list[tuple[Candidato, int]]:
    if not candidatos:
        return []
    puntos = np.zeros((len(candidatos), len(FDIS)))
    for i, c in enumerate(candidatos):
        for f, s in c.fdis.items():
            puntos[i, FDIS.index(f)] = s
    filas, columnas = linear_sum_assignment(puntos, maximize=True)
    return [
        (candidatos[i], FDIS[j]) for i, j in zip(filas, columnas) if puntos[i, j] > 0
    ]


def _a_prediccion(pares: list[tuple[Candidato, int]]) -> dict:
    if not pares:
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros(0, dtype=torch.int64),
            "scores": torch.zeros(0),
            "indices": torch.zeros(0, dtype=torch.int64),
        }
    return {
        "boxes": torch.stack([c.caja for c, _ in pares]),
        "labels": torch.tensor([FDI_A_CLASE[f] for _, f in pares]),
        "scores": torch.tensor([c.presencia for c, _ in pares]),
        "indices": torch.tensor([c.indice for c, _ in pares]),
    }


def postprocesar(
    pred: dict,
    modo: str = "orden",
    *,
    umbral: float = 0.5,
    kappa: float = 0.2,
    umbral_iou: float = 0.5,
) -> dict:
    """Aplica las restricciones a la salida del detector de una radiografía.

    Devuelve un dict como el del detector (boxes, labels, scores) más
    `indices`: la detección original de cada pieza, para recuperar su máscara.
    """
    if modo not in MODOS:
        raise ValueError(f"Modo desconocido: {modo}. Opciones: {MODOS}")

    if modo == "ninguno":
        scores = pred["scores"].cpu()
        keep = torch.nonzero(scores >= umbral).flatten()
        return {
            "boxes": pred["boxes"].cpu()[keep],
            "labels": pred["labels"].cpu()[keep],
            "scores": scores[keep],
            "indices": keep,
        }

    candidatos = [
        c for c in agrupar(pred, umbral_iou=umbral_iou) if c.presencia >= umbral
    ]
    if modo == "unicidad":
        return _a_prediccion(_unicos(candidatos))

    pares = []
    for nombre, secuencia in ARCADAS.items():
        propios = [c for c in candidatos if arcada(c.fdi) == nombre]
        pares += _alinear(propios, secuencia, kappa)
    return _a_prediccion(pares)
