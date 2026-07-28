"""Métricas del odontograma.

La métrica estándar en detección es mAP. Es útil para comparar con la
literatura y no le dice absolutamente nada a un dentista.

Aquí se miden tres cosas que sí significan algo en la consulta:

  cobertura   de las piezas realmente presentes, cuántas encuentra
  numeración  de las que encuentra, a cuántas les pone el número FDI correcto
  perfectos   en cuántas radiografías el odontograma sale entero sin un fallo

La tercera es la que decide si la herramienta se usa. Un sistema con 97% de
numeración correcta suena magnífico hasta que caes en que, con 28 piezas por
boca, eso son casi 1 error por radiografía: el profesional tiene que revisarlo
todo igualmente y el ahorro de tiempo desaparece.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    @property
    def cobertura(self) -> float:
        return self.piezas_encontradas / max(1, self.piezas_reales)

    @property
    def numeracion(self) -> float:
        return self.numeracion_correcta / max(1, self.piezas_encontradas)

    @property
    def perfectas(self) -> float:
        return self.radiografias_perfectas / max(1, self.radiografias)

    @property
    def errores_por_radiografia(self) -> float:
        fallos = (self.piezas_reales - self.numeracion_correcta) + self.falsos_positivos
        return fallos / max(1, self.radiografias)

    def __str__(self) -> str:
        return (
            f"cobertura {self.cobertura:6.1%}   "
            f"numeracion {self.numeracion:6.1%}   "
            f"odontogramas perfectos {self.perfectas:6.1%}   "
            f"errores/radiografia {self.errores_por_radiografia:.2f}"
        )


def evaluar(
    predicciones: list[dict],
    verdades: list[dict],
    *,
    umbral_iou: float = 0.5,
    umbral_score: float = 0.5,
) -> Resultado:
    r = Resultado()
    for pred, real in zip(predicciones, verdades):
        r.radiografias += 1
        keep = pred["scores"] >= umbral_score
        cajas_p = pred["boxes"][keep].cpu()
        etiq_p = pred["labels"][keep].cpu()
        cajas_r = real["boxes"].cpu()
        etiq_r = real["labels"].cpu()

        r.piezas_reales += len(cajas_r)
        matriz = iou(cajas_r, cajas_p)

        usadas: set[int] = set()
        aciertos = 0
        for i in range(len(cajas_r)):
            if matriz.shape[1] == 0:
                break
            # Emparejamiento voraz por solapamiento. Suficiente aquí: los dientes
            # apenas se solapan entre si, asi que no hay ambiguedad real.
            orden = torch.argsort(matriz[i], descending=True)
            for j in orden.tolist():
                if j in usadas or matriz[i, j] < umbral_iou:
                    break
                usadas.add(j)
                r.piezas_encontradas += 1
                if etiq_p[j] == etiq_r[i]:
                    r.numeracion_correcta += 1
                    aciertos += 1
                break

        r.falsos_positivos += len(cajas_p) - len(usadas)
        if aciertos == len(cajas_r) and len(cajas_p) == len(cajas_r):
            r.radiografias_perfectas += 1
    return r
