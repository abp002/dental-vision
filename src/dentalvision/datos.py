"""Dataset de PyTorch para detección y segmentación de piezas dentales.

Formato de salida: el que espera torchvision.models.detection — una imagen como
tensor float [0,1] y un diccionario con `boxes` (xyxy), `labels` y `masks`.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .dentex import PATOLOGIAS, Diente, Radiografia, Subconjunto

# 32 clases FDI. La 0 queda reservada al fondo, como exige torchvision.
FDIS = [q * 10 + p for q in (1, 2, 3, 4) for p in range(1, 9)]
FDI_A_CLASE = {f: i + 1 for i, f in enumerate(FDIS)}
CLASE_A_FDI = {i + 1: f for i, f in enumerate(FDIS)}
N_CLASES = len(FDIS) + 1

# Segunda tarea: los 4 hallazgos de DENTEX, sobre la caja del diente afectado.
PATOLOGIA_A_CLASE = {p: i + 1 for i, p in enumerate(PATOLOGIAS)}
CLASE_A_PATOLOGIA = {i + 1: p for i, p in enumerate(PATOLOGIAS)}


@dataclass(frozen=True)
class Tarea:
    """Qué se aprende de cada diente anotado: su número o su hallazgo."""

    nombre: str
    n_clases: int  # incluido el fondo
    etiqueta: Callable[[Diente], int | None]
    # Qué pasa con la etiqueta al voltear la imagen en horizontal.
    espejo: Callable[[int], int]


def _clase_fdi(d: Diente) -> int:
    return FDI_A_CLASE[d.fdi]


def _clase_patologia(d: Diente) -> int | None:
    return PATOLOGIA_A_CLASE.get(d.patologia)


def _espejo_fdi(clase: int) -> int:
    return FDI_A_CLASE[fdi_espejado(CLASE_A_FDI[clase])]


def _sin_cambio(clase: int) -> int:
    # Una caries en el espejo sigue siendo una caries.
    return clase


# Resolución de trabajo (ancho en px). Es el tamaño al que torchvision llevaría
# de todos modos una panorámica con su configuración por defecto; aquí se fija
# explícitamente y el modelo se configura para no reescalar (ver modelo.py).
ANCHO = 1333

# Al voltear la imagen en horizontal, la derecha del paciente pasa a ser la
# izquierda. Los cuadrantes 1 y 2 se intercambian, y el 3 con el 4.
ESPEJO_CUADRANTE = {1: 2, 2: 1, 3: 4, 4: 3}


def fdi_espejado(fdi: int) -> int:
    return ESPEJO_CUADRANTE[fdi // 10] * 10 + fdi % 10


def cargar_imagen(ruta: Path | Image.Image, ancho: int = ANCHO) -> tuple[Image.Image, float]:
    """Radiografía en gris redimensionada a `ancho` px, y la escala aplicada."""
    img = (ruta if isinstance(ruta, Image.Image) else Image.open(ruta)).convert("L")
    escala = ancho / img.width
    return img.resize((ancho, round(img.height * escala)), Image.BILINEAR), escala


def a_tensor(img: Image.Image) -> torch.Tensor:
    """Imagen en gris -> tensor float [0,1] de 3 canales (lo que espera el backbone)."""
    gris = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)
    return gris.unsqueeze(0).repeat(3, 1, 1)


def voltear(
    cajas: list[list[float]],
    etiquetas: list[int],
    poligonos: list[list[float]],
    ancho: int,
    espejo: Callable[[int], int] = _espejo_fdi,
) -> tuple[list[list[float]], list[int], list[list[float]]]:
    """Espejo horizontal de las anotaciones de una imagen de `ancho` px."""
    cajas = [[ancho - x2, y1, ancho - x1, y2] for x1, y1, x2, y2 in cajas]
    # En numeración el remapeo de cuadrantes es obligatorio: sin él, el modelo
    # aprende que el mismo diente es a veces 16 y a veces 26.
    etiquetas = [espejo(c) for c in etiquetas]
    poligonos = [
        [ancho - v if j % 2 == 0 else v for j, v in enumerate(p)] for p in poligonos
    ]
    return cajas, etiquetas, poligonos


class DientesDataset(Dataset):
    """Radiografías panorámicas con sus piezas anotadas.

    `ancho` redimensiona la imagen manteniendo la proporción: las panorámicas
    originales rondan los 2.900 px, y rasterizar ~28 máscaras a ese tamaño por
    imagen es caro en CPU y en memoria. El modelo trabaja exactamente a este
    ancho.
    """

    def __init__(
        self,
        sub: Subconjunto,
        radiografias: list[Radiografia],
        *,
        ancho: int = ANCHO,
        aumentar: bool = False,
        con_mascaras: bool = True,
        tarea: Tarea | None = None,
    ):
        self.tarea = tarea or TAREAS["numeracion"]
        self.sub = sub
        self.items = radiografias
        self.ancho = ancho
        self.aumentar = aumentar
        self.con_mascaras = con_mascaras

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        r = self.items[i]
        img, escala = cargar_imagen(self.sub.ruta(r), self.ancho)
        alto = img.height

        cajas, etiquetas, poligonos = [], [], []
        for d in r.dientes:
            clase = self.tarea.etiqueta(d)
            x, y, w, h = (v * escala for v in d.bbox)
            if clase is None or w < 2 or h < 2:
                continue  # sin etiqueta, o caja degenerada que torchvision rechazaría
            cajas.append([x, y, x + w, y + h])
            etiquetas.append(clase)
            poligonos.append([v * escala for v in d.poligono])

        if self.aumentar and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            cajas, etiquetas, poligonos = voltear(
                cajas, etiquetas, poligonos, self.ancho, self.tarea.espejo
            )

        tensor = a_tensor(img)
        if self.aumentar:
            # Variación de brillo y contraste: los equipos de rayos de distinta
            # marca producen exposiciones distintas, y el modelo debe aguantarlo.
            tensor = (tensor - 0.5) * random.uniform(0.85, 1.15) + 0.5
            tensor = (tensor + random.uniform(-0.08, 0.08)).clamp(0, 1)

        objetivo = {
            "boxes": torch.tensor(cajas, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(etiquetas, dtype=torch.int64),
            "image_id": torch.tensor(r.id),
        }
        if self.con_mascaras:
            objetivo["masks"] = self._mascaras(poligonos, self.ancho, alto)
        return tensor, objetivo

    @staticmethod
    def _mascaras(poligonos: list[list[float]], w: int, h: int) -> torch.Tensor:
        capas = []
        for p in poligonos:
            m = Image.new("1", (w, h), 0)
            if len(p) >= 6:
                ImageDraw.Draw(m).polygon(list(zip(p[0::2], p[1::2])), fill=1)
            capas.append(torch.from_numpy(np.asarray(m, dtype="uint8")))
        if not capas:
            return torch.zeros((0, h, w), dtype=torch.uint8)
        return torch.stack(capas)


def colacion(lote):
    """Las imágenes de detección tienen distinto número de objetos: no se apilan."""
    return tuple(zip(*lote))


TAREAS = {
    "numeracion": Tarea("numeracion", N_CLASES, _clase_fdi, _espejo_fdi),
    "patologia": Tarea("patologia", len(PATOLOGIAS) + 1, _clase_patologia, _sin_cambio),
}
