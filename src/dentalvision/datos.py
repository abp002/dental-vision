"""Dataset de PyTorch para detección y segmentación de piezas dentales.

Formato de salida: el que espera torchvision.models.detection — una imagen como
tensor float [0,1] y un diccionario con `boxes` (xyxy), `labels` y `masks`.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .dentex import Radiografia, Subconjunto

# 32 clases FDI. La 0 queda reservada al fondo, como exige torchvision.
FDIS = [q * 10 + p for q in (1, 2, 3, 4) for p in range(1, 9)]
FDI_A_CLASE = {f: i + 1 for i, f in enumerate(FDIS)}
CLASE_A_FDI = {i + 1: f for i, f in enumerate(FDIS)}
N_CLASES = len(FDIS) + 1

# Al voltear la imagen en horizontal, la derecha del paciente pasa a ser la
# izquierda. Los cuadrantes 1 y 2 se intercambian, y el 3 con el 4.
ESPEJO_CUADRANTE = {1: 2, 2: 1, 3: 4, 4: 3}


def fdi_espejado(fdi: int) -> int:
    return ESPEJO_CUADRANTE[fdi // 10] * 10 + fdi % 10


class DientesDataset(Dataset):
    """Radiografías panorámicas con sus piezas anotadas.

    `ancho` redimensiona la imagen manteniendo la proporción: las panorámicas
    originales rondan los 2.900 px y no caben en memoria de GPU a batch
    razonable.
    """

    def __init__(
        self,
        sub: Subconjunto,
        radiografias: list[Radiografia],
        *,
        ancho: int = 1024,
        aumentar: bool = False,
        con_mascaras: bool = True,
    ):
        self.sub = sub
        self.items = radiografias
        self.ancho = ancho
        self.aumentar = aumentar
        self.con_mascaras = con_mascaras

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        r = self.items[i]
        img = Image.open(self.sub.ruta(r)).convert("L")

        escala = self.ancho / img.width
        alto = round(img.height * escala)
        img = img.resize((self.ancho, alto), Image.BILINEAR)

        cajas, etiquetas, poligonos = [], [], []
        for d in r.dientes:
            x, y, w, h = (v * escala for v in d.bbox)
            if w < 2 or h < 2:
                continue  # caja degenerada: torchvision la rechazaria
            cajas.append([x, y, x + w, y + h])
            etiquetas.append(FDI_A_CLASE[d.fdi])
            poligonos.append([v * escala for v in d.poligono])

        volteada = self.aumentar and random.random() < 0.5
        if volteada:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            W = self.ancho
            cajas = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in cajas]
            # El remapeo de cuadrantes es obligatorio: sin el, el modelo aprende
            # que el mismo diente es a veces 16 y a veces 26.
            etiquetas = [FDI_A_CLASE[fdi_espejado(CLASE_A_FDI[c])] for c in etiquetas]
            poligonos = [
                [W - v if j % 2 == 0 else v for j, v in enumerate(p)] for p in poligonos
            ]

        tensor = torch.from_numpy(
            np.asarray(img, dtype="float32") / 255.0
        )
        if self.aumentar:
            # Variación de brillo y contraste: los equipos de rayos de distinta
            # marca producen exposiciones distintas, y el modelo debe aguantarlo.
            tensor = (tensor - 0.5) * random.uniform(0.85, 1.15) + 0.5
            tensor = (tensor + random.uniform(-0.08, 0.08)).clamp(0, 1)
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # el backbone espera 3 canales

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
