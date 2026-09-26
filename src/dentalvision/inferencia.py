"""Del fichero de una panorámica a sus piezas numeradas.

Junta carga del modelo, predicción y post-procesado para que los scripts (figura,
animación) y la demo web no repitan el mismo código.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from .datos import CLASE_A_FDI, CLASE_A_PATOLOGIA, a_tensor, cargar_imagen
from .evaluacion import iou
from .modelo import cargar, dispositivo, sin_mascaras
from .postproceso import postprocesar
from .render import Hallazgo, Pieza


@dataclass
class Prediccion:
    imagen: Image.Image  # radiografía a la resolución de trabajo
    escala: float  # resolución de trabajo / original
    piezas: list[Pieza]
    final: dict  # salida del post-procesado (boxes, labels, scores, indices)
    hallazgos: list[Hallazgo] = field(default_factory=list)

    def por_diente(self) -> dict[int, list[str]]:
        salida: dict[int, list[str]] = {}
        for h in self.hallazgos:
            if h.fdi is not None:
                salida.setdefault(h.fdi, []).append(h.patologia)
        return salida


class Odontografo:
    """Modelo entrenado más su post-procesado, listo para usar."""

    def __init__(self, carpeta: Path, dev: torch.device | None = None,
                 patologia: Path | None = None):
        self.dev = dev or dispositivo()
        self.modelo, self.config = cargar(carpeta, dev=self.dev)
        self.ajustes = json.loads((carpeta / "postproceso.json").read_text())["orden"]
        self.patologia = None
        if patologia is not None:
            self.patologia, _ = cargar(patologia, dev=self.dev)
            self.umbral_patologia = json.loads((patologia / "umbral.json").read_text())["umbral"]

    @torch.no_grad()
    def predecir(self, imagen: Path | Image.Image) -> Prediccion:
        img, escala = cargar_imagen(imagen, self.config["ancho"])
        salida = {k: v.cpu() for k, v in self.modelo([a_tensor(img).to(self.dev)])[0].items()}
        final = postprocesar(
            salida, "orden", umbral=self.ajustes["umbral"], kappa=self.ajustes["kappa"]
        )
        mascaras = salida["masks"][final["indices"], 0] > 0.5
        piezas = [
            Pieza(fdi=CLASE_A_FDI[int(c)], caja=tuple(b.tolist()), mascara=m.numpy())
            for c, b, m in zip(final["labels"], final["boxes"], mascaras)
        ]
        hallazgos = self._hallazgos(img, piezas) if self.patologia is not None else []
        return Prediccion(img, escala, piezas, final, hallazgos)

    def _hallazgos(self, img: Image.Image, piezas: list[Pieza]) -> list[Hallazgo]:
        with sin_mascaras(self.patologia):
            p = {k: v.cpu() for k, v in self.patologia([a_tensor(img).to(self.dev)])[0].items()}
        keep = p["scores"] >= self.umbral_patologia
        cajas, clases, scores = p["boxes"][keep], p["labels"][keep], p["scores"][keep]
        salida = []
        dientes = torch.tensor([pz.caja for pz in piezas]).reshape(-1, 4)
        solape = iou(cajas, dientes)
        for i in range(len(cajas)):
            fdi = None
            if len(piezas) and solape[i].max() > 0.3:
                fdi = piezas[int(solape[i].argmax())].fdi
            salida.append(Hallazgo(CLASE_A_PATOLOGIA[int(clases[i])], tuple(cajas[i].tolist()),
                                   float(scores[i]), fdi))
        return salida
