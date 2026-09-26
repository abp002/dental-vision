"""Detector de piezas dentales.

Mask R-CNN de torchvision (licencia BSD) con backbone ResNet-50 FPN preentrenado
en COCO. La alternativa cómoda del mercado, Ultralytics YOLO, es AGPL-3.0: si
este proyecto llegara a ofrecerse como servicio, esa licencia obligaría a
publicar todo el código del servicio o a comprar licencia comercial. La decisión
se toma ahora, no cuando haya un cliente.

Transfer learning: los pesos de COCO no saben nada de dientes, pero sí saben
detectar bordes, texturas y formas. Reaprender eso desde cero con 432 imágenes
sería imposible; reaprovecharlo y sustituir solo las cabezas de clasificación
funciona con datasets pequeños.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from .datos import ANCHO, N_CLASES


def dispositivo() -> torch.device:
    """CUDA si hay GPU NVIDIA; si no, CPU.

    MPS no se elige solo: en un Apple M4 resultó mucho más lento que la CPU con
    este modelo (ver README). Se puede forzar con --dispositivo mps.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crear(
    n_clases: int = N_CLASES,
    preentrenado: bool = True,
    capas_backbone: int = 3,
    ancho: int = ANCHO,
    rois: int = 128,
    congelar_bn: bool = False,
):
    """
    `capas_backbone` (0..5) es cuántos bloques del ResNet se reentrenan. Menos
    capas = entrenamiento más rápido y menos riesgo de sobreajuste con pocos
    datos, a costa de adaptarse peor a un dominio muy distinto de COCO. Una
    radiografía en escala de grises es bastante distinta de una foto, así que
    congelarlo entero (0) no funciona bien aquí.

    `ancho` tiene que coincidir con el del Dataset: es la resolución a la que
    trabaja la red.
    """
    modelo = maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if preentrenado else None,
        weights_backbone=None,
        trainable_backbone_layers=capas_backbone if preentrenado else None,
        # torchvision reescala cada imagen por dentro para que el lado corto
        # mida min_size sin que el largo pase de max_size (800 y 1333 por
        # defecto): una panorámica acabaría siempre a ~1333 px de ancho, fuera
        # cual fuera el tamaño que le diera el Dataset. Con los dos límites
        # iguales al ancho, el reescalado interno no hace nada y la resolución
        # la decide solo `ancho`.
        min_size=ancho,
        max_size=ancho,
        # Los valores por defecto (512 regiones muestreadas, 2.000 propuestas del
        # RPN) estan pensados para COCO, donde los objetos aparecen en cualquier
        # sitio y en cualquier numero. Aqui hay como mucho 32 dientes, siempre en
        # una banda central. Bajar a 128/500 acelera un 32%; el efecto en la
        # precision se mide en el README.
        box_batch_size_per_image=rois,
        rpn_post_nms_top_n_train=500,
        # box_detections_per_img se queda en 100 aunque una boca tenga 32
        # piezas: el NMS es por clase, así que una misma pieza puede salir como
        # 16 y como 17 a la vez, y el post-procesado necesita esos candidatos.
    )

    # Cabeza de clasificación de cajas: 91 clases de COCO -> 33 (32 dientes + fondo)
    ent = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(ent, n_clases)

    # Cabeza de máscaras
    ent_m = modelo.roi_heads.mask_predictor.conv5_mask.in_channels
    modelo.roi_heads.mask_predictor = MaskRCNNPredictor(ent_m, 256, n_clases)

    if congelar_bn:
        # La v2 usa BatchNorm normal en el backbone. Con lotes de 2 imágenes sus
        # estadísticas se recalculan con muy poca muestra, también en las capas
        # "congeladas" (requires_grad no impide actualizar running_mean/var).
        _congelar_bn(modelo.backbone.body)

    return modelo


def _congelar_bn(modulo: nn.Module) -> None:
    for nombre, hijo in modulo.named_children():
        if isinstance(hijo, nn.BatchNorm2d):
            fija = FrozenBatchNorm2d(hijo.num_features, eps=hijo.eps)
            with torch.no_grad():
                fija.weight.copy_(hijo.weight)
                fija.bias.copy_(hijo.bias)
                fija.running_mean.copy_(hijo.running_mean)
                fija.running_var.copy_(hijo.running_var)
            setattr(modulo, nombre, fija)
        else:
            _congelar_bn(hijo)


@contextmanager
def sin_mascaras(modelo):
    """Apaga la rama de máscaras durante la inferencia.

    Las métricas solo usan cajas y etiquetas, y cada máscara se devuelve pegada
    a tamaño completo en float32: con 100 detecciones son cientos de MB por
    radiografía que no sirven para nada.
    """
    cabezas = modelo.roi_heads
    guardada = cabezas.mask_roi_pool
    cabezas.mask_roi_pool = None  # RoIHeads.has_mask() pasa a ser False
    try:
        yield modelo
    finally:
        cabezas.mask_roi_pool = guardada


def cargar(carpeta: Path, cual: str = "mejor", dev: torch.device | None = None):
    """Reconstruye un modelo entrenado a partir de su carpeta en modelos/."""
    config = json.loads((carpeta / "config.json").read_text())
    from .datos import TAREAS

    modelo = crear(
        TAREAS[config.get("tarea", "numeracion")].n_clases,
        preentrenado=False,
        ancho=config["ancho"],
        rois=config["rois"],
        congelar_bn=config["congelar_bn"],
    )
    estado = torch.load(carpeta / f"{cual}.pt", map_location="cpu", weights_only=True)
    modelo.load_state_dict(estado)
    return modelo.to(dev or dispositivo()).eval(), config
