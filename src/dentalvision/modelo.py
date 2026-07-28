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

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .datos import N_CLASES


def dispositivo() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crear(
    n_clases: int = N_CLASES,
    preentrenado: bool = True,
    capas_backbone: int = 3,
):
    """
    `capas_backbone` (0..5) es cuántos bloques del ResNet se reentrenan. Menos
    capas = entrenamiento más rápido y menos riesgo de sobreajuste con pocos
    datos, a costa de adaptarse peor a un dominio muy distinto de COCO. Una
    radiografía en escala de grises es bastante distinta de una foto, así que
    congelarlo entero (0) no funciona bien aquí.
    """
    modelo = maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if preentrenado else None,
        # Una boca tiene hasta 32 piezas. El valor por defecto (100) desperdicia
        # cómputo en detecciones que la anatomía descarta de antemano.
        box_detections_per_img=40,
        trainable_backbone_layers=capas_backbone,
        # Los valores por defecto (512 regiones muestreadas, 2.000 propuestas del
        # RPN) estan pensados para COCO, donde los objetos aparecen en cualquier
        # sitio y en cualquier numero. Aqui hay como mucho 32 dientes, siempre en
        # una banda central. Medido: bajar a 128/500 acelera un 32% y por debajo
        # de 128 ya no se gana nada.
        box_batch_size_per_image=128,
        rpn_post_nms_top_n_train=500,
    )

    # Cabeza de clasificación de cajas: 91 clases de COCO -> 33 (32 dientes + fondo)
    ent = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(ent, n_clases)

    # Cabeza de máscaras
    ent_m = modelo.roi_heads.mask_predictor.conv5_mask.in_channels
    modelo.roi_heads.mask_predictor = MaskRCNNPredictor(ent_m, 256, n_clases)

    return modelo
