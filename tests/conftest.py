import pytest
import torch

from dentalvision.datos import FDI_A_CLASE


@pytest.fixture
def pred():
    """Construye una salida de detector a partir de (x1, y1, x2, y2, fdi, score)."""

    def construir(*detecciones):
        return {
            "boxes": torch.tensor([d[:4] for d in detecciones], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([FDI_A_CLASE[d[4]] for d in detecciones], dtype=torch.int64),
            "scores": torch.tensor([d[5] for d in detecciones], dtype=torch.float32),
        }

    return construir
