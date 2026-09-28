import json

import torch
from safetensors.torch import save_file

from dentalvision.modelo import cargar, crear


def test_cargar_la_version_publicada_en_safetensors(tmp_path):
    # La carpeta publicada trae safetensors en lugar de .pt: mismos pesos.
    estado = crear(preentrenado=False).state_dict()
    save_file(estado, tmp_path / "mejor.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"ancho": 1333, "rois": 128, "congelar_bn": False}))

    modelo, _ = cargar(tmp_path, dev=torch.device("cpu"))

    cargado = modelo.state_dict()
    assert cargado.keys() == estado.keys()
    assert all(torch.equal(cargado[k], v) for k, v in estado.items())
