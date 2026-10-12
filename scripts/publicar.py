"""Prepara la demo web y los pesos para Hugging Face y, con --subir, los publica.

Uso:
    uv run --group demo python scripts/publicar.py --ejemplos   # regenera demo/ejemplos/
    uv run --group demo python scripts/publicar.py              # prepara publicar/
    uv run --group demo python scripts/publicar.py --subir      # y lo sube (antes: hf auth login)

publicar/modelo  repositorio de pesos: safetensors, ajustes y ficha del modelo
publicar/space   Space de Gradio: demo/app.py, el paquete dentalvision, los
                 ejemplos, las dependencias (CPU) y la ficha del Space
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

RAIZ = Path(__file__).resolve().parents[1]
DEMO = RAIZ / "demo"
SALIDA = RAIZ / "publicar"
REPO = "dental-vision"

# Carpeta publicada -> (ejecución en modelos/, ajustes elegidos en validación)
MODELOS = {
    "numeracion": ("base", ["postproceso.json", "duda.json"]),
    "patologia": ("patologia", ["umbral.json"]),
}


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--ejemplos", action="store_true",
                   help="regenera las radiografías de ejemplo (necesita el dataset)")
    p.add_argument("--subir", action="store_true", help="publica en Hugging Face")
    return p.parse_args()


def ejemplos() -> None:
    """Las radiografías de las figuras del README, elegidas con las mismas reglas
    fijas. Ninguno de los dos modelos las vio al entrenar.

    Se guardan en gris y a la resolución de trabajo: el modelo recibe
    exactamente lo mismo que con el fichero original, que pesa diez veces más.
    """
    from dentalvision.datos import ANCHO, cargar_imagen
    from dentalvision.dentex import Subconjunto
    from dentalvision.particiones import particion_de

    sub, particion = particion_de("numeracion")
    errores = json.loads((RAIZ / "modelos/base/resultados_prueba.json").read_text())
    errores = errores["orden"]["errores_por_imagen"]
    valores = sorted(errores.values())
    mediana, p90 = valores[(len(valores) - 1) // 2], valores[int(0.9 * (len(valores) - 1))]
    rutas = {
        "tipica": sub.dir_imagenes / next(f for f in errores if errores[f] == mediana),
        "dificil": sub.dir_imagenes / next(f for f in errores if errores[f] == p90),
    }

    # Hallazgos: la validación oficial, que el modelo de patología no vio; de
    # ella, las que tampoco vio el de numeración. La del número mediano de
    # hallazgos anotados, la primera por nombre.
    vistas = {sub.huella(r) for r in particion.entrenamiento + particion.validacion}
    oficial = Subconjunto("validacion_oficial")
    nuevas = sorted((r for r in oficial.radiografias if oficial.huella(r) not in vistas),
                    key=lambda r: r.fichero)
    cuentas = sorted(len(r.dientes) for r in nuevas)
    mediana = cuentas[(len(cuentas) - 1) // 2]
    rutas["hallazgos"] = oficial.ruta(next(r for r in nuevas if len(r.dientes) == mediana))

    destino = DEMO / "ejemplos"
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, ruta in rutas.items():
        img, _ = cargar_imagen(ruta, ANCHO)
        img.save(destino / f"{nombre}.png", optimize=True)
        print(f"{nombre}: {ruta.name} -> {destino / nombre}.png")


def exportar_pesos(destino: Path) -> None:
    """Pesos en safetensors (sin pickle) con sus ajustes, y la ficha del modelo."""
    for carpeta, (ejecucion, ajustes) in MODELOS.items():
        origen = RAIZ / "modelos" / ejecucion
        (destino / carpeta).mkdir(parents=True)
        estado = torch.load(origen / "mejor.pt", map_location="cpu", weights_only=True)
        fichero = destino / carpeta / "mejor.safetensors"
        save_file(estado, fichero, metadata={"ejecucion": ejecucion})
        releido = load_file(fichero)
        if releido.keys() != estado.keys() or not all(torch.equal(releido[k], v) for k, v in estado.items()):
            raise RuntimeError(f"Los pesos exportados de {ejecucion} no coinciden con los originales")
        for f in ["config.json", *ajustes]:
            shutil.copy2(origen / f, destino / carpeta / f)
        print(f"{carpeta}: {fichero.stat().st_size / 1e6:.0f} MB, idénticos a modelos/{ejecucion}/mejor.pt")
    shutil.copy2(DEMO / "modelo.md", destino / "README.md")


def montar_space(destino: Path) -> None:
    destino.mkdir(parents=True)
    for f in ("app.py", "requirements.txt", "packages.txt"):
        shutil.copy2(DEMO / f, destino / f)
    shutil.copy2(DEMO / "space.md", destino / "README.md")
    shutil.copytree(DEMO / "ejemplos", destino / "ejemplos")
    shutil.copytree(RAIZ / "src" / "dentalvision", destino / "dentalvision",
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"space: {sum(1 for f in destino.rglob('*') if f.is_file())} ficheros")


def subir() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    repo = f"{api.whoami()['name']}/{REPO}"
    # El Space descarga los pesos del repositorio de modelo: primero los pesos.
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=SALIDA / "modelo", delete_patterns="*",
                      commit_message="Pesos y ficha del modelo")
    api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(repo, "DENTALVISION_REPO", repo)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=SALIDA / "space",
                      delete_patterns="*", commit_message="Demo web")
    print(f"Modelo: https://huggingface.co/{repo}")
    print(f"Demo:   https://huggingface.co/spaces/{repo}")


def main() -> int:
    a = argumentos()
    if a.ejemplos:
        ejemplos()
        return 0
    shutil.rmtree(SALIDA, ignore_errors=True)
    exportar_pesos(SALIDA / "modelo")
    montar_space(SALIDA / "space")
    if a.subir:
        subir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
