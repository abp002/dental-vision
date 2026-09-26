"""Animación del proceso: la radiografía, las piezas apareciendo una a una en
orden de arcada y el odontograma rellenándose a la vez.

Uso:
    uv run python scripts/animacion.py --ejecucion base --fichero train_514.png --salida docs/demo.gif
    uv run python scripts/animacion.py --ejecucion base --imagen panoramica.png --salida demo.mp4

La extensión decide el formato: .gif, o .mp4 (necesita ffmpeg).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from dentalvision.dentex import ARCADAS, Subconjunto
from dentalvision.inferencia import Odontografo
from dentalvision.particiones import particionar
from dentalvision.render import (
    COLOR_FONDO,
    _fuente,
    apilar,
    dibujar_odontograma,
    dibujar_prediccion,
)

RAIZ = Path(__file__).resolve().parents[1]
ORDEN = ARCADAS["superior"] + ARCADAS["inferior"]


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--ejecucion", required=True)
    origen = p.add_mutually_exclusive_group(required=True)
    origen.add_argument("--imagen")
    origen.add_argument("--fichero", help="radiografía de la partición de prueba")
    p.add_argument("--salida", required=True)
    p.add_argument("--patologia", help="modelo de hallazgos en modelos/ (opcional)")
    p.add_argument("--ancho", type=int, default=960)
    p.add_argument("--fps", type=int, default=12)
    return p.parse_args()


def cabecera(ancho: int, texto: str) -> Image.Image:
    alto = round(ancho * 0.05)
    img = Image.new("RGB", (ancho, alto), COLOR_FONDO)
    dib = ImageDraw.Draw(img)
    fuente = _fuente(round(alto * 0.45))
    dib.text((round(ancho * 0.03), alto / 2), texto, font=fuente, fill=(235, 237, 240), anchor="lm")
    return img


def main() -> int:
    a = argumentos()
    if a.imagen:
        ruta = Path(a.imagen)
    else:
        sub = Subconjunto("enumeracion")
        ruta = sub.ruta(next(r for r in particionar(sub).prueba if r.fichero == a.fichero))

    pred = Odontografo(
        RAIZ / "modelos" / a.ejecucion,
        patologia=RAIZ / "modelos" / a.patologia if a.patologia else None,
    ).predecir(ruta)
    piezas = sorted(pred.piezas, key=lambda p: ORDEN.index(p.fdi))
    todas = set(ORDEN)

    def fotograma(k: int, texto: str, final: bool = False) -> Image.Image:
        vistas = piezas[:k]
        presentes = {p.fdi for p in vistas}
        # Cada hallazgo aparece con su diente; los que no tienen diente, al final.
        hallazgos = [h for h in pred.hallazgos if h.fdi in presentes or (final and h.fdi is None)]
        por_diente = {f: v for f, v in pred.por_diente().items() if f in presentes}
        # Mientras aparecen, solo se decide la casilla de las piezas ya vistas;
        # al final, las que no salieron quedan tachadas como ausentes.
        ultima = ORDEN.index(vistas[-1].fdi) if vistas else -1
        ocultas = set() if final else {f for f in todas if ORDEN.index(f) > ultima}
        return apilar(
            cabecera(a.ancho, texto),
            dibujar_prediccion(pred.imagen, vistas, hallazgos=hallazgos, ancho_salida=a.ancho),
            dibujar_odontograma(presentes, ocultas=ocultas, hallazgos=por_diente, ancho=a.ancho,
                                titulo="Odontograma generado"),
        )

    n = len(piezas)
    fotogramas: list[tuple[Image.Image, float]] = [
        (fotograma(0, "Radiografía panorámica"), 1.5)
    ]
    for k in range(1, n + 1):
        fotogramas.append((fotograma(k, f"Detectando y numerando piezas… {k}"), 1 / a.fps * 2))
    resumen = f"{n} piezas numeradas en notación FDI"
    if a.patologia:
        resumen += f" · {len(pred.hallazgos)} hallazgos"
    fotogramas.append((fotograma(n, resumen, final=True), 3.0))

    destino = Path(a.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.suffix == ".gif":
        cuadros = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f, _ in fotogramas]
        cuadros[0].save(destino, save_all=True, append_images=cuadros[1:], loop=0,
                        duration=[round(d * 1000) for _, d in fotogramas], optimize=True)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            i = 0
            for f, d in fotogramas:
                for _ in range(max(1, round(d * a.fps))):
                    f.save(Path(tmp) / f"{i:05d}.png")
                    i += 1
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                 "-i", f"{tmp}/%05d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(destino)],
                check=True,
            )
    print(f"Guardado en {destino} ({destino.stat().st_size / 1e6:.1f} MB, {len(fotogramas)} fotogramas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
