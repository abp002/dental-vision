"""Dibujo de anotaciones sobre las radiografias.

Este modulo no entrena nada, y aun asi es de los mas importantes del proyecto:
hasta que no ves una panoramica con su odontograma encima, no sabes si el
mapeo de cuadrantes es correcto, si los poligonos estan desplazados o si has
confundido los ejes. Los errores de datos se ven, no se deducen.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .dentex import Diente, Radiografia

# Un color por cuadrante. Sobre una radiografia (gris) los tonos saturados
# destacan sin tapar la anatomia.
COLOR_CUADRANTE = {
    1: (56, 217, 214),   # turquesa
    2: (255, 193, 68),   # ambar
    3: (255, 111, 145),  # rosa
    4: (126, 217, 87),   # verde
}
COLOR_PATOLOGIA = (255, 71, 87)  # rojo, solo para hallazgos

_FUENTES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def _fuente(tam: int) -> ImageFont.FreeTypeFont:
    for ruta in _FUENTES:
        if Path(ruta).exists():
            try:
                return ImageFont.truetype(ruta, tam)
            except OSError:
                continue
    return ImageFont.load_default(tam)


def _etiqueta(
    dib: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    texto: str,
    color: tuple[int, int, int],
    fuente: ImageFont.FreeTypeFont,
) -> None:
    """Texto con cartela de fondo, para que se lea sobre cualquier gris."""
    x, y = xy
    izq, arr, der, aba = dib.textbbox((0, 0), texto, font=fuente)
    w, h = der - izq, aba - arr
    pad = max(3, h // 5)
    caja = (x - w / 2 - pad, y - h / 2 - pad, x + w / 2 + pad, y + h / 2 + pad)
    dib.rounded_rectangle(caja, radius=pad, fill=(12, 14, 18, 220))
    dib.text((x - w / 2 - izq, y - h / 2 - arr), texto, font=fuente, fill=color)


def dibujar(
    ruta_imagen: Path,
    radiografia: Radiografia,
    *,
    mostrar_fdi: bool = True,
    mostrar_patologia: bool = True,
    ancho_salida: int | None = 1600,
) -> Image.Image:
    """Devuelve la radiografia con sus piezas marcadas y numeradas."""
    base = Image.open(ruta_imagen).convert("RGB")
    capa = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dib = ImageDraw.Draw(capa)

    grosor = max(2, base.width // 900)
    fuente = _fuente(max(16, base.width // 85))

    for d in radiografia.dientes:
        color = COLOR_CUADRANTE[d.cuadrante]
        if d.poligono and len(d.poligono) >= 6:
            pts = list(zip(d.poligono[0::2], d.poligono[1::2]))
            dib.polygon(pts, fill=(*color, 46), outline=(*color, 235), width=grosor)
        else:
            x, y, w, h = d.bbox
            dib.rectangle((x, y, x + w, y + h), outline=(*color, 235), width=grosor)

    # Las etiquetas van en una segunda pasada para que ningun poligono
    # posterior las tape.
    if mostrar_fdi:
        for d in radiografia.dientes:
            _etiqueta(dib, d.centro, str(d.fdi), COLOR_CUADRANTE[d.cuadrante], fuente)

    if mostrar_patologia:
        for d in radiografia.dientes:
            if not d.patologia:
                continue
            x, y, w, h = d.bbox
            dib.rectangle(
                (x - grosor, y - grosor, x + w + grosor, y + h + grosor),
                outline=(*COLOR_PATOLOGIA, 255),
                width=grosor * 2,
            )
            _etiqueta(
                dib, (x + w / 2, y - h * 0.12), d.patologia, COLOR_PATOLOGIA, fuente
            )

    fusion = Image.alpha_composite(base.convert("RGBA"), capa).convert("RGB")

    if ancho_salida and fusion.width > ancho_salida:
        alto = round(fusion.height * ancho_salida / fusion.width)
        fusion = fusion.resize((ancho_salida, alto), Image.LANCZOS)
    return fusion
