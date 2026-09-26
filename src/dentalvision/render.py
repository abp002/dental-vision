"""Dibujo de anotaciones sobre las radiografias.

Este modulo no entrena nada, y aun asi es de los mas importantes del proyecto:
hasta que no ves una panoramica con su odontograma encima, no sabes si el
mapeo de cuadrantes es correcto, si los poligonos estan desplazados o si has
confundido los ejes. Los errores de datos se ven, no se deducen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .dentex import ARCADAS, Radiografia

# Un color por cuadrante. Sobre una radiografia (gris) los tonos saturados
# destacan sin tapar la anatomia.
COLOR_CUADRANTE = {
    1: (56, 217, 214),   # turquesa
    2: (255, 193, 68),   # ambar
    3: (255, 111, 145),  # rosa
    4: (126, 217, 87),   # verde
}
COLOR_PATOLOGIA = (255, 71, 87)  # rojo, solo para hallazgos
COLOR_ERROR = (255, 71, 87)  # rojo: fallos del modelo frente a la anotacion
COLOR_FONDO = (12, 14, 18)

_FUENTES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Linux. Liberation Sans tiene las mismas metricas que Arial.
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
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


# --- Predicciones del modelo ------------------------------------------------


@dataclass
class Pieza:
    """Una pieza lista para dibujar sobre la radiografia.

    estado: "bien" (numero correcto), "mal" (encontrada pero mal numerada;
    `real` es el numero verdadero), "sobra" (deteccion sin pieza real),
    "falta" (pieza real no detectada; `caja` es la de la anotacion) o None si
    no hay anotacion con la que comparar.
    """

    fdi: int
    caja: tuple[float, float, float, float]  # x1, y1, x2, y2
    mascara: np.ndarray | None = None  # bool, del tamano de la imagen
    estado: str | None = None
    real: int | None = None


ABREVIATURA = {"Caries": "C", "Deep Caries": "CP", "Periapical Lesion": "LP", "Impacted": "I"}
COLOR_HALLAZGO = (255, 92, 92)


@dataclass
class Hallazgo:
    """Un hallazgo predicho: caja, tipo y el diente al que se asigna."""

    patologia: str  # nombre DENTEX: Caries, Deep Caries, Periapical Lesion, Impacted
    caja: tuple[float, float, float, float]
    confianza: float
    fdi: int | None = None


def _pintar_mascara(capa, mascara, color, grosor) -> None:
    m = Image.fromarray(mascara.astype(np.uint8) * 255, "L")
    capa.paste(Image.new("RGBA", capa.size, (*color, 46)), (0, 0), m)
    lado = 2 * grosor + 1
    borde = ImageChops.subtract(m.filter(ImageFilter.MaxFilter(lado)), m)
    capa.paste(Image.new("RGBA", capa.size, (*color, 235)), (0, 0), borde)


def dibujar_prediccion(
    imagen: Image.Image,
    piezas: list[Pieza],
    *,
    hallazgos: list[Hallazgo] = (),
    ancho_salida: int | None = 1600,
) -> Image.Image:
    """Radiografia con las piezas predichas: mascara y numero FDI.

    Los fallos van en rojo: "14>15" es una pieza que el modelo numero 14 y es
    la 15; un recuadro rojo sin relleno es una pieza que no encontro.
    """
    base = imagen.convert("RGB")
    capa = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dib = ImageDraw.Draw(capa)
    grosor = max(2, base.width // 700)
    fuente = _fuente(max(16, base.width // 70))

    for p in piezas:
        color = COLOR_ERROR if p.estado in ("mal", "sobra", "falta") else COLOR_CUADRANTE[p.fdi // 10]
        if p.estado == "falta":
            dib.rectangle(p.caja, outline=(*COLOR_ERROR, 255), width=grosor * 2)
        elif p.mascara is not None:
            _pintar_mascara(capa, p.mascara, color, grosor)
        else:
            dib.rectangle(p.caja, outline=(*color, 235), width=grosor)

    for p in piezas:
        color = COLOR_ERROR if p.estado in ("mal", "sobra", "falta") else COLOR_CUADRANTE[p.fdi // 10]
        texto = f"{p.fdi}>{p.real}" if p.estado == "mal" else str(p.fdi)
        x1, y1, x2, y2 = p.caja
        _etiqueta(dib, ((x1 + x2) / 2, (y1 + y2) / 2), texto, color, fuente)

    fuente_h = _fuente(max(14, base.width // 90))
    for h in hallazgos:
        x1, y1, x2, y2 = h.caja
        dib.rectangle(h.caja, outline=(*COLOR_HALLAZGO, 255), width=grosor)
        _etiqueta(dib, ((x1 + x2) / 2, y1 - fuente_h.size * 0.6), ABREVIATURA[h.patologia],
                  COLOR_HALLAZGO, fuente_h)

    fusion = Image.alpha_composite(base.convert("RGBA"), capa).convert("RGB")
    if ancho_salida and fusion.width != ancho_salida:
        alto = round(fusion.height * ancho_salida / fusion.width)
        fusion = fusion.resize((ancho_salida, alto), Image.LANCZOS)
    return fusion


def dibujar_odontograma(
    presentes: set[int],
    *,
    errores: set[int] = frozenset(),
    ocultas: set[int] = frozenset(),
    hallazgos: dict[int, list[str]] | None = None,
    ancho: int = 1600,
    titulo: str = "Odontograma",
) -> Image.Image:
    """Ficha de 32 casillas en la misma disposicion que la radiografia.

    Casilla de color: pieza presente. Gris tachada: ausente. Borde rojo: la
    casilla no coincide con la anotacion (sobra, falta o esta mal numerada).
    """
    margen = round(ancho * 0.03)
    hueco = max(4, ancho // 250)
    linea_media = 4 * hueco
    celda = (ancho - 2 * margen - 14 * hueco - linea_media) / 16
    alto_celda = round(celda * 0.8)
    fuente = _fuente(round(celda * 0.34))
    fuente_titulo = _fuente(round(celda * 0.3))
    cabecera = round(celda * 0.7)
    alto = cabecera + 2 * alto_celda + hueco * 3 + margen

    img = Image.new("RGB", (ancho, alto), COLOR_FONDO)
    dib = ImageDraw.Draw(img)
    dib.text((margen, cabecera * 0.3), titulo, font=fuente_titulo, fill=(235, 237, 240))

    leyenda = [("presente", COLOR_CUADRANTE[1], None), ("ausente", (44, 48, 56), None)]
    if hallazgos:
        leyenda.append(("hallazgo (C caries, CP profunda, LP lesión, I impactado)", COLOR_HALLAZGO, None))
    if errores:
        leyenda.append(("no coincide con la anotación", (44, 48, 56), COLOR_ERROR))
    x = ancho - margen
    for texto, relleno, borde in reversed(leyenda):
        w = dib.textlength(texto, font=fuente_titulo)
        x -= w
        dib.text((x, cabecera * 0.3), texto, font=fuente_titulo, fill=(170, 175, 185))
        lado = round(celda * 0.28)
        y = cabecera * 0.3 + (fuente_titulo.size - lado) / 2 + 2
        dib.rounded_rectangle((x - lado - 8, y, x - 8, y + lado), radius=3, fill=relleno,
                              outline=borde, width=3 if borde else 0)
        x -= lado + 8 + 2 * margen // 3

    for fila, arcada in enumerate(("superior", "inferior")):
        y = cabecera + fila * (alto_celda + hueco * 2)
        for k, fdi in enumerate(ARCADAS[arcada]):
            x = margen + k * (celda + hueco) + (linea_media if k >= 8 else 0)
            caja = (x, y, x + celda, y + alto_celda)
            presente = fdi in presentes
            relleno = COLOR_CUADRANTE[fdi // 10] if presente else (44, 48, 56)
            dib.rounded_rectangle(caja, radius=round(celda * 0.12), fill=relleno)
            if fdi in ocultas:
                # Casilla aún sin decidir (animación): ni presente ni tachada.
                dib.rounded_rectangle(caja, radius=round(celda * 0.12), fill=(28, 31, 38))
                continue
            if not presente:
                dib.line((x + celda * 0.2, y + alto_celda * 0.8, x + celda * 0.8, y + alto_celda * 0.2),
                         fill=(80, 86, 96), width=max(2, hueco // 2))
            if fdi in errores:
                dib.rounded_rectangle(caja, radius=round(celda * 0.12), outline=COLOR_ERROR,
                                      width=max(3, hueco))
            texto = str(fdi)
            izq, arr, der, aba = dib.textbbox((0, 0), texto, font=fuente)
            dib.text((x + (celda - (der - izq)) / 2 - izq, y + (alto_celda - (aba - arr)) / 2 - arr),
                     texto, font=fuente, fill=COLOR_FONDO if presente else (120, 126, 136))
            if hallazgos and hallazgos.get(fdi):
                marca = " ".join(ABREVIATURA[h] for h in sorted(set(hallazgos[fdi])))
                fuente_m = _fuente(round(celda * 0.2))
                w = dib.textlength(marca, font=fuente_m) + celda * 0.12
                caja_m = (x + celda - w - 3, y + 3, x + celda - 3, y + 3 + fuente_m.size * 1.3)
                dib.rounded_rectangle(caja_m, radius=4, fill=COLOR_HALLAZGO)
                dib.text(((caja_m[0] + caja_m[2]) / 2, (caja_m[1] + caja_m[3]) / 2), marca,
                         font=fuente_m, fill=(255, 255, 255), anchor="mm")
    return img


def apilar(*imagenes: Image.Image) -> Image.Image:
    """Une imagenes del mismo ancho una debajo de otra."""
    ancho = max(i.width for i in imagenes)
    salida = Image.new("RGB", (ancho, sum(i.height for i in imagenes)), COLOR_FONDO)
    y = 0
    for i in imagenes:
        salida.paste(i, (0, y))
        y += i.height
    return salida
