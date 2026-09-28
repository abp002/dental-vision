"""Demo web de dental-vision: de una radiografía panorámica a su odontograma.

Es la aplicación del Space de Hugging Face. En local:

    uv run --group demo python scripts/publicar.py     # exporta los pesos a publicar/modelo
    DENTALVISION_PESOS=publicar/modelo uv run --group demo python demo/app.py

Sin DENTALVISION_PESOS, descarga los pesos publicados en el Hub.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

from dentalvision.datos import FDIS
from dentalvision.dentex import NOMBRE_CUADRANTE, NOMBRE_POSICION
from dentalvision.inferencia import Odontografo, Prediccion
from dentalvision.render import (
    Hallazgo,
    apilar,
    dibujar_odontograma,
    dibujar_prediccion,
)

AQUI = Path(__file__).resolve().parent
REPO_PESOS = os.environ.get("DENTALVISION_REPO", "abp0040/dental-vision")
CODIGO = "https://github.com/abp002/dental-vision"

# Las radiografías de las figuras del README, elegidas con reglas fijas y que
# ninguno de los dos modelos vio al entrenar (ver scripts/publicar.py).
EJEMPLOS = {
    "tipica.png": "Radiografía típica",
    "hallazgos.png": "Con hallazgos",
    "dificil.png": "Caso difícil",
}

# El modelo de hallazgos se ejecuta una vez desde este umbral y el control solo
# filtra lo que se dibuja: moverlo no vuelve a pasar la radiografía por la red.
MINIMO = 0.1

# Nombre en singular y en plural, y código para el JSON.
HALLAZGOS = {
    "Impacted": ("diente impactado", "dientes impactados", "impactado"),
    "Caries": ("caries", "caries", "caries"),
    "Deep Caries": ("caries profunda", "caries profundas", "caries_profunda"),
    "Periapical Lesion": ("lesión periapical", "lesiones periapicales", "lesion_periapical"),
}


def cpus() -> int:
    """CPUs que el proceso puede usar de verdad.

    En un contenedor con cuota de CPU, como un Space gratuito (2 vCPU),
    os.cpu_count() cuenta las de la máquina entera y PyTorch abriría un hilo
    por cada una, peleándose todos por las dos CPUs de la cuota.
    """
    try:
        cuota, periodo = Path("/sys/fs/cgroup/cpu.max").read_text().split()  # cgroup v2
        if cuota != "max":
            return max(1, int(cuota) // int(periodo))
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def carpeta_pesos() -> Path:
    local = os.environ.get("DENTALVISION_PESOS")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO_PESOS))


torch.set_num_threads(cpus())
print(f"Hilos de PyTorch: {torch.get_num_threads()} (la máquina tiene {os.cpu_count()} CPUs)")
_pesos = carpeta_pesos()
ODONTOGRAFO = Odontografo(_pesos / "numeracion", patologia=_pesos / "patologia")
UMBRAL = ODONTOGRAFO.umbral_patologia  # elegido en validación


# --- Entrada ------------------------------------------------------------------


def abrir(ruta: str) -> tuple[Image.Image, list[str]]:
    """La radiografía en gris y avisos si no parece una panorámica."""
    try:
        img = ImageOps.exif_transpose(Image.open(ruta))
    except (OSError, Image.DecompressionBombError) as e:
        raise gr.Error("No he podido abrir la imagen. Prueba con un PNG o un JPG.") from e
    if img.height > img.width:
        # Todo se lleva a 1.333 px de ancho: una imagen alta dispararía el
        # tiempo y la memoria en la CPU compartida del Space, y no es una panorámica.
        raise gr.Error("La imagen es más alta que ancha. Una panorámica es unas dos veces "
                       "más ancha que alta.")
    avisos = []
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        # PNG de 16 bits, habitual al exportar de DICOM. convert("L") recortaría
        # todo lo que pase de 255 y dejaría la imagen en blanco: se reescala con
        # una ventana que ignora los píxeles extremos.
        v = np.asarray(img, dtype=np.float32)
        bajo, alto = np.percentile(v, (0.5, 99.5))
        img = Image.fromarray((255 * (v - bajo) / max(alto - bajo, 1.0)).clip(0, 255).astype(np.uint8))
    elif img.mode != "L":
        img = img.convert("RGB")
        muestra = np.asarray(img.resize((256, 128)), dtype=np.float32)
        if np.abs(muestra - muestra.mean(axis=2, keepdims=True)).mean() > 8:
            avisos.append("La imagen tiene color: parece una foto, no una radiografía.")
    if not 1.6 <= img.width / img.height <= 3.2:
        avisos.append("No tiene la forma de una panorámica, que es unas dos veces más ancha que alta.")
    return img.convert("L"), avisos


# --- Predicción, con memoria de las últimas -----------------------------------

_MEMORIA: OrderedDict[str, tuple[float, Prediccion, float]] = OrderedDict()
_CERROJO = threading.Lock()
CAPACIDAD, CADUCIDAD = 8, 30 * 60  # radiografías y segundos


def predecir(img: Image.Image) -> tuple[Prediccion, float]:
    """Predicción y segundos que tardó. Guarda las últimas radiografías un rato,
    para que mover el umbral de los hallazgos solo vuelva a dibujar."""
    clave = hashlib.sha1(img.tobytes()).hexdigest() + f"{img.size}"
    ahora = time.monotonic()
    with _CERROJO:
        for k in [k for k, (t, _, _) in _MEMORIA.items() if ahora - t > CADUCIDAD]:
            del _MEMORIA[k]
        if clave in _MEMORIA:
            _MEMORIA.move_to_end(clave)
            return _MEMORIA[clave][1:]
        inicio = time.perf_counter()
        pred = ODONTOGRAFO.predecir(img, umbral_hallazgos=MINIMO)
        segundos = time.perf_counter() - inicio
        _MEMORIA[clave] = (ahora, pred, segundos)
        if len(_MEMORIA) > CAPACIDAD:
            _MEMORIA.popitem(last=False)
        print(f"{img.width}x{img.height}: {len(pred.piezas)} piezas, "
              f"{len(pred.hallazgos)} hallazgos desde {MINIMO}, {segundos:.1f} s")
        return pred, segundos


# --- Salidas ------------------------------------------------------------------


def coma(x: float, decimales: int = 1) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")


def contar(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def nombre_pieza(fdi: int) -> str:
    return f"{NOMBRE_POSICION[fdi % 10]} {NOMBRE_CUADRANTE[fdi // 10]}"


def resumen(pred: Prediccion, segundos: float) -> str:
    presentes = {p.fdi for p in pred.piezas}
    ausentes = [str(f) for f in FDIS if f not in presentes]
    tipos = Counter(h.patologia for h in pred.hallazgos)
    detalle = ", ".join(contar(n, *HALLAZGOS[t][:2]) for t, n in tipos.most_common())
    partes = [
        f"**{contar(len(presentes), 'pieza numerada', 'piezas numeradas')}**",
        "ausentes: " + (", ".join(ausentes) if ausentes else "ninguna"),
        f"**{contar(len(pred.hallazgos), 'hallazgo', 'hallazgos')}**" + (f" ({detalle})" if detalle else ""),
        f"{coma(segundos)} s en {'GPU' if ODONTOGRAFO.dev.type == 'cuda' else 'CPU'}",
    ]
    return " · ".join(partes)


def tabla(hallazgos: list[Hallazgo]) -> dict:
    orden = sorted(hallazgos, key=lambda h: (h.fdi is None, h.fdi or 0, -h.confianza))
    return {
        "headers": ["Pieza", "Hallazgo", "Confianza"],
        "data": [
            [f"{h.fdi} · {nombre_pieza(h.fdi)}" if h.fdi else "sin pieza asignada",
             HALLAZGOS[h.patologia][0], f"{h.confianza:.0%}"]
            for h in orden
        ],
    }


def exportar(pred: Prediccion, umbral: float) -> dict:
    """El odontograma como datos, para integrarlo en otro programa. Las cajas
    van en píxeles de la imagen subida: [x1, y1, x2, y2]."""

    def caja(c) -> list[int]:
        return [round(v / pred.escala) for v in c]

    presentes = {p.fdi for p in pred.piezas}
    return {
        "notacion": "FDI",
        "piezas": [{"fdi": p.fdi, "caja": caja(p.caja)} for p in sorted(pred.piezas, key=lambda p: p.fdi)],
        "ausentes": [f for f in FDIS if f not in presentes],
        "hallazgos": [
            {"fdi": h.fdi, "tipo": HALLAZGOS[h.patologia][2], "confianza": round(h.confianza, 3),
             "caja": caja(h.caja)}
            for h in sorted(pred.hallazgos, key=lambda h: (h.fdi is None, h.fdi or 0))
        ],
        "umbral_hallazgos": umbral,
        "aviso": "Borrador automático para revisión profesional. No apto para uso clínico.",
    }


def json_compacto(datos: dict) -> str:
    """JSON válido con una pieza o un hallazgo por línea: indentado del todo,
    cada caja ocuparía seis líneas."""
    lineas = []
    for clave, valor in datos.items():
        if isinstance(valor, list) and valor and isinstance(valor[0], dict):
            elementos = ",\n".join(f"    {json.dumps(v, ensure_ascii=False)}" for v in valor)
            lineas.append(f"  {json.dumps(clave)}: [\n{elementos}\n  ]")
        else:
            lineas.append(f"  {json.dumps(clave)}: {json.dumps(valor, ensure_ascii=False)}")
    return "{\n" + ",\n".join(lineas) + "\n}"


def analizar(ruta: str | None, umbral: float, progress=gr.Progress()):
    if ruta is None:
        raise gr.Error("Sube una radiografía panorámica o elige uno de los ejemplos.")
    img, avisos = abrir(ruta)
    for aviso in avisos:
        gr.Warning(f"{aviso} El resultado no será fiable.", title="Aviso")
    progress(0, desc="Analizando la radiografía…")
    pred, segundos = predecir(img)
    if not pred.piezas:
        gr.Warning("No he encontrado ninguna pieza. ¿Es una radiografía panorámica?", title="Aviso")
    vista = replace(pred, hallazgos=[h for h in pred.hallazgos if h.confianza >= umbral])
    figura = apilar(
        dibujar_prediccion(vista.imagen, vista.piezas, hallazgos=vista.hallazgos),
        dibujar_odontograma({p.fdi for p in vista.piezas}, hallazgos=vista.por_diente(),
                            titulo="Odontograma generado"),
    )
    return resumen(vista, segundos), figura, tabla(vista.hallazgos), json_compacto(exportar(vista, umbral))


def ejemplo(ruta: str, progress=gr.Progress()):
    """Un ejemplo se muestra con el umbral por defecto, y el control vuelve a él."""
    return *analizar(ruta, UMBRAL, progress), UMBRAL


def reajustar(ruta: str | None, umbral: float):
    if ruta is None:
        return [gr.skip()] * 4
    return analizar(ruta, umbral)


# --- Interfaz -----------------------------------------------------------------

CABECERA = f"""
# dental-vision

Odontograma automático a partir de una radiografía panorámica: detecta cada
pieza, la numera en notación FDI y marca los hallazgos (caries, caries
profunda, lesión periapical y diente impactado).

[Código y evaluación]({CODIGO}) · [Modelo](https://huggingface.co/{REPO_PESOS})
"""

AVISO = """
**Proyecto de investigación, no apto para uso clínico.** No subas radiografías
de pacientes: la demo es pública. No guarda las imágenes; los ficheros
temporales se borran en menos de una hora.
"""

COMO_FUNCIONA = f"""
Dos modelos Mask R-CNN (torchvision, ResNet-50 FPN) entrenados con el dataset
DENTEX:

1. **Numeración.** Detecta y segmenta cada pieza y le asigna su número FDI.
   Después, un post-procesado impone dos reglas anatómicas: cada número una sola
   vez (algoritmo húngaro) y el orden de la arcada (alineamiento por
   programación dinámica).
2. **Hallazgos.** Detecta caries, caries profunda, lesión periapical y diente
   impactado, y asigna cada hallazgo a la pieza con la que más solapa.

**Qué esperar**, medido con radiografías que los modelos no vieron al entrenar:

- Numeración: de media hay que corregir 1,6 piezas por radiografía, y el 54% de
  los odontogramas sale perfecto. Los fallos típicos son premolares y molares
  numerados un puesto corrido.
- Dientes impactados: encuentra el 90%.
- Caries: encuentra el 47% y marca unas 2 falsas por radiografía. Sirve para
  señalar dónde mirar, no para fiarse de lo que no marca.
- Lesiones periapicales: encuentra el 36%.

**Límites.** Solo radiografías panorámicas, no periapicales, aletas de mordida
ni fotos. Solo dentición permanente: DENTEX no incluye menores de 12 años. Todas
las radiografías de entrenamiento vienen de los tres centros de DENTEX; con
otros equipos lo normal es que rinda algo peor.

Detalles, métricas con intervalos de confianza y decisiones técnicas en el
[README del proyecto]({CODIGO}#readme).
"""

PIE = f"""
Datos: [DENTEX 2023](https://zenodo.org/records/7812323) (Hamamci et al.), de
donde salen también las radiografías de ejemplo; su licencia, en el
[README]({CODIGO}#datos). Código con licencia MIT.
"""

TEMA = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    # Fuentes del sistema: sin peticiones a Google Fonts.
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
)

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
#aviso { border-left: 4px solid var(--color-accent); padding: 4px 14px;
         background: var(--block-background-fill); border-radius: 6px; }
#tabla table, #tabla th, #tabla td, #tabla span { font-family: var(--font) !important; }
"""

with gr.Blocks(title="dental-vision · odontograma automático", analytics_enabled=False,
               delete_cache=(1800, 1800)) as demo:
    gr.Markdown(CABECERA)
    gr.Markdown(AVISO, elem_id="aviso")
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=300) as izquierda:
            entrada = gr.Image(type="filepath", image_mode=None, sources=["upload", "clipboard"],
                               label="Radiografía panorámica", height=260)
            umbral = gr.Slider(MINIMO, 0.9, value=UMBRAL, step=0.05, label="Umbral de los hallazgos",
                               info=f"{coma(UMBRAL)} es el elegido en validación. Más bajo: encuentra "
                                    "más, pero con más falsas alarmas.")
        with gr.Column(scale=5):
            texto = gr.Markdown()
            # Radiografía y odontograma en una sola figura, como en el README: los
            # botones de la esquina caen sobre el fondo negro de la radiografía.
            figura = gr.Image(label="Radiografía anotada y odontograma", show_label=False,
                              interactive=False, buttons=["download", "fullscreen"])
            with gr.Tabs():
                with gr.Tab("Hallazgos"):
                    hallazgos = gr.Dataframe(interactive=False, wrap=True, elem_id="tabla")
                with gr.Tab("Exportar (JSON)"):
                    datos = gr.Code(language="json", interactive=False, show_label=False, lines=12,
                                    max_lines=24, buttons=["copy", "download"])
                with gr.Tab("Cómo funciona"):
                    gr.Markdown(COMO_FUNCIONA)
    salidas = [texto, figura, hallazgos, datos]
    # Los ejemplos van bajo la entrada, pero necesitan las salidas ya creadas.
    with izquierda:
        gr.Examples(
            examples=[[str(AQUI / "ejemplos" / f)] for f in EJEMPLOS],
            inputs=[entrada],
            outputs=[*salidas, umbral],
            fn=ejemplo,
            cache_examples=True,
            cache_mode="eager",
            example_labels=list(EJEMPLOS.values()),
            label="Ejemplos: radiografías que los modelos no vieron al entrenar",
        )
    gr.Markdown(PIE)

    entrada.upload(analizar, [entrada, umbral], salidas, api_name="analizar")
    umbral.release(reajustar, [entrada, umbral], salidas, api_visibility="private")
    entrada.clear(lambda: [None] * 4, None, salidas, api_visibility="private")

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    # Sin historial de ejecuciones: guardaría en el navegador las radiografías subidas.
    demo.launch(theme=TEMA, css=CSS, max_file_size="20mb", run_history=False)
