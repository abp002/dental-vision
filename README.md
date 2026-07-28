# dental-vision

Odontograma automático a partir de radiografías panorámicas: detectar cada pieza
dental, numerarla según la notación FDI y registrar su estado visible.

## El problema

Rellenar el odontograma de un paciente nuevo son entre cinco y diez minutos de
trabajo manual, pieza por pieza, en cada primera visita. Es tedioso, repetitivo y
se hace con prisa. Este proyecto genera el borrador automáticamente para que el
profesional solo tenga que revisarlo.

## Qué es y qué no es

Esto **registra el estado presente** (qué piezas hay, cuáles llevan corona,
implante o endodoncia). Es documentación clínica.

Esto **no diagnostica**. No dice si hay caries ni sugiere tratamiento. Esa
distinción no es cosmética: en la UE, un software destinado a proporcionar
información para decisiones diagnósticas o terapéuticas es producto sanitario y
requiere marcado CE conforme al MDR 2017/745. Este proyecto se mantiene
deliberadamente fuera de esa categoría.

**No es apto para uso clínico.** Es un proyecto de investigación y aprendizaje.

## Datos

[DENTEX Challenge 2023](https://zenodo.org/records/7812323), licencia CC BY 4.0.

| Subconjunto | Imágenes | Anotaciones |
|---|---|---|
| Enumeración | 634 | 18.095 piezas con cuadrante + posición FDI |
| Patología | 705 | 3.529 hallazgos (caries, caries profunda, lesión periapical, impactado) |

Las anotaciones traen polígonos de segmentación (mediana de 12 puntos), no solo
cajas delimitadoras.

El dataset no se versiona aquí. Para obtenerlo:

```sh
sh scripts/descargar_dentex.sh
```

## Auditoría del dataset

Antes de entrenar nada se revisó la calidad de las anotaciones. Dos hallazgos:

**1. Las anotaciones son coherentes con la epidemiología real.** La posición 8
(cordales) aparece un 40% menos que el resto, y el primer molar inferior (36/46)
aparece menos que su equivalente superior (16/26) — es el diente que más caries y
extracciones acumula a lo largo de la vida. Que los datos reproduzcan un hecho
clínico conocido sin que nadie se lo pida es una señal fuerte de que el mapeo de
cuadrantes y posiciones es correcto.

**2. Un 2,5% de las imágenes tiene errores de anotación.** 16 de las 634 imágenes
contienen números FDI duplicados, lo que es anatómicamente imposible. En al menos
un caso hay dos series de anotaciones superpuestas y desplazadas una posición
sobre el mismo cuadrante.

La detección es automática y usa una restricción del dominio como test: una boca
no puede tener dos dientes con el mismo número.

```python
from dentalvision.dentex import Subconjunto
corruptas = [r for r in Subconjunto("enumeracion").radiografias if r.duplicados]
```

## Puesta en marcha

```sh
uv sync
sh scripts/descargar_dentex.sh
```

Requiere Python 3.12. PyTorch usa el backend Metal (MPS) en Apple Silicon.

## Estructura

```
src/dentalvision/
  dentex.py     carga del dataset, composición del número FDI, control de calidad
  render.py     dibujo de polígonos y etiquetas sobre la radiografía
scripts/
  descargar_dentex.sh
```

## Estado

- [x] Descarga reproducible y auditoría de anotaciones
- [x] Carga del dataset y composición FDI
- [x] Visor de anotaciones
- [ ] Separación entrenamiento/validación
- [ ] Detector de piezas dentales
- [ ] Post-procesado con restricciones anatómicas
- [ ] Odontograma visual
- [ ] Evaluación

## Atribución

Dataset DENTEX 2023, publicado bajo CC BY 4.0. Hamamci et al., *DENTEX: An
Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for
Panoramic X-rays*.
