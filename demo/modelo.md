---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - dental
  - panoramic-radiograph
  - mask-rcnn
  - torchvision
datasets:
  - ibrahimhamamci/DENTEX
---

# dental-vision

Pesos de los dos modelos de [dental-vision](https://github.com/abp002/dental-vision):
numeración de las piezas dentales en notación FDI y detección de hallazgos en
radiografías panorámicas. Se pueden probar en la
[demo](https://huggingface.co/spaces/abp0040/dental-vision).

**No apto para uso clínico.** Es un proyecto de investigación y aprendizaje. En
la UE, un uso clínico real sería un producto sanitario (MDR 2017/745).

## Modelos

| Carpeta | Qué hace | Salida |
|---|---|---|
| `numeracion/` | Detecta y segmenta cada pieza y le asigna su número FDI | 32 clases; cajas y máscaras |
| `patologia/` | Detecta hallazgos sobre el diente afectado | Diente impactado, caries, lesión periapical y caries profunda; cajas |

Los dos son Mask R-CNN v2 de torchvision (ResNet-50 FPN, preentrenado en COCO)
y trabajan a 1.333 px de ancho. Cada carpeta trae los pesos
(`mejor.safetensors`, la mejor época en validación), la configuración de
entrenamiento y lo que se ajustó en validación: el post-procesado anatómico de la
numeración (`postproceso.json`), el umbral de las piezas de número dudoso
(`duda.json`) y el umbral de confianza de los hallazgos (`umbral.json`).

## Uso

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from dentalvision.inferencia import Odontografo  # pip install git+https://github.com/abp002/dental-vision

pesos = Path(snapshot_download("abp0040/dental-vision"))
modelo = Odontografo(pesos / "numeracion", patologia=pesos / "patologia")
pred = modelo.predecir(Path("panoramica.png"))

print(sorted(p.fdi for p in pred.piezas))
print([(p.fdi, p.alternativa) for p in pred.piezas if p.dudosa])  # revisar: ¿36 o 37?
print([(h.fdi, h.patologia, round(h.confianza, 2)) for h in pred.hallazgos])
```

Las cajas salen a la resolución de trabajo; divididas por `pred.escala`, en
píxeles de la imagen original. `Odontografo` aplica el post-procesado anatómico:
cada número una sola vez y en el orden de la arcada. Las piezas cuyo número
no es firme llevan `alternativa`: en prueba, una de cada cinco marcadas está mal
numerada (frente a 1 de cada 30 en general), pero solo se marca un tercio de
los fallos.

## Resultados

**Numeración.** Partición de prueba: 91 radiografías que el modelo no vio al
entrenar y que no se usaron para elegir nada. Entre paréntesis, intervalo de
confianza del 95% (bootstrap por radiografía).

| Exactitud | Errores por radiografía | Odontogramas perfectos |
|---|---|---|
| 95,5% (93,9–97,1) | 1,64 (1,11–2,18) | 53,8% (44,0–64,8) |

La exactitud es la parte de las piezas presentes que sale encontrada y bien
numerada. Los errores por radiografía son las correcciones que tiene que hacer
el profesional: piezas que faltan, mal numeradas o que sobran.

**Hallazgos.** Conjuntos oficiales del concurso DENTEX, que el modelo no vio. La
prueba oficial se publicó con las etiquetas clínicas originales y solo se
evalúan las tres clases con traducción inequívoca (no hay caries profunda).

| | Validación oficial (50) | Prueba oficial (250) |
|---|---|---|
| AP diagnóstico | 0,33 (0,25–0,41) | 0,35 (0,33–0,39) |
| AP50 | 0,51 (0,38–0,62) | 0,54 (0,50–0,59) |

Con el umbral de 0,5, en la prueba oficial:

| Hallazgo | Encontrados | De lo marcado, real | Falsas alarmas por radiografía |
|---|---|---|---|
| Diente impactado | 90% | 86% | 0,13 |
| Caries | 47% | 44% | 1,83 |
| Lesión periapical | 36% | 48% | 0,12 |

Como referencia, el método de los organizadores obtuvo un AP de diagnóstico de
0,37 en la prueba oculta del concurso. La comparación es orientativa: no es el
mismo conjunto ni las mismas clases.

## Datos de entrenamiento

[DENTEX 2023](https://zenodo.org/records/7812323): panorámicas de tres centros,
con equipos y protocolos distintos, de pacientes de 12 años o más.

| Modelo | Subconjunto | Entrenamiento | Validación | Prueba | Excluidas |
|---|---|---|---|---|---|
| Numeración | Enumeración (634) | 425 | 91 | 91 | 27 |
| Hallazgos | Diagnóstico (705) | 575 | 102 | conjuntos oficiales | 28 |

Se excluyen las copias exactas (por MD5). En numeración, también las imágenes
con números FDI duplicados, que son errores de anotación. En hallazgos, todo lo
que aparece en los conjuntos oficiales, que son su prueba.

## Limitaciones

- Solo radiografías panorámicas: no periapicales, aletas de mordida ni fotos.
- Solo dentición permanente: DENTEX no incluye menores de 12 años, así que el
  modelo no conoce los dientes temporales (51–85).
- Sin validación externa: todas las radiografías vienen de los centros de
  DENTEX. Con otros equipos lo normal es que rinda algo peor.
- La partición es por imagen porque DENTEX no publica identificador de paciente.
- Caries: encuentra la mitad y marca casi dos falsas por radiografía, y confunde
  a menudo caries con caries profunda. Sirve para señalar dónde mirar, no para
  descartar.
- Numeración: los fallos se concentran en premolares y molares, sobre todo en
  bocas con muchas ausencias.
- No anota restauraciones (coronas, implantes, endodoncias): DENTEX no las
  etiqueta.

## Licencia

Pesos bajo CC BY-NC-SA 4.0. DENTEX figura como CC BY 4.0 en su registro de
Zenodo y como CC BY-NC-SA 4.0 en su ficha de Hugging Face: los pesos se publican
con la más restrictiva, que cumple las dos. El código del proyecto es MIT.

## Cita

Hamamci et al., *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and
Diagnosis Benchmark for Panoramic X-rays*, [arXiv:2305.19112](https://arxiv.org/abs/2305.19112), 2023.
