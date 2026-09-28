---
title: dental-vision
emoji: 🦷
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
license: mit
short_description: Odontograma automático desde una radiografía panorámica
models:
  - abp002/dental-vision
preload_from_hub:
  - abp002/dental-vision
datasets:
  - ibrahimhamamci/DENTEX
tags:
  - medical-imaging
  - dental
---

# dental-vision

Demo del proyecto [dental-vision](https://github.com/abp002/dental-vision): a
partir de una radiografía panorámica, detecta cada pieza dental, la numera en
notación FDI y marca los hallazgos (caries, caries profunda, lesión periapical y
diente impactado). Devuelve la radiografía anotada, el odontograma y el
resultado en JSON.

**Proyecto de investigación, no apto para uso clínico.**

- Código, evaluación y decisiones técnicas: [GitHub](https://github.com/abp002/dental-vision)
- Pesos y ficha del modelo: [abp002/dental-vision](https://huggingface.co/abp002/dental-vision)

Las radiografías de ejemplo proceden de [DENTEX 2023](https://zenodo.org/records/7812323)
(Hamamci et al., CC BY 4.0). El código es MIT.
