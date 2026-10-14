# dental-vision

Radiografía panorámica → odontograma: un Mask R-CNN numera las piezas (FDI) y otro marca
hallazgos (caries, caries profunda, lesión periapical, impactado). Código, comentarios y
README en español. Comandos, estructura y resultados: el README.

## QA
Nivel: activo

Bitácora: DV

## Dónde se trabaja
- Se entrena y evalúa en el PC (`~/Proyectos/dental-vision`, RTX 4060 Ti de 8 GB): ahí
  están `data/` (DENTEX, 10 GB) y `modelos/`, los dos fuera de git. Un entrenamiento a
  1.333 px dura unos 20 min (2,5 GB de VRAM); a 2.000 px, unos 65 min (4,8 GB),
  y no mejoró los hallazgos (DV-5).
- El PC no tiene credenciales de GitHub: los commits se traen a un clon del Mac
  (`git fetch <pc>:Proyectos/dental-vision main`) y se suben desde allí.
- La suite (`uv run pytest`) y `uv run ruff check .` no necesitan ni datos ni pesos: es lo
  que corre la CI.

## Reglas que no se rompen
- **La prueba se mira una sola vez y con todo fijado en validación.** `evaluar.py` se
  niega a ajustar en prueba. Un cambio que no gana en validación no se pasa a prueba: se
  documenta como resultado negativo en «Decisiones técnicas» del README.
- Para comparar modelos de numeración: `scripts/comparar.py` (bootstrap emparejado por
  radiografía), nunca dos cifras sueltas.
- `base` (numeración) y `patologia` (hallazgos) son los modelos publicados en HF
  (`abp0040/dental-vision`, con `scripts/publicar.py --subir`). Cada carpeta de
  `modelos/` lleva sus ajustes de validación: `postproceso.json`, `duda.json` y `umbral.json`.
- Cada modelo trabaja a su propio ancho (`config.json`); la inferencia carga la original a
  la resolución de cada uno.
- Hugging Face ya no aloja Spaces de Gradio gratis: la demo se enseña en vídeo y se
  ejecuta en local (`demo/app.py`).
