# dental-vision

Odontograma automático a partir de radiografías panorámicas: detectar cada pieza
dental y numerarla según la notación FDI.

![Predicción del modelo sobre una radiografía de prueba y el odontograma que genera](docs/prediccion_tipica.png)

*Predicción del modelo sobre una radiografía de la partición de prueba, que no vio al entrenar: las 30 piezas encontradas y bien numeradas. Es la radiografía típica, no la mejor: se eligió la primera de la partición con el número de fallos mediano, que es cero. Debajo, el odontograma que genera.*

## El problema

Rellenar el odontograma de un paciente nuevo son entre cinco y diez minutos de
trabajo manual, pieza por pieza, en cada primera visita. Es tedioso, repetitivo y
se hace con prisa. Este proyecto genera el borrador automáticamente para que el
profesional solo tenga que revisarlo.

## Qué es y qué no es

Esto **registra qué piezas hay y qué número tiene cada una**. Es documentación
clínica. El siguiente paso natural, anotar su estado (corona, implante,
endodoncia), no se puede aprender de DENTEX: no etiqueta restauraciones.

Esto **no diagnostica**. No dice si hay caries ni sugiere tratamiento. Aun así, la
frontera con el producto sanitario no es nítida: en la UE, un software destinado a
proporcionar información para decisiones diagnósticas o terapéuticas es producto
sanitario y requiere marcado CE conforme al MDR 2017/745, y un odontograma entra en
la historia clínica y se usa para planificar tratamientos. Un uso clínico real
habría que tratarlo como producto sanitario.

**No es apto para uso clínico.** Es un proyecto de investigación y aprendizaje.

## Resultados

Partición de prueba: 93 radiografías que el modelo no vio al entrenar y que no se
usaron para elegir nada. Entre paréntesis, intervalo de confianza del 95%
(bootstrap por radiografía).

| Post-procesado | Cobertura | Numeración | Exactitud | Errores por radiografía | Odontogramas perfectos |
|---|---|---|---|---|---|
| Ninguno | 98,7% (98,1–99,2) | 96,2% (94,7–97,5) | 94,9% (93,2–96,5) | 1,89 (1,32–2,51) | 50,5% (39,8–61,3) |
| Unicidad | 98,8% (98,2–99,3) | 96,4% (95,0–97,7) | 95,2% (93,5–96,8) | 1,69 (1,15–2,29) | 55,9% (46,2–65,6) |
| Unicidad + orden | 98,8% (98,2–99,2) | 96,4% (95,0–97,7) | 95,2% (93,5–96,8) | 1,69 (1,15–2,29) | 55,9% (46,2–65,6) |

Con unas 29 piezas por radiografía, el borrador sale perfecto en 52 de las 93 y de media hay que corregir 1,7 piezas. La distribución es desigual: 60 radiografías tienen como mucho un fallo y 22 acumulan tres o más. Las cifras de validación fueron muy parecidas (1,60 errores y 62% de odontogramas perfectos), así que no hay sobreajuste a la validación.

**Dónde falla.** Casi todos los fallos son de numeración en premolares y molares: piezas que se parecen a sus vecinas y que, cuando falta alguna, se prestan a correr la numeración un puesto. Las bocas con muchas ausencias concentran los errores.

| Posición | Mal numeradas | No encontradas | Sobrantes |
|---|---|---|---|
| 1 incisivo central | 4 | 4 | 2 |
| 2 incisivo lateral | 5 | 4 | 3 |
| 3 canino | 4 | 7 | 7 |
| 4 primer premolar | 15 | 6 | 8 |
| 5 segundo premolar | 12 | 9 | 7 |
| 6 primer molar | 17 | 2 | 1 |
| 7 segundo molar | 30 | 0 | 2 |
| 8 cordal | 7 | 1 | 0 |

![Radiografía de prueba con muchas ausencias y fallos de numeración](docs/prediccion_fallos.png)

*Un caso difícil, elegido con regla fija (la primera radiografía en el percentil 90 de fallos: cinco). Boca con muchas ausencias: el modelo corre un puesto la numeración de los incisivos inferiores, no encuentra el 42 y toma por 27 el cordal que sigue al hueco. En rojo, los fallos: «27>28» es una pieza que el modelo numeró 27 y es la 28; el recuadro sin relleno, una pieza que no encontró.*

## Datos

[DENTEX Challenge 2023](https://zenodo.org/records/7812323), licencia CC BY 4.0.

| Subconjunto | Imágenes | Anotaciones |
|---|---|---|
| Enumeración | 634 | 18.095 piezas con cuadrante + posición FDI |
| Patología | 705 | 3.529 hallazgos (caries, caries profunda, lesión periapical, impactado) |

Las anotaciones traen polígonos de segmentación (mediana de 12 puntos), no solo
cajas delimitadoras.

![Anotaciones del dataset: las 32 piezas numeradas en notación FDI](docs/ejemplo_odontograma.png)

*Anotaciones del dataset (no es una predicción): las 32 piezas segmentadas y
numeradas, un color por cuadrante. La marca `R` de la esquina inferior izquierda
confirma la convención radiológica: la derecha del paciente aparece a la izquierda
de la imagen.*

![Hallazgos patológicos marcados sobre una panorámica](docs/ejemplo_patologia.png)

*El subconjunto de patología, con los hallazgos marcados en rojo. Este proyecto
no diagnostica: el objetivo es el odontograma. Se documenta aquí porque forma
parte del dataset.*

El dataset no se versiona aquí. Para obtenerlo:

```sh
sh scripts/descargar_dentex.sh
```

Zenodo puede ir muy lento (medido: 0,6 MB/s, cinco horas para los 10 GB). Los
autores publican el mismo fichero en Hugging Face, bastante más rápido; el script
lo acepta con `DENTEX_URL` y comprueba el MD5 que publica Zenodo, así que el
resultado es idéntico byte a byte. Ojo con la licencia: el registro de Zenodo dice
CC BY 4.0 y la ficha de Hugging Face, CC BY-NC-SA 4.0. Este proyecto se apoya en la
del registro oficial, pero la discrepancia importaría en un uso comercial.

## Auditoría del dataset

Antes de entrenar nada se revisó la calidad de las anotaciones. Dos hallazgos:

**1. Las anotaciones son coherentes con la epidemiología real.** La posición 8
(cordales) aparece un 37% menos que la media del resto, y el primer molar inferior
(36/46) un 14% menos que su equivalente superior (16/26) — es el diente que más
caries y extracciones acumula a lo largo de la vida. Que los datos reproduzcan un hecho
clínico conocido sin que nadie se lo pida es una señal fuerte de que el mapeo de
arcadas y posiciones es correcto. Izquierda y derecha no las distingue (la
epidemiología es simétrica); eso lo confirma la marca `R` de las radiografías.

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

Requiere Python 3.12 (uv lo instala solo) y, para entrenar en un tiempo razonable,
una GPU NVIDIA con CUDA.

```sh
uv run python scripts/entrenar.py --nombre base              # ~20 min en una RTX 4060 Ti
uv run python scripts/evaluar.py --ejecucion base            # ajusta el post-procesado en validación
uv run python scripts/evaluar.py --ejecucion base --particion prueba
uv run python scripts/odontograma.py --ejecucion base --imagen panoramica.png
uv run pytest
```

## Estructura

```
src/dentalvision/
  dentex.py       carga del dataset, composición del número FDI, control de calidad
  render.py       dibujo de anotaciones, predicciones y odontograma
  particiones.py  separación reproducible entrenamiento/validación/prueba
  datos.py        Dataset de torchvision, máscaras y aumentación
  modelo.py       Mask R-CNN con transfer learning
  postproceso.py  restricciones anatómicas: unicidad y orden de las piezas
  evaluacion.py   métricas de dominio e intervalos de confianza
scripts/
  descargar_dentex.sh
  entrenar.py     entrenamiento; guarda cada ejecución en modelos/<nombre>/
  evaluar.py      ajuste en validación y evaluación final en prueba
  comparar.py     comparación de ejecuciones con bootstrap emparejado
  odontograma.py  figura de una radiografía con su odontograma
tests/
```

## Decisiones técnicas

Decisiones tomadas con una medición delante, no por costumbre.

**La resolución la decide el Dataset, no torchvision.** Mask R-CNN reescala cada
imagen por dentro: el lado corto a 800 px sin que el largo pase de 1.333. Con una
panorámica, la red trabaja siempre a unos 1.333 px de ancho, se le dé lo que se le
dé. La primera versión de este proyecto reducía las radiografías a 800 px y la red
las volvía a ampliar: se perdía detalle a cambio de nada. Eso también invalidaba
una medición anterior («reducir de 800 a 512 px solo ahorra un 11%, luego el coste
está en las cabezas»): la red procesó el mismo tamaño en las dos pruebas. Ahora el
modelo se crea con `min_size = max_size = ancho` y la resolución la fija una sola
constante.

**Entrenamiento en GPU NVIDIA; en un Mac, en CPU.** En una RTX 4060 Ti (8 GB) con
precisión mixta, un paso de entrenamiento (lote de 2 radiografías a 1.333 px)
tarda 0,2 s y ocupa 2,5 GB: una época con su validación, 50 segundos; las 24
épocas, 20 minutos. En un Apple M4, según las mediciones de la primera versión
de este proyecto, la CPU tardaba unos 4 s por paso (casi 6 horas para 24 épocas)
y el backend Metal (MPS) era mucho más lento todavía, unos 107 s. La causa no está identificada: torchvision 0.28 sí tiene kernels MPS
para `roi_align` y `nms`, así que no es un *fallback* a CPU de esas operaciones.

**128 regiones ROI en lugar de 512.** Los valores por defecto de torchvision
están pensados para COCO, donde los objetos aparecen en cualquier posición y
número. Una boca tiene como mucho 32 piezas, siempre en una banda central y en
orden previsible. En la CPU del Mac, bajar a 128 ROIs y 500 propuestas del RPN
aceleraba un 32%. Faltaba medir si costaba precisión, y no se nota: entrenando
las dos versiones completas en la 4060 Ti, 512 ROIs tardan un 24% más por paso
(0,24 s frente a 0,20) y dan 1,48 errores por radiografía en validación frente a
1,60, una diferencia de −0,12 con intervalo de −0,37 a +0,12 (bootstrap
emparejado, una sola semilla). Dentro del ruido: se quedan las 128.

**Mask R-CNN de torchvision (BSD), no Ultralytics YOLO (AGPL-3.0).** La AGPL
obligaría a publicar el código completo de cualquier servicio que use el modelo,
o a adquirir licencia comercial. La decisión de licencia se toma al principio del
proyecto, no cuando ya hay un cliente. Mismo criterio al elegir el dataset:
DENTEX (CC BY 4.0) frente a alternativas CC BY-NC.

**El volteo horizontal remapea los cuadrantes.** Es la aumentación más común y
en imagen médica lateralizada es destructiva: al voltear, el diente 16 pasa a
ocupar la posición del 26. Sin remapear las etiquetas (1↔2, 3↔4), la mitad de
los ejemplos enseñan lo contrario que la otra mitad, el entrenamiento no da
ningún error y el modelo simplemente nunca aprende a distinguir izquierda de
derecha.

**Restricciones anatómicas después del detector.** El detector decide cada pieza
por separado y su NMS es por clase: una misma pieza puede salir como 16 y como 17
a la vez, y dos piezas distintas pueden recibir el mismo número. El post-procesado
agrupa las detecciones que caen sobre la misma pieza y asigna los números con dos
reglas: cada número una sola vez (algoritmo húngaro) y el orden de la arcada. El
orden se resuelve como un alineamiento de secuencias, igual que en
bioinformática: las piezas, de izquierda a derecha, contra 18…11 21…28, con
huecos permitidos para las que faltan. Cuando el orden obliga a cambiar un número,
se prefiere el vecino más cercano a lo que vio el detector. Medido en prueba, agrupar y exigir unicidad baja los errores de 1,89 a 1,69 por radiografía y los odontogramas perfectos del 50% al 56%: sobre todo quita duplicados y recupera piezas en las que el detector dudaba entre dos números. El orden no añade nada con el modelo final (sí ayudaba a mitad de entrenamiento): el detector ya aprende la secuencia, y los fallos que quedan corren la numeración respetando el orden, así que esa regla no los ve. Se mantiene porque garantiza que el odontograma nunca sea anatómicamente imposible.

**BatchNorm del backbone.** La v2 usa BatchNorm normal, y con lotes de 2 imágenes sus estadísticas se recalculan con muy poca muestra. Congelarla (`FrozenBatchNorm2d`), lo habitual al ajustar con lotes pequeños, no mejora nada medible: 1,67 errores por radiografía en validación frente a 1,60, diferencia +0,06 con intervalo de −0,18 a +0,34. Se queda como viene.

**La prueba se mira una vez.** Los umbrales del post-procesado y la elección entre
configuraciones se hacen en validación, y `scripts/evaluar.py` se niega a evaluar
la partición de prueba si no están fijados. Los intervalos de confianza se
calculan remuestreando radiografías, no piezas: los dientes de una misma boca no
son independientes, y tratarlos como tales daría intervalos falsamente estrechos.

## Métricas

Se reportan medidas de dominio en lugar de mAP, que no significa nada para un
clínico:

| Métrica | Qué mide |
|---|---|
| Cobertura | De las piezas presentes, cuántas se encuentran |
| Numeración | De las encontradas, a cuántas se les asigna el FDI correcto |
| Exactitud | De las piezas presentes, cuántas salen encontradas y bien numeradas |
| Errores por radiografía | Correcciones que tiene que hacer el profesional: piezas que faltan, números mal puestos y piezas que sobran |
| Odontogramas perfectos | En cuántas radiografías sale todo bien, sin un solo fallo |

Los porcentajes por pieza engañan: un 97% de numeración correcta suena excelente,
pero con 28 piezas por boca son 0,84 errores por radiografía. Como el profesional
revisa siempre el borrador, lo que mide el ahorro de tiempo es cuántas
correcciones tiene que hacer: los errores por radiografía.

## Estado

- [x] Descarga reproducible y auditoría de anotaciones
- [x] Carga del dataset y composición FDI
- [x] Visor de anotaciones
- [x] Separación entrenamiento/validación/prueba (432 / 93 / 93, semilla fija)
- [x] Detector de piezas dentales
- [x] Post-procesado con restricciones anatómicas
- [x] Odontograma visual
- [x] Evaluación

Siguientes pasos:

- **Etiquetas jerárquicas.** DENTEX anota cuadrante y posición por separado, y son
  señales visuales distintas: el cuadrante se deduce de dónde está la pieza y la
  posición, de su forma. Entrenar solo la posición (8 clases, unas cuatro veces
  más ejemplos por clase que las 32 actuales) y sacar el cuadrante por geometría
  ataca directamente el error dominante.
- **Estado de cada pieza** (corona, implante, endodoncia): requiere otro conjunto
  de datos.
- **Validación externa**, con radiografías de otro centro y otro equipo. La
  partición es por imagen porque DENTEX no publica identificador de paciente
  (ver `particiones.py`).

## Licencia

Código bajo licencia MIT. Las imágenes de `docs/` proceden del dataset DENTEX
(CC BY 4.0) y se reproducen con atribución.

## Atribución

Dataset DENTEX 2023, publicado bajo CC BY 4.0. Hamamci et al., *DENTEX: An
Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for
Panoramic X-rays*.
