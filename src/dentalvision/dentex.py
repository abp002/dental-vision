"""Carga del dataset DENTEX 2023.

DENTEX no usa COCO estandar: en vez de un unico `category_id`, cada anotacion
trae varias etiquetas jerarquicas.

    category_id_1  ->  cuadrante   (0..3  =  cuadrantes FDI 1..4)
    category_id_2  ->  posicion    (0..7  =  posiciones 1..8, del centro afuera)
    category_id_3  ->  patologia   (solo en el subconjunto 'disease')

El numero FDI se compone: cuadrante 1 + posicion 6 -> diente 16.

Esa descomposicion es una buena decision de diseno de los autores. Con 32 clases
planas habria ~560 ejemplos de cada una; separado hay ~4.500 por cuadrante y
~2.400 por posicion. Ademas son dos senales visuales distintas: el cuadrante se
deduce de DONDE esta el diente en la imagen, la posicion de COMO es su forma.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data" / "dentex"

PATOLOGIAS = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
NOMBRE_PATOLOGIA = {
    "Impacted": "impactado",
    "Caries": "caries",
    "Periapical Lesion": "lesión periapical",
    "Deep Caries": "caries profunda",
}

# La prueba oficial del concurso se publicó con las etiquetas clínicas
# originales (LabelMe, en turco), no con las 4 clases de DENTEX. Solo se
# traducen las que no admiten duda. "kanal" (endodoncia) y "küretaj" (curetaje)
# no tienen equivalente claro entre las 4 clases, y sano, extracción y
# fractura no son clases de DENTEX: se descartan.
TRADUCCION_PRUEBA = {
    "çürük": "Caries",
    "gömülü": "Impacted",
    "lezyon": "Periapical Lesion",
}

# Nombre clinico por posicion FDI (1..8), del centro hacia atras.
NOMBRE_POSICION = {
    1: "incisivo central",
    2: "incisivo lateral",
    3: "canino",
    4: "primer premolar",
    5: "segundo premolar",
    6: "primer molar",
    7: "segundo molar",
    8: "cordal",
}

# Cuadrantes vistos en la radiografia. Ojo: en imagen medica la izquierda del
# paciente aparece a la DERECHA de la imagen. Es la convencion radiologica y
# confundirla invierte medio odontograma.
NOMBRE_CUADRANTE = {
    1: "superior derecho",
    2: "superior izquierdo",
    3: "inferior izquierdo",
    4: "inferior derecho",
}

# Orden de las piezas en cada arcada tal como aparecen en la radiografia, de
# izquierda a derecha: primero el lado derecho del paciente, del cordal al
# incisivo central, y despues el izquierdo, del incisivo central al cordal.
ARCADAS = {
    "superior": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "inferior": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
}


@dataclass(frozen=True)
class Diente:
    """Una pieza anotada en una radiografia."""

    cuadrante: int  # 1..4
    posicion: int  # 1..8
    bbox: tuple[float, float, float, float]  # x, y, ancho, alto
    poligono: list[float]  # [x1,y1,x2,y2,...]
    patologia: str | None = None

    @property
    def fdi(self) -> int:
        """Numero FDI de la pieza: 11..18, 21..28, 31..38, 41..48."""
        return self.cuadrante * 10 + self.posicion

    @property
    def nombre(self) -> str:
        return f"{NOMBRE_POSICION[self.posicion]} {NOMBRE_CUADRANTE[self.cuadrante]}"

    @property
    def centro(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2


@dataclass(frozen=True)
class Radiografia:
    id: int
    fichero: str
    ancho: int
    alto: int
    dientes: list[Diente]

    @property
    def fdis(self) -> list[int]:
        return sorted(d.fdi for d in self.dientes)

    @property
    def duplicados(self) -> list[int]:
        """Numeros FDI repetidos. Anatomicamente imposible: senala un error."""
        vistos, repes = set(), set()
        for f in self.fdis:
            (repes if f in vistos else vistos).add(f)
        return sorted(repes)


class Subconjunto:
    """Un subconjunto de DENTEX: los dos etiquetados de entrenamiento y los dos
    conjuntos oficiales del concurso (validación y prueba)."""

    # nombre -> (carpeta de imágenes, anotaciones), relativas a data/dentex
    CARPETAS = {
        "enumeracion": (
            "training_data/quadrant_enumeration/xrays",
            "training_data/quadrant_enumeration/train_quadrant_enumeration.json",
        ),
        "patologia": (
            "training_data/quadrant-enumeration-disease/xrays",
            "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
        ),
        "validacion_oficial": (
            "validation_data/quadrant_enumeration_disease/xrays",
            "validation_triple.json",
        ),
        # Una anotación LabelMe por imagen, en esta carpeta.
        "prueba_oficial": ("test_data/disease/input", "test_data/disease/label"),
    }

    def __init__(self, nombre: str, raiz: Path = DATOS):
        if nombre not in self.CARPETAS:
            raise ValueError(f"Subconjunto desconocido: {nombre}")
        imagenes, anotaciones = self.CARPETAS[nombre]
        self.nombre = nombre
        self.dir_imagenes = raiz / imagenes
        self.ruta_json = raiz / anotaciones
        self._huellas: dict[str, str] = {}
        if not self.ruta_json.exists():
            raise FileNotFoundError(
                f"No encuentro {self.ruta_json}. Descomprime el dataset primero."
            )

    @cached_property
    def _bruto(self) -> dict:
        return json.loads(self.ruta_json.read_text())

    @cached_property
    def radiografias(self) -> list[Radiografia]:
        if self.nombre == "prueba_oficial":
            return self._labelme()
        por_imagen: dict[int, list[Diente]] = {}
        for a in self._bruto["annotations"]:
            pat = None
            if "category_id_3" in a:
                idx = a["category_id_3"]
                pat = PATOLOGIAS[idx] if 0 <= idx < len(PATOLOGIAS) else None
            poli = a["segmentation"][0] if a.get("segmentation") else []
            por_imagen.setdefault(a["image_id"], []).append(
                Diente(
                    cuadrante=a["category_id_1"] + 1,
                    posicion=a["category_id_2"] + 1,
                    bbox=tuple(a["bbox"]),
                    poligono=poli,
                    patologia=pat,
                )
            )

        salida = []
        for im in self._bruto["images"]:
            salida.append(
                Radiografia(
                    id=im["id"],
                    fichero=im["file_name"],
                    ancho=im["width"],
                    alto=im["height"],
                    dientes=por_imagen.get(im["id"], []),
                )
            )
        return salida

    def _labelme(self) -> list[Radiografia]:
        salida = []
        for i, f in enumerate(sorted(self.ruta_json.glob("*.json"))):
            datos = json.loads(f.read_text())
            dientes = []
            for forma in datos["shapes"]:
                _, clase, fdi = forma["label"].split("-")
                if clase not in TRADUCCION_PRUEBA:
                    continue
                xs = [x for x, _ in forma["points"]]
                ys = [y for _, y in forma["points"]]
                dientes.append(Diente(
                    cuadrante=int(fdi) // 10,
                    posicion=int(fdi) % 10,
                    bbox=(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                    poligono=[v for punto in forma["points"] for v in punto],
                    patologia=TRADUCCION_PRUEBA[clase],
                ))
            salida.append(Radiografia(
                id=i,
                fichero=f.with_suffix(".png").name,
                ancho=datos["imageWidth"],
                alto=datos["imageHeight"],
                dientes=dientes,
            ))
        return salida

    def ruta(self, r: Radiografia) -> Path:
        return self.dir_imagenes / r.fichero

    def huella(self, r: Radiografia) -> str:
        """MD5 del fichero: identifica copias exactas con distinto nombre."""
        if r.fichero not in self._huellas:
            self._huellas[r.fichero] = hashlib.md5(self.ruta(r).read_bytes()).hexdigest()
        return self._huellas[r.fichero]

    def __len__(self) -> int:
        return len(self.radiografias)

    def __getitem__(self, i: int) -> Radiografia:
        return self.radiografias[i]
