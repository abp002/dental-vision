"""Separacion en entrenamiento, validacion y prueba.

Tres decisiones se toman aqui, y las tres se documentan porque afectan a como
hay que leer cualquier metrica posterior.

1. SE EXCLUYEN LAS IMAGENES CON FDI DUPLICADO.
   Son un 2,5% y tienen anotaciones superpuestas. Entrenar con ellas ensena al
   modelo a predecir dos veces el mismo diente; evaluar con ellas hace imposible
   saber si un fallo es del modelo o de la etiqueta.

2. LA PARTICION ES POR IMAGEN, SIN DUPLICADOS.
   Lo correcto en imagen medica es separar por PACIENTE: si dos radiografias de
   la misma boca caen una en entrenamiento y otra en prueba, el modelo ya ha
   visto ese caso y las metricas salen infladas. DENTEX no publica identificador
   de paciente, asi que asumimos una imagen por paciente. Es un supuesto, no un
   hecho, y queda escrito aqui.

   Lo que si se puede comprobar es la copia exacta: DENTEX repite algunas
   radiografias con distinto nombre de fichero (11 de las 634 de enumeracion).
   De cada grupo de copias identicas se queda la primera por nombre y el resto
   se excluye. Sin esto, una radiografia de prueba estaba tambien en
   entrenamiento. `excluir` permite ademas apartar imagenes que se usan como
   prueba en otro sitio (los conjuntos oficiales del concurso).

3. LA SEMILLA ES FIJA.
   Misma particion en cada ejecucion. Sin esto, comparar dos entrenamientos no
   significa nada: parte de la diferencia vendria de que el reparto cambio.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .dentex import Radiografia, Subconjunto

SEMILLA = 20260728


@dataclass(frozen=True)
class Particion:
    entrenamiento: list[Radiografia]
    validacion: list[Radiografia]
    prueba: list[Radiografia]
    excluidas: list[Radiografia]

    def resumen(self) -> str:
        total = len(self.entrenamiento) + len(self.validacion) + len(self.prueba)
        return (
            f"entrenamiento {len(self.entrenamiento):>4}  "
            f"validacion {len(self.validacion):>4}  "
            f"prueba {len(self.prueba):>4}  "
            f"(total {total}, excluidas {len(self.excluidas)})"
        )


def particionar(
    sub: Subconjunto,
    *,
    val: float = 0.15,
    prueba: float = 0.15,
    semilla: int = SEMILLA,
    excluir_corruptas: bool = True,
    excluir: frozenset[str] = frozenset(),
) -> Particion:
    limpias, fuera = [], []
    vistas: set[str] = set(excluir)
    for r in sorted(sub.radiografias, key=lambda r: r.fichero):
        huella = sub.huella(r)
        if huella in vistas:
            fuera.append(r)  # copia exacta de otra radiografia
            continue
        vistas.add(huella)
        if excluir_corruptas and r.duplicados:
            fuera.append(r)
        elif not r.dientes:
            fuera.append(r)  # imagen sin anotacion: no aporta senal
        else:
            limpias.append(r)

    # Ordenar antes de barajar: el orden de lectura del JSON no esta garantizado
    # entre versiones, y sin esto la semilla no reproduce la misma particion.
    # (Ya vienen ordenadas del bucle; se deja explicito.)
    limpias.sort(key=lambda r: r.fichero)
    random.Random(semilla).shuffle(limpias)

    n = len(limpias)
    n_val = round(n * val)
    n_pru = round(n * prueba)
    return Particion(
        entrenamiento=limpias[n_val + n_pru :],
        validacion=limpias[:n_val],
        prueba=limpias[n_val : n_val + n_pru],
        excluidas=fuera,
    )


OFICIALES = ("validacion_oficial", "prueba_oficial")


def particion_de(tarea: str) -> tuple[Subconjunto, Particion]:
    """Subconjunto y partición de cada tarea.

    Numeración: 70/15/15 sobre el subconjunto de enumeración.
    Patología: 85/15 sobre el de patología, sin prueba propia, porque la prueba
    son los conjuntos oficiales del concurso; por eso se apartan las copias de
    esos conjuntos que aparecen entre las imágenes de entrenamiento.
    """
    if tarea == "numeracion":
        sub = Subconjunto("enumeracion")
        return sub, particionar(sub)
    if tarea == "patologia":
        sub = Subconjunto("patologia")
        oficiales = frozenset(
            s.huella(r) for s in map(Subconjunto, OFICIALES) for r in s.radiografias
        )
        return sub, particionar(
            sub, val=0.15, prueba=0.0, excluir_corruptas=False, excluir=oficiales
        )
    raise ValueError(f"Tarea desconocida: {tarea}")
