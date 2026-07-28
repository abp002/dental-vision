"""Separacion en entrenamiento, validacion y prueba.

Tres decisiones se toman aqui, y las tres se documentan porque afectan a como
hay que leer cualquier metrica posterior.

1. SE EXCLUYEN LAS IMAGENES CON FDI DUPLICADO.
   Son un 2,5% y tienen anotaciones superpuestas. Entrenar con ellas ensena al
   modelo a predecir dos veces el mismo diente; evaluar con ellas hace imposible
   saber si un fallo es del modelo o de la etiqueta.

2. LA PARTICION ES POR IMAGEN.
   Lo correcto en imagen medica es separar por PACIENTE: si dos radiografias de
   la misma boca caen una en entrenamiento y otra en prueba, el modelo ya ha
   visto ese caso y las metricas salen infladas. DENTEX no publica identificador
   de paciente, asi que asumimos una imagen por paciente. Es un supuesto, no un
   hecho, y queda escrito aqui.

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
) -> Particion:
    limpias, fuera = [], []
    for r in sub.radiografias:
        if excluir_corruptas and r.duplicados:
            fuera.append(r)
        elif not r.dientes:
            fuera.append(r)  # imagen sin anotacion: no aporta senal
        else:
            limpias.append(r)

    # Ordenar antes de barajar: el orden de lectura del JSON no esta garantizado
    # entre versiones, y sin esto la semilla no reproduce la misma particion.
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
