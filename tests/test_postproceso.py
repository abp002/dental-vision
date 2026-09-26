import pytest

from dentalvision.datos import CLASE_A_FDI
from dentalvision.postproceso import agrupar, postprocesar


def fdis(salida) -> list[int]:
    """Números FDI de la salida, de izquierda a derecha en la imagen."""
    orden = salida["boxes"][:, 0].argsort()
    return [CLASE_A_FDI[int(c)] for c in salida["labels"][orden]]


def test_agrupar_junta_las_hipotesis_de_una_misma_pieza(pred):
    p = pred((0, 0, 10, 20, 16, 0.9), (0, 0, 10, 21, 17, 0.6), (30, 0, 40, 20, 15, 0.8))
    grupos = agrupar(p)
    assert len(grupos) == 2
    assert grupos[0].fdis == pytest.approx({16: 0.9, 17: 0.6})
    assert grupos[1].fdi == 15


def test_ninguno_solo_filtra_por_confianza(pred):
    p = pred((0, 0, 10, 20, 16, 0.9), (0, 0, 10, 20, 17, 0.6), (30, 0, 40, 20, 15, 0.3))
    assert sorted(fdis(postprocesar(p, "ninguno", umbral=0.5))) == [16, 17]


def test_unicidad_quita_duplicados(pred):
    # Misma pieza vista como 16 y 17, y otra pieza también vista como 16.
    p = pred((0, 0, 10, 20, 16, 0.9), (0, 0, 10, 20, 17, 0.6), (20, 0, 30, 20, 16, 0.7))
    salida = postprocesar(p, "unicidad")
    assert fdis(salida).count(16) == 1


def test_orden_renumera_segun_la_fila(pred):
    # Cuadrante 1 (izquierda de la imagen): de izquierda a derecha 18..11.
    # Dos piezas contiguas vistas como 16: la de la derecha tiene que ser 15.
    p = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 16, 0.7))
    assert fdis(postprocesar(p, "orden")) == [16, 15]


def test_orden_en_la_arcada_inferior(pred):
    # Cuadrante 4 (abajo a la izquierda): de izquierda a derecha 48..41.
    p = pred((0, 50, 10, 70, 46, 0.9), (20, 50, 30, 70, 46, 0.7))
    assert fdis(postprocesar(p, "orden")) == [46, 45]


def test_orden_cruza_la_linea_media(pred):
    # Dos incisivos centrales superiores vistos como 11: el de la derecha de la
    # imagen es el izquierdo del paciente, 21.
    p = pred((0, 0, 10, 20, 11, 0.9), (20, 0, 30, 20, 11, 0.6))
    assert fdis(postprocesar(p, "orden")) == [11, 21]


def test_orden_respeta_los_huecos(pred):
    # Falta el 16: el 17 y el 15 no deben moverse para "rellenarlo".
    p = pred((0, 0, 10, 20, 17, 0.9), (40, 0, 50, 20, 15, 0.9))
    assert fdis(postprocesar(p, "orden")) == [17, 15]


def test_una_pieza_dudosa_no_se_pierde(pred):
    # El detector duda entre 16 (0,45) y 17 (0,40): ninguna hipótesis pasa 0,5,
    # pero juntas dicen que ahí hay un diente.
    p = pred((0, 0, 10, 20, 16, 0.45), (0, 0, 10, 20, 17, 0.40))
    assert fdis(postprocesar(p, "ninguno", umbral=0.5)) == []
    assert fdis(postprocesar(p, "orden", umbral=0.5)) == [16]


def test_sin_detecciones(pred):
    vacio = pred()
    for modo in ("ninguno", "unicidad", "orden"):
        assert len(postprocesar(vacio, modo)["boxes"]) == 0
