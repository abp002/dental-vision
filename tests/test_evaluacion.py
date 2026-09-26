import pytest

from dentalvision.evaluacion import (
    Resultado,
    evaluar,
    evaluar_por_radiografia,
    intervalo,
    intervalo_diferencia,
)


def test_prediccion_perfecta(pred):
    real = pred((0, 0, 10, 20, 16, 1.0), (20, 0, 30, 20, 15, 1.0))
    r = evaluar([real], [real])
    assert (r.cobertura, r.numeracion, r.exactitud, r.perfectas) == (1, 1, 1, 1)
    assert r.errores_por_radiografia == 0


def test_con_dos_predicciones_sobre_un_diente_gana_la_mas_segura(pred):
    real = pred((0, 0, 10, 20, 16, 1.0))
    # La más segura dice 17 (mal) y la otra 16 (bien): cuenta la segura, y la
    # otra es un falso positivo aunque acierte el número.
    p = pred((0, 0, 10, 20, 17, 0.9), (0, 0, 10, 20, 16, 0.8))
    r = evaluar([p], [real])
    assert r.piezas_encontradas == 1
    assert r.numeracion_correcta == 0
    assert r.falsos_positivos == 1
    assert r.errores_por_radiografia == 2


def test_si_su_mejor_pareja_esta_ocupada_prueba_la_siguiente(pred):
    # Dos dientes que se solapan. Las dos predicciones solapan más con el
    # primero (IoU 0,90 y 0,74); la segunda, al encontrarlo ocupado, debe
    # emparejarse con el otro (IoU 0,60) en lugar de quedarse sin pareja.
    real = pred((0, 0, 10, 20, 16, 1.0), (4, 0, 14, 20, 15, 1.0))
    p = pred((0.5, 0, 10.5, 20, 16, 0.9), (1.5, 0, 11.5, 20, 15, 0.8))
    r = evaluar([p], [real])
    assert r.piezas_encontradas == 2
    assert r.numeracion_correcta == 2


def test_pieza_no_encontrada_y_umbral_de_confianza(pred):
    real = pred((0, 0, 10, 20, 16, 1.0), (20, 0, 30, 20, 15, 1.0))
    p = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 15, 0.3))
    r = evaluar([p], [real], umbral_score=0.5)
    assert r.cobertura == 0.5
    assert r.numeracion == 1
    assert r.exactitud == 0.5
    assert r.perfectas == 0


def test_resultados_se_suman():
    a = Resultado(piezas_reales=28, numeracion_correcta=27, radiografias=1)
    b = Resultado(piezas_reales=30, numeracion_correcta=30, radiografias=1, radiografias_perfectas=1)
    assert (a + b).piezas_reales == 58
    assert (a + b).radiografias_perfectas == 1


def test_intervalo_contiene_la_estimacion(pred):
    real = pred((0, 0, 10, 20, 16, 1.0), (20, 0, 30, 20, 15, 1.0))
    bien = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 15, 0.9))
    mal = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 14, 0.9))
    por_rx = evaluar_por_radiografia([bien, mal] * 10, [real] * 20)
    lo, hi = intervalo(por_rx, lambda r: r.perfectas)
    assert lo <= 0.5 <= hi
    assert hi - lo == pytest.approx(0.4, abs=0.2)


def test_diferencia_emparejada(pred):
    real = pred((0, 0, 10, 20, 16, 1.0), (20, 0, 30, 20, 15, 1.0))
    bien = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 15, 0.9))
    mal = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 14, 0.9))
    # Mismo modelo contra sí mismo: diferencia exactamente 0.
    a = evaluar_por_radiografia([bien, mal] * 10, [real] * 20)
    assert intervalo_diferencia(a, a, lambda r: r.errores_por_radiografia) == (0, 0)
    # b acierta en todas: siempre mejor, el intervalo no toca el 0.
    b = evaluar_por_radiografia([bien] * 20, [real] * 20)
    lo, hi = intervalo_diferencia(a, b, lambda r: r.errores_por_radiografia)
    assert hi < 0
