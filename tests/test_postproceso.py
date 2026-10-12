import pytest
import torch

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


def dudas(salida) -> dict[int, tuple[float, int]]:
    """FDI -> (pérdida, alternativa) de cada pieza."""
    return {
        CLASE_A_FDI[int(c)]: (float(p), int(a))
        for c, p, a in zip(salida["labels"], salida["perdidas"], salida["alternativas"])
    }


def test_la_duda_no_cambia_la_numeracion(pred):
    p = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 16, 0.7), (40, 0, 50, 20, 14, 0.5),
             (40, 0, 50, 20, 13, 0.45))
    assert fdis(postprocesar(p, "orden", dudas=True)) == fdis(postprocesar(p, "orden"))


def test_una_pieza_entre_dos_numeros_pierde_poco_sin_el_suyo(pred):
    # Dos piezas en el cuadrante 1: una firme (17) y otra que duda entre 15 y 14.
    p = pred((0, 0, 10, 20, 17, 0.9), (40, 0, 50, 20, 15, 0.5), (40, 0, 50, 20, 14, 0.45))
    d = dudas(postprocesar(p, "orden", dudas=True))
    assert d[15][1] == 14  # la alternativa es el número con el que dudaba
    assert d[15][0] == pytest.approx(0.05)
    assert d[17][0] > 10 * d[15][0]
    assert all(perdida >= 0 for perdida, _ in d.values())


def test_la_alternativa_de_una_fila_apretada_es_el_vecino(pred):
    # Tres piezas contiguas 16, 15, 14, todas seguras: sin el suyo, el 14 solo
    # puede correrse hacia el 13 (el 15 está ocupado).
    p = pred((0, 0, 10, 20, 16, 0.9), (20, 0, 30, 20, 15, 0.9), (40, 0, 50, 20, 14, 0.9))
    d = dudas(postprocesar(p, "orden", kappa=0.7, dudas=True))
    assert d[14][1] == 13
    assert d[14][0] == pytest.approx(0.9 * (1 - 0.7))


# --- Detector de arcada y posición (sin el lado) --------------------------------


@pytest.fixture
def pred_posicion():
    """Como `pred`, pero con (x1, y1, x2, y2, (arcada, posición), score)."""
    from dentalvision.datos import POSICION_A_CLASE

    def construir(*detecciones):
        return {
            "boxes": torch.tensor([d[:4] for d in detecciones], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([POSICION_A_CLASE[d[4]] for d in detecciones], dtype=torch.int64),
            "scores": torch.tensor([d[5] for d in detecciones], dtype=torch.float32),
        }

    return construir


def test_sin_orden_el_lado_sale_de_la_mitad_de_la_imagen(pred_posicion):
    # Primer molar superior a cada lado del centro (x = 100).
    p = pred_posicion((0, 0, 10, 20, ("superior", 6), 0.9), (190, 0, 200, 20, ("superior", 6), 0.9),
                      (0, 50, 10, 70, ("inferior", 6), 0.9))
    salida = postprocesar(p, "ninguno", etiquetas="posicion", centro=100)
    assert sorted(fdis(salida)) == [16, 26, 46]
    assert len(salida["boxes"]) == 3  # sin el lado contrario como pieza aparte


def test_el_orden_de_la_fila_manda_sobre_la_mitad(pred_posicion):
    # Los dos incisivos centrales caen a la izquierda del centro (paciente algo
    # desplazado): por la mitad serían los dos 11, pero en fila son 11 y 21.
    p = pred_posicion((70, 0, 80, 20, ("superior", 1), 0.9), (85, 0, 95, 20, ("superior", 1), 0.9))
    assert fdis(postprocesar(p, "orden", etiquetas="posicion", centro=100, beta=0.5)) == [11, 21]


def test_los_dos_lados_no_suman_presencia(pred_posicion):
    # 0,4 no pasa un umbral de 0,5 aunque su copia del otro lado añada 0,4·beta.
    p = pred_posicion((0, 0, 10, 20, ("superior", 6), 0.4))
    assert len(postprocesar(p, "orden", umbral=0.5, etiquetas="posicion", centro=100, beta=0.9)["boxes"]) == 0


def test_los_indices_apuntan_a_la_deteccion_original(pred_posicion):
    # Los índices sirven para recuperar la máscara de la salida del detector.
    p = pred_posicion((190, 0, 200, 20, ("superior", 6), 0.6), (0, 0, 10, 20, ("superior", 7), 0.9))
    salida = postprocesar(p, "orden", etiquetas="posicion", centro=100)
    por_fdi = dict(zip(fdis(salida), sorted(salida["indices"].tolist(), key=lambda i: p["boxes"][i, 0])))
    assert por_fdi == {17: 1, 26: 0}


def test_posicion_sin_centro_es_un_error(pred_posicion):
    with pytest.raises(ValueError):
        postprocesar(pred_posicion(), "orden", etiquetas="posicion")
