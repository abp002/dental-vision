from dentalvision.datos import CLASE_A_FDI, FDI_A_CLASE, fdi_espejado, voltear


def test_espejo_intercambia_cuadrantes_y_conserva_posicion():
    assert fdi_espejado(16) == 26
    assert fdi_espejado(21) == 11
    assert fdi_espejado(38) == 48
    assert fdi_espejado(44) == 34


def test_voltear_mueve_cajas_poligonos_y_renumera():
    cajas, etiquetas, poligonos = voltear(
        [[10, 5, 30, 25]], [FDI_A_CLASE[16]], [[10, 5, 30, 5, 30, 25]], ancho=100
    )
    assert cajas == [[70, 5, 90, 25]]
    assert [CLASE_A_FDI[c] for c in etiquetas] == [26]
    assert poligonos == [[90, 5, 70, 5, 70, 25]]


def test_voltear_dos_veces_deja_todo_igual():
    original = ([[10, 5, 30, 25]], [FDI_A_CLASE[37]], [[10, 5, 30, 5, 30, 25]])
    assert voltear(*voltear(*original, ancho=100), ancho=100) == original
