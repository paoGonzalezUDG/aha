"""Pruebas rapidas:  python -m pytest pruebas"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from algoritmo.aha import aha, evaluaciones_esperadas  # noqa: E402
from algoritmo.funciones import FUNCIONES  # noqa: E402
from experimentos import TABLA_25, correr_fila, filas_funcion  # noqa: E402

CSV = os.path.join(RAIZ, "resultados", "tabla_25", "corridas.csv")

# Columna "Evaluaciones" del PDF del plan, ejecuciones 1 a 25
EVALUACIONES_PLAN = [50060] * 5 + [10060, 20045, 30047, 80087, 100105,
                                   5051, 12553, 25055, 75065, 100070,
                                   50060, 50060, 50090, 50054, 50050] \
    + [50060] * 5


def test_evaluaciones_del_plan():
    for fila, esperado in zip(filas_funcion("Sphere"), EVALUACIONES_PLAN):
        p = fila[2]
        assert evaluaciones_esperadas(p["n_colibries"], p["max_iter"],
                                      p["coef_migracion"]) == esperado


def test_cuenta_real_de_evaluaciones():
    cfg = FUNCIONES["Sphere"]
    for n, T, m in [(10, 60, 2.0), (7, 30, 0.5), (5, 20, 0.0)]:
        r = aha(cfg["f"], cfg["lb"], cfg["ub"], 3, n_colibries=n,
                max_iter=T, coef_migracion=m, seed=1)
        assert r["evaluaciones"] == evaluaciones_esperadas(n, T, m)


def test_misma_semilla_mismo_resultado():
    cfg = FUNCIONES["Rastrigin"]
    a = aha(cfg["f"], cfg["lb"], cfg["ub"], 5, 20, 50, seed=3)
    b = aha(cfg["f"], cfg["lb"], cfg["ub"], 5, 20, 50, seed=3)
    c = aha(cfg["f"], cfg["lb"], cfg["ub"], 5, 20, 50, seed=4)
    assert a["mejor_f"] == b["mejor_f"]
    assert np.array_equal(a["curva"], b["curva"])
    assert not np.array_equal(a["curva"], c["curva"])


@pytest.mark.parametrize("modo", ["mixto", "axial", "diagonal",
                                  "omnidireccional"])
def test_modos_de_vuelo_y_zona(modo):
    cfg = FUNCIONES["Beale"]
    r = aha(cfg["f"], cfg["lb"], cfg["ub"], 2, 15, 40, modo_vuelo=modo,
            seed=0, guardar_posiciones=True)
    P = np.array(r["posiciones"])
    assert P.min() >= cfg["lb"] and P.max() <= cfg["ub"]
    assert np.all(np.diff(r["curva"]) <= 0)          # nunca empeora
    assert r["mejor_f"] == pytest.approx(cfg["f"](r["mejor_x"]))


def test_axial_mueve_las_dos_coordenadas():
    """En AHA_python.py el vuelo axial nunca movia x (bug corregido)."""
    cfg = FUNCIONES["Sphere"]
    r = aha(cfg["f"], cfg["lb"], cfg["ub"], 2, 10, 30, modo_vuelo="axial",
            coef_migracion=0, seed=0, guardar_posiciones=True)
    P = np.array(r["posiciones"])
    cambio = np.abs(np.diff(P, axis=0)).sum(axis=(0, 1))
    assert cambio[0] > 0 and cambio[1] > 0


def test_sphere_converge():
    cfg = FUNCIONES["Sphere"]
    r = aha(cfg["f"], cfg["lb"], cfg["ub"], 10, 30, 300, seed=0)
    assert r["mejor_f"] < 1e-20


@pytest.mark.skipif(not os.path.exists(CSV), reason="falta corridas.csv")
def test_resultados_guardados_se_repiten():
    """Re-corre algunas filas y las compara con corridas.csv."""
    df = pd.read_csv(CSV)
    assert len(df) == 25 * len(FUNCIONES)
    assert list(df.groupby("funcion").evaluaciones.apply(list)["Beale"]) \
        == EVALUACIONES_PLAN
    for nombre, num in [("Beale", 6), ("Ackley", 11), ("Rastrigin", 12)]:
        _, _, p, dim, seed, _ = filas_funcion(nombre)[num - 1]
        r = correr_fila(nombre, p, dim, seed)
        fila = df[(df.funcion == nombre) & (df.ejecucion == num)].iloc[0]
        assert r["mejor_f"] == pytest.approx(fila.best_f, rel=1e-9, abs=0)
        assert r["iter_convergencia"] == fila.iter_convergencia


def test_tabla_25_completa():
    assert [t[0] for t in TABLA_25] == list(range(1, 26))
    assert filas_funcion("Rosenbrock")[24][3] == 30
    assert filas_funcion("Ackley")[24][3] == 2


def test_app_carga_sin_errores():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
