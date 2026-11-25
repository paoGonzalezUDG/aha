"""
Las 5 funciones del plan experimental (las primeras 5 de Wikipedia "Test
functions for optimization"), con los rangos y dimensiones del plan:

    Funcion      Dimension          Rango
    Rastrigin    10 (30 en Ej. 25)  [-5.12, 5.12]
    Ackley       2 (fija)           [-32.768, 32.768]
    Sphere       10 (30 en Ej. 25)  [-100, 100]
    Rosenbrock   10 (30 en Ej. 25)  [-30, 30]
    Beale        2 (fija)           [-4.5, 4.5]

Todas son de MINIMIZACION y su minimo vale 0. Cada funcion recibe un vector
numpy x.
"""
import numpy as np


def rastrigin(x, A=10.0):
    x = np.asarray(x)
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def ackley(x):
    x, y = x[0], x[1]
    return (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + np.e + 20.0)


def sphere(x):
    x = np.asarray(x)
    return np.sum(x ** 2)


def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def beale(x):
    x, y = x[0], x[1]
    return ((1.5 - x + x * y) ** 2
            + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


# dim = None -> n variables (10 en el plan, 30 en la ejecucion 25)
# dim = 2    -> la funcion solo existe con 2 variables
FUNCIONES = {
    "Rastrigin": dict(
        f=rastrigin, dim=None, dim_default=10, lb=-5.12, ub=5.12,
        optimo_x=lambda n: np.zeros(n), optimo_f=0.0,
        apodo="Cartón de huevo",
        paisaje="Altamente multimodal con múltiples mínimos locales "
                "regulares.",
        formula="f(x) = 10n + Σ [ x_i² − 10 cos(2π x_i) ]"),
    "Ackley": dict(
        f=ackley, dim=2, dim_default=2, lb=-32.768, ub=32.768,
        optimo_x=lambda n: np.zeros(2), optimo_f=0.0,
        apodo="Llanura con un pozo",
        paisaje="Meseta exterior casi plana con un embudo central profundo.",
        formula="f(x,y) = −20 exp(−0.2 √(0.5(x²+y²))) "
                "− exp(0.5(cos 2πx + cos 2πy)) + e + 20"),
    "Sphere": dict(
        f=sphere, dim=None, dim_default=10, lb=-100.0, ub=100.0,
        optimo_x=lambda n: np.zeros(n), optimo_f=0.0,
        apodo="Tazón",
        paisaje="Unimodal, convexa y simétrica; mide velocidad de "
                "convergencia.",
        formula="f(x) = Σ x_i²"),
    "Rosenbrock": dict(
        f=rosenbrock, dim=None, dim_default=10, lb=-30.0, ub=30.0,
        optimo_x=lambda n: np.ones(n), optimo_f=0.0,
        apodo="Valle del plátano",
        paisaje="Valle estrecho y curvo (forma de banana); difícil sin "
                "gradientes.",
        formula="f(x) = Σ [ 100 (x_{i+1} − x_i²)² + (1 − x_i)² ]"),
    "Beale": dict(
        f=beale, dim=2, dim_default=2, lb=-4.5, ub=4.5,
        optimo_x=lambda n: np.array([3.0, 0.5]), optimo_f=0.0,
        apodo="Valles angostos",
        paisaje="Multimodal asimétrica con picos pronunciados y valles "
                "planos.",
        formula="f(x,y) = (1.5 − x + xy)² + (2.25 − x + xy²)² "
                "+ (2.625 − x + xy³)²"),
}
