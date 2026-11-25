"""
Plan experimental del AHA: 25 ejecuciones por funcion (125 corridas), UNA
corrida por fila con su semilla. Genera resultados/tabla_25/.

Uso (desde la raiz del proyecto):
    python experimentos.py           (las 125 corridas, ~4 min)
    python experimentos.py --figuras (solo redibujar las PNG desde los CSV)

Archivos que salen:
    corridas.csv               una fila por ejecucion con todo lo que pide el
                               plan (best_f, log10, evaluaciones, vuelta de
                               convergencia, tiempo, mejoras, best_x)
    historial_<Funcion>.csv    his_best_fit: mejor valor en cada vuelta
                               (columna ej_1 ... ej_25)
    resultados.png             mapa de calor funcion x ejecucion
    convergencia_<n>_<Funcion>.png   curvas por bloque experimental
    mejoras.png                mejoras por estrategia
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from algoritmo.aha import aha, evaluaciones_esperadas, log10_error
from algoritmo.funciones import FUNCIONES

CARPETA = "resultados/tabla_25"
CSV_CORRIDAS = f"{CARPETA}/corridas.csv"

# Valores base (Ej. 1) del plan
PARAMS_DEFAULT = dict(n_colibries=50, max_iter=1000, p_guiado=0.5,
                      coef_migracion=2.0, modo_vuelo="mixto", sigma=1.0)
# Nombre de cada parametro en el plan (PDF)
NOMBRE_PLAN = dict(n_colibries="npop", max_iter="max_it",
                   p_guiado="p_guided", coef_migracion="mig_coef",
                   modo_vuelo="flight_mode", sigma="sigma")

# (ejecucion, bloque, cambios respecto a la base, semilla, que se evalua)
TABLA_25 = [
    (1, "1. Base oficial y azar", {}, 0, "Configuración base (referencia)."),
    (2, "1. Base oficial y azar", {}, 1, "Misma base, otro azar."),
    (3, "1. Base oficial y azar", {}, 2, "Misma base, otro azar."),
    (4, "1. Base oficial y azar", {}, 3, "Misma base, otro azar."),
    (5, "1. Base oficial y azar", {}, 4, "Misma base, otro azar."),
    (6, "2. Tamaño de bandada (N)", {"n_colibries": 10}, 0,
     "Bandada muy chica."),
    (7, "2. Tamaño de bandada (N)", {"n_colibries": 20}, 0,
     "Bandada chica."),
    (8, "2. Tamaño de bandada (N)", {"n_colibries": 30}, 0,
     "Bandada mediana."),
    (9, "2. Tamaño de bandada (N)", {"n_colibries": 80}, 0,
     "Bandada grande."),
    (10, "2. Tamaño de bandada (N)", {"n_colibries": 100}, 0,
     "Bandada muy grande."),
    (11, "3. Ciclos (max_it)", {"max_iter": 100}, 0, "Muy pocas vueltas."),
    (12, "3. Ciclos (max_it)", {"max_iter": 250}, 0, "Pocas vueltas."),
    (13, "3. Ciclos (max_it)", {"max_iter": 500}, 0, "Mitad de la base."),
    (14, "3. Ciclos (max_it)", {"max_iter": 1500}, 0, "Más vueltas."),
    (15, "3. Ciclos (max_it)", {"max_iter": 2000}, 0, "El doble de la base."),
    (16, "4. Forrajeo y migración", {"p_guiado": 0.2}, 0,
     "Casi siempre territorial (20% guiado)."),
    (17, "4. Forrajeo y migración", {"p_guiado": 0.8}, 0,
     "Casi siempre guiado (80% guiado)."),
    (18, "4. Forrajeo y migración", {"coef_migracion": 0.5}, 0,
     "Migración muy seguida (cada 25 vueltas)."),
    (19, "4. Forrajeo y migración", {"coef_migracion": 5.0}, 0,
     "Migración rara (cada 250 vueltas)."),
    (20, "4. Forrajeo y migración", {"coef_migracion": 0.0}, 0,
     "Sin migración."),
    (21, "5. Vuelos, paso y 30D", {"modo_vuelo": "axial"}, 0,
     "Solo vuelo axial (1 coordenada)."),
    (22, "5. Vuelos, paso y 30D", {"modo_vuelo": "diagonal"}, 0,
     "Solo vuelo diagonal (varias coordenadas)."),
    (23, "5. Vuelos, paso y 30D", {"modo_vuelo": "omnidireccional"}, 0,
     "Solo vuelo omnidireccional (todas)."),
    (24, "5. Vuelos, paso y 30D", {"sigma": 0.5}, 0,
     "Pasos más cortos (sigma = 0.5)."),
    (25, "5. Vuelos, paso y 30D", {"dim": 30}, 0,
     "30 variables (Ackley y Beale siguen en 2)."),
]
BLOQUES = list(dict.fromkeys(b for _, b, _, _, _ in TABLA_25))


def filas_funcion(nombre_f):
    """Las 25 filas de una funcion: (num, bloque, params, dim, seed, texto).
    Ackley y Beale siempre van en 2 variables (en la fila 25 tambien)."""
    cfg = FUNCIONES[nombre_f]
    filas = []
    for num, bloque, cambios, seed, texto in TABLA_25:
        cambios = dict(cambios)
        d30 = cambios.pop("dim", None)
        dim = cfg["dim"] or d30 or cfg["dim_default"]
        filas.append((num, bloque, {**PARAMS_DEFAULT, **cambios}, int(dim),
                      seed, texto))
    return filas


def correr_fila(nombre_f, p, dim, seed):
    cfg = FUNCIONES[nombre_f]
    t0 = time.perf_counter()
    r = aha(cfg["f"], cfg["lb"], cfg["ub"], dim, seed=seed, **p)
    r["tiempo"] = time.perf_counter() - t0
    return r


def experimento(carpeta=CARPETA):
    """Corre las 25 filas en las 5 funciones y guarda CSV + figuras."""
    os.makedirs(carpeta, exist_ok=True)
    filas = []
    for nombre in FUNCIONES:
        historiales = {}
        for num, bloque, p, dim, seed, texto in filas_funcion(nombre):
            r = correr_fila(nombre, p, dim, seed)
            esperadas = evaluaciones_esperadas(p["n_colibries"],
                                               p["max_iter"],
                                               p["coef_migracion"])
            assert r["evaluaciones"] == esperadas, (nombre, num)
            filas.append({
                "funcion": nombre, "ejecucion": num, "bloque": bloque,
                **p, "dim": dim, "seed": seed,
                "best_f": r["mejor_f"],
                "log10_best_f": log10_error(r["mejor_f"]),
                "evaluaciones": r["evaluaciones"],
                "iter_convergencia": r["iter_convergencia"],
                "tiempo_s": round(r["tiempo"], 4),
                "mejoras_guiado": r["mejoras"]["guiado"],
                "mejoras_territorial": r["mejoras"]["territorial"],
                "mejoras_migracion": r["mejoras"]["migracion"],
                "migraciones": r["migraciones"],
                "distancia_optimo": float(np.linalg.norm(
                    r["mejor_x"] - FUNCIONES[nombre]["optimo_x"](dim))),
                "best_x": json.dumps([float(v) for v in r["mejor_x"]]),
                "que_se_evalua": texto})
            historiales[f"ej_{num}"] = pd.Series(r["historial"])
            print(f"{nombre:<11} ej {num:>2}  best_f = {r['mejor_f']:.3e}  "
                  f"({r['tiempo']:.1f} s)", flush=True)
        h = pd.DataFrame(historiales)
        h.insert(0, "iteracion", np.arange(1, len(h) + 1))
        h.to_csv(f"{carpeta}/historial_{nombre}.csv", index=False)
    df = pd.DataFrame(filas)
    df.to_csv(f"{carpeta}/corridas.csv", index=False)
    figuras(carpeta)
    return df


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
PISO = -16          # log10 del plan para el 0 exacto


def _etiqueta(v):
    if v == 0:
        return "0"
    return f"{v:.0e}".replace("e-0", "e-") if v < 0.01 else f"{v:.2g}"


def figura_mapa(df, plt, ruta):
    """Mapa de calor: funcion x ejecucion, color = log10(best_f)."""
    nombres = list(FUNCIONES)
    nums = sorted(df.ejecucion.unique())
    m = np.array([[df[(df.funcion == f) & (df.ejecucion == k)].best_f.iloc[0]
                   for k in nums] for f in nombres])
    arriba = 2
    color = np.clip(np.log10(np.maximum(m, 1e-300)), PISO, arriba)
    fig, ax = plt.subplots(figsize=(18, 4.6))
    im = ax.imshow(color, cmap="RdYlGn_r", vmin=PISO, vmax=arriba,
                   aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, _etiqueta(m[i, j]), ha="center", va="center",
                    fontsize=7)
    ax.set_xticks(range(len(nums)), [str(k) for k in nums])
    ax.set_yticks(range(len(nombres)), nombres)
    ax.set_xlabel("Ejecución (fila del plan)")
    for x in (4.5, 9.5, 14.5, 19.5):            # separa los 5 bloques
        ax.axvline(x, color="k", lw=1.5)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_ticks([PISO, -12, -8, -4, 0, arriba])
    cb.set_ticklabels(["≤1e-16", "1e-12", "1e-8", "1e-4", "1", "100"])
    cb.set_label("Mejor valor (verde = casi 0)")
    ax.set_title("AHA: mejor valor de cada ejecución (1 corrida por fila)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_convergencia(df, hist, nombre, plt, ruta):
    """5 paneles (uno por bloque) con el mejor valor vuelta por vuelta."""
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
    d = df[df.funcion == nombre]
    for ax, bloque in zip(axs, BLOQUES):
        for _, fila in d[d.bloque == bloque].iterrows():
            y = hist[f"ej_{fila.ejecucion}"].dropna().to_numpy()
            y = np.maximum(y, 10.0 ** PISO)
            ax.semilogy(np.arange(1, len(y) + 1), y, lw=1.4,
                        label=f"{fila.ejecucion}: {_param_fila(fila)}")
        ax.set_title(bloque, fontsize=10)
        ax.set_xlabel("Vuelta")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Mejor valor (log; 0 se dibuja en 1e-16)")
    fig.suptitle(f"{nombre}: cómo fue mejorando cada ejecución")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def _param_fila(fila):
    """Lo que distingue a la fila (para la leyenda)."""
    cambios = TABLA_25[int(fila.ejecucion) - 1][2]
    if not cambios:
        return f"semilla {fila.seed}"
    k, v = next(iter(cambios.items()))
    if k == "dim":
        return f"{fila.dim} variables"
    return f"{NOMBRE_PLAN[k]} = {v}"


def figura_mejoras(df, plt, ruta):
    """Mejoras por estrategia en cada ejecucion (barras apiladas)."""
    nombres = list(FUNCIONES)
    fig, axs = plt.subplots(len(nombres), 1, figsize=(14, 12), sharex=True)
    colores = {"mejoras_guiado": "#1f77b4", "mejoras_territorial": "#2ca02c",
               "mejoras_migracion": "#ff7f0e"}
    for ax, nombre in zip(axs, nombres):
        d = df[df.funcion == nombre].sort_values("ejecucion")
        abajo = np.zeros(len(d))
        for col, c in colores.items():
            ax.bar(d.ejecucion, d[col], bottom=abajo, color=c,
                   label=col.replace("mejoras_", "").capitalize())
            abajo += d[col].to_numpy()
        ax.set_ylabel(nombre)
        ax.grid(axis="y", alpha=.3)
    axs[0].legend(ncol=3, fontsize=9, loc="upper right")
    axs[-1].set_xticks(range(1, 26))
    axs[-1].set_xlabel("Ejecución (fila del plan)")
    fig.suptitle("Cuántas veces cada estrategia encontró una flor mejor")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def figuras(carpeta=CARPETA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(f"{carpeta}/corridas.csv")
    figura_mapa(df, plt, f"{carpeta}/resultados.png")
    figura_mejoras(df, plt, f"{carpeta}/mejoras.png")
    for k, nombre in enumerate(FUNCIONES, 1):
        hist = pd.read_csv(f"{carpeta}/historial_{nombre}.csv")
        figura_convergencia(df, hist, nombre, plt,
                            f"{carpeta}/convergencia_{k}_{nombre}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figuras", action="store_true",
                    help="solo redibujar las PNG desde los CSV")
    args = ap.parse_args()
    if args.figuras:
        figuras()
    else:
        t0 = time.perf_counter()
        experimento()
        print(f"Listo en {(time.perf_counter() - t0) / 60:.1f} min. "
              f"Resultados en {CARPETA}/")


if __name__ == "__main__":
    main()
