"""
Artificial Hummingbird Algorithm (AHA) - Zhao, Wang y Mirjalili (2021),
Computer Methods in Applied Mechanics and Engineering 388, 114194.

Basado en AHA_python.py (el codigo que nos dieron), con los parametros del
plan experimental para poder cambiarlos.

Idea: cada "colibri" vive en una flor (una solucion candidata). El valor de
la funcion en esa flor es el "nectar" al reves: entre MAS BAJO el valor,
MEJOR la flor. Cada colibri recuerda hace cuanto no visita cada flor
(tabla de visitas). En cada vuelta, cada colibri:

  1) elige una forma de volar (que coordenadas puede mover):
       axial            -> solo 1 coordenada
       diagonal         -> varias coordenadas (entre 2 y d-1)
       omnidireccional  -> todas las coordenadas
  2) y hace UNA de estas dos busquedas:
       forrajeo guiado     (prob. p_guiado)   -> vuela hacia la flor que
                                                 lleva mas tiempo sin visitar
       forrajeo territorial (prob. 1-p_guiado)-> busca alrededor de su flor
     Solo se cambia de flor si la nueva es mejor (si no, se queda).
  3) Cada coef_migracion * n_colibries vueltas, el colibri con la PEOR flor
     se muda a un lugar al azar (migracion).

Parametros (los del plan experimental):
  n_colibries     npop      tamano de la bandada (50)
  max_iter        max_it    numero de vueltas (1000)
  p_guiado        p_guided  probabilidad de forrajeo guiado (0.5)
  coef_migracion  mig_coef  migracion cada coef * n vueltas (2.0; 0 = nunca)
  modo_vuelo      flight_mode  "mixto" (1/3 cada uno), "axial", "diagonal"
                               u "omnidireccional"
  sigma           sigma     tamano del paso: el numero al azar N(0,1) de las
                            ecuaciones se multiplica por sigma (1.0)
  seed            seed      semilla

Diferencias con AHA_python.py (no cambian la idea del algoritmo):
  - El azar sale de np.random.default_rng(seed) (antes torch.randperm y
    np.random sin semilla): asi cada corrida se puede repetir.
  - Vuelo axial: AHA_python.py elegia la coordenada con ceil(rand*(d-1)),
    que nunca toca la primera coordenada (en 2 variables, x nunca se movia
    en vuelo axial). Aqui se elige cualquiera de las d, como en el codigo
    original de MATLAB de los autores.
  - Vuelo diagonal: mueve entre 2 y d-1 coordenadas (en 2 variables, las 2),
    como en el codigo de MATLAB. AHA_python.py movia entre 1 y d-2.
  - mejor_x se copia (en AHA_python.py era una vista que podia cambiar
    despues sin que cambiara mejor_f).
  - Se cuentan las evaluaciones de la funcion y las mejoras de cada
    estrategia (lo pide el plan).
"""
import numpy as np

MODOS_VUELO = ("mixto", "axial", "diagonal", "omnidireccional")
VUELOS = ("axial", "diagonal", "omnidireccional")      # indices 0, 1, 2
# Accion de cada colibri en una vuelta: 0 guiado, 1 territorial, 2 migro
ACCIONES = ("guiado", "territorial", "migracion")


def _vuelo(rng, dim, modo):
    """Vector de direccion D (1 = esa coordenada se mueve) y tipo de vuelo."""
    r = rng.random()
    if modo == "mixto":
        if r < 1 / 3:
            modo = "diagonal"
        elif r > 2 / 3:
            modo = "omnidireccional"
        else:
            modo = "axial"
    D = np.zeros(dim)
    if modo == "diagonal":
        orden = rng.permutation(dim)
        u = rng.random()
        k = int(np.ceil(u * (dim - 2) + 1)) if dim >= 3 else dim
        D[orden[:max(k, 1)]] = 1
    elif modo == "omnidireccional":
        D[:] = 1
    else:
        D[rng.integers(dim)] = 1
    return D, VUELOS.index(modo)


def _dentro(x, rng, lb, ub):
    """Las coordenadas que se salen de la zona se cambian por una al azar
    (igual que space_bound de AHA_python.py)."""
    fuera = (x > ub) | (x < lb)
    azar = rng.random(len(x)) * (ub - lb) + lb
    return np.where(fuera, azar, x)


def periodo_migracion(n_colibries, coef_migracion):
    """Cada cuantas vueltas hay migracion (0 = nunca)."""
    if coef_migracion <= 0:
        return 0
    return max(1, int(round(coef_migracion * n_colibries)))


def evaluaciones_esperadas(n_colibries, max_iter, coef_migracion):
    """Llamadas a la funcion: n al inicio + n por vuelta + 1 por migracion
    (la migracion ocurre en las vueltas 0, P, 2P, ...)."""
    P = periodo_migracion(n_colibries, coef_migracion)
    migraciones = len(range(0, max_iter, P)) if P else 0
    return n_colibries + n_colibries * max_iter + migraciones


def aha(f, lb, ub, dim, n_colibries=50, max_iter=1000, p_guiado=0.5,
        coef_migracion=2.0, modo_vuelo="mixto", sigma=1.0, seed=None,
        guardar_posiciones=False):
    """Minimiza f en [lb, ub]^dim (lb, ub: numero o lista por coordenada).
    Devuelve un diccionario con los resultados que pide el plan."""
    if modo_vuelo not in MODOS_VUELO:
        raise ValueError(f"modo_vuelo debe ser uno de {MODOS_VUELO}")
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(np.asarray(lb, float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, float), (dim,)).copy()
    n = int(n_colibries)
    evals = 0

    def evaluar(x):
        nonlocal evals
        evals += 1
        return float(f(x))

    # --- 1. Inicio: cada colibri en una flor al azar ------------------------
    X = rng.random((n, dim)) * (ub - lb) + lb
    fit = np.array([evaluar(x) for x in X])
    i_mejor = int(np.argmin(fit))
    mejor_x, mejor_f = X[i_mejor].copy(), fit[i_mejor]
    curva = [mejor_f]

    # Tabla de visitas: tabla[i, j] = cuantas vueltas lleva el colibri i sin
    # visitar la flor j. La diagonal no cuenta (nan / -inf).
    tabla = np.zeros((n, n))
    diag = np.diag_indices(n)
    tabla[diag] = np.nan
    P = periodo_migracion(n, coef_migracion)
    mejoras = {"guiado": 0, "territorial": 0, "migracion": 0}
    n_migraciones = 0

    # Solo para dibujar (no cambian el algoritmo)
    posiciones = [X.copy()] if guardar_posiciones else None
    mejores = [mejor_x.copy()] if guardar_posiciones else None
    acciones, vuelos, mejoro = [], [], []
    accion = np.zeros(n, dtype=int)
    vuelo = np.zeros(n, dtype=int)
    exito = np.zeros(n, dtype=bool)

    # --- 2. Ciclo principal -------------------------------------------------
    for it in range(max_iter):
        tabla[diag] = -np.inf
        for i in range(n):
            D, vuelo[i] = _vuelo(rng, dim, modo_vuelo)
            if rng.random() < p_guiado:
                # Forrajeo guiado: ir a la flor que mas tiempo lleva sin
                # visitar; si empatan, la de mejor valor.
                fila = tabla[i]
                candidatas = np.flatnonzero(fila == fila.max())
                obj = candidatas[np.argmin(fit[candidatas])]
                a = sigma * rng.standard_normal()
                nueva = X[obj] + a * D * (X[i] - X[obj])
                nueva = _dentro(nueva, rng, lb, ub)
                f_nueva = evaluar(nueva)
                tabla[i] += 1
                tabla[i, obj] = 0
                exito[i] = f_nueva < fit[i]
                if exito[i]:
                    fit[i], X[i] = f_nueva, nueva
                    tabla[:, i] = np.max(tabla, axis=1) + 1
                    tabla[i, i] = -np.inf
                    mejoras["guiado"] += 1
                accion[i] = 0
            else:
                # Forrajeo territorial: buscar cerca de su propia flor
                b = sigma * rng.standard_normal()
                nueva = X[i] + b * D * X[i]
                nueva = _dentro(nueva, rng, lb, ub)
                f_nueva = evaluar(nueva)
                tabla[i] += 1
                exito[i] = f_nueva < fit[i]
                if exito[i]:
                    fit[i], X[i] = f_nueva, nueva
                    tabla[:, i] = np.max(tabla, axis=1) + 1
                    tabla[i, i] = -np.inf
                    mejoras["territorial"] += 1
                accion[i] = 1
        tabla[diag] = np.nan

        # Migracion: el de la peor flor se muda a un lugar al azar
        if P and it % P == 0:
            tabla[diag] = -np.inf
            j = int(np.argmax(fit))
            X[j] = rng.random(dim) * (ub - lb) + lb
            tabla[j] += 1
            tabla[:, j] = np.max(tabla, axis=1) + 1
            tabla[j, j] = -np.inf
            f_nueva = evaluar(X[j])
            if f_nueva < fit[j]:
                mejoras["migracion"] += 1
            fit[j] = f_nueva
            tabla[diag] = np.nan
            n_migraciones += 1
            accion[j] = 2

        i_mejor = int(np.argmin(fit))
        if fit[i_mejor] < mejor_f:
            mejor_f, mejor_x = fit[i_mejor], X[i_mejor].copy()
        curva.append(mejor_f)
        if guardar_posiciones:
            posiciones.append(X.copy())
            mejores.append(mejor_x.copy())
            acciones.append(accion.copy())
            vuelos.append(vuelo.copy())
            mejoro.append(exito.copy())

    curva = np.array(curva)
    historial = curva[1:]                       # his_best_fit (vueltas 1..T)
    # Primera vuelta donde ya se tiene el valor final (o error < 1e-12)
    llego = (historial <= mejor_f) | (historial < 1e-12)
    iter_conv = int(np.argmax(llego)) + 1 if len(historial) else 0
    return dict(mejor_x=mejor_x, mejor_f=float(mejor_f), curva=curva,
                historial=historial, evaluaciones=evals,
                iter_convergencia=iter_conv, mejoras=mejoras,
                migraciones=n_migraciones, posiciones=posiciones,
                mejores=mejores,
                acciones=np.array(acciones) if guardar_posiciones else None,
                vuelos=np.array(vuelos) if guardar_posiciones else None,
                mejoro=np.array(mejoro) if guardar_posiciones else None)


def log10_error(v):
    """Log10 del error final; -16 cuando es exactamente 0 (regla del plan)."""
    return -16.0 if v == 0 else float(np.log10(v))
