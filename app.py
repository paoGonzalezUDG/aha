"""
Interfaz web local para el proyecto AHA (Streamlit).
Ejecutar con:   streamlit run app.py

3 secciones: como funciona el algoritmo, la simulacion 3D y los resultados
del plan experimental de 25 ejecuciones. Textos en lenguaje sencillo a
proposito: la app sirve para entender el algoritmo y escribir el reporte.
"""
import io
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algoritmo.aha import (MODOS_VUELO, aha, evaluaciones_esperadas,
                           periodo_migracion)
from algoritmo.funciones import FUNCIONES
from experimentos import (CARPETA, NOMBRE_PLAN, PARAMS_DEFAULT, TABLA_25,
                          filas_funcion)
from interfaz import visual3d

st.set_page_config(page_title="AHA - Proyecto final", layout="wide")
st.markdown("""<style>
div[data-testid="stTabs"] [data-testid="stTab"] p {
    font-size: 1.12rem; font-weight: 600; }
.aha-cargando {
    position: relative; overflow: hidden;
    display: flex; align-items: center; gap: 1rem;
    padding: 0.9rem 1.2rem 1.1rem; border-radius: 0.6rem;
    margin-bottom: 0.6rem; font-size: 1.05rem;
    background: #fff4e5; color: #4a2c00; border: 2px solid #ff9f1c; }
.aha-rueda {
    width: 2rem; height: 2rem; flex: none; border-radius: 50%;
    border: 4px solid #ffd8a8; border-top-color: #e8590c;
    border-right-color: #e8590c;
    animation: aha-girar 0.7s linear infinite; }
@keyframes aha-girar { to { transform: rotate(360deg); } }
</style>""", unsafe_allow_html=True)

RAIZ = os.path.dirname(os.path.abspath(__file__))
CARPETA_RES = os.path.join(RAIZ, CARPETA)
CSV_CORRIDAS = os.path.join(CARPETA_RES, "corridas.csv")
VUELO_BONITO = {"mixto": "Mixto", "axial": "Axial", "diagonal": "Diagonal",
                "omnidireccional": "Omnidireccional"}


def num_corto(x):
    """0 se ve como 0 (no 0.00e+00); lo demas en notacion cientifica."""
    return "0" if x == 0 else f"{x:.2e}"


st.title("El algoritmo del colibrí (AHA)")
st.caption("Colibríes virtuales buscando el punto más bajo de 5 funciones "
           "matemáticas. Proyecto final.")

cargando = st.empty()
cargando.markdown(
    '<div class="aha-cargando"><div class="aha-rueda"></div>'
    "<div><b>⏳ Cargando…</b> preparando la simulación y las tablas. Cuando "
    "desaparezca este aviso, todo está listo.</div></div>",
    unsafe_allow_html=True)

tab_inicio, tab_3d, tab_res = st.tabs(
    ["Cómo funciona", "Simulación 3D", "Resultados (25 ejecuciones)"])

# ------------------------------ Como funciona -------------------------------
with tab_inicio:
    st.markdown("""
Queremos que la compu encuentre **el punto más bajo** de una función. Imagina
un terreno lleno de cerros y hoyos: hay que encontrar el hoyo más profundo
**con los ojos vendados**. En lugar de una persona, mandamos una **bandada
de colibríes** que buscan al mismo tiempo.

Cada colibrí vive en una **flor** (un punto del terreno). Entre **más bajo**
el valor de la función en esa flor, **más néctar** tiene: es mejor flor.
Los colibríes tienen buena memoria: cada uno recuerda **hace cuánto no visita
cada flor**, y eso le ayuda a decidir a dónde ir.
""")
    st.markdown("#### ¿Por dónde empiezo?")
    st.markdown("""
| Sección | Para qué |
|---|---|
| **Cómo funciona** (aquí) | Las palabras, los movimientos y los ajustes del algoritmo |
| **Simulación 3D** | Ver a los colibríes moverse sobre el terreno, vuelta por vuelta |
| **Resultados (25 ejecuciones)** | Lo que salió del plan experimental: 25 ejecuciones × 5 funciones |
""")

    st.markdown("#### Palabras que vas a ver")
    with st.expander("Lo básico: colibrí, flor, vuelta, corrida, semilla",
                     expanded=True):
        st.markdown("""
- **Función (el terreno).** Una fórmula que le da una "altura" a cada punto.
  Ejemplo con Sphere: en el punto (3, 4) vale 3² + 4² = **25**; en (0, 0)
  vale **0**, que es lo más bajo posible.
- **Flor.** Un punto de prueba, por ejemplo (3, 4). Cada flor es una posible
  respuesta. En el artículo se llama *fuente de alimento*.
- **Colibrí.** Cada colibrí cuida **una** flor. La bandada completa es la
  población (`npop` en el plan).
- **Mejor flor.** La flor con el valor más bajo encontrada hasta ahora (la
  respuesta del algoritmo).
- **Tabla de visitas.** La memoria de la bandada: cuántas vueltas lleva cada
  colibrí **sin visitar** la flor de cada uno de los otros.
- **Vuelta (iteración).** Una ronda en la que **todos** los colibríes
  intentan moverse una vez. Con 50 colibríes y 1000 vueltas hay 50,000
  intentos.
- **Evaluación.** Cada vez que se calcula el valor de la función en un punto.
  Es la medida de "cuánto trabajo" hizo el algoritmo.
- **Corrida (ejecución).** Correr el algoritmo completo una vez, de la
  primera vuelta a la última.
- **Semilla.** El algoritmo usa azar, así que cada corrida sale distinta. La
  semilla "fija" ese azar, como barajar las cartas siempre igual: misma
  semilla, mismo resultado.
- **Variables.** Cuántos números tiene cada punto. Con 2 es (x, y) y se puede
  dibujar como mapa. Con 10 no se puede dibujar, pero funciona igual.
""")

    with st.expander("Las 3 formas de volar (qué coordenadas se mueven)"):
        st.markdown("""
Antes de moverse, cada colibrí "elige" cómo va a volar. Eso decide **qué
coordenadas** del punto puede cambiar en ese movimiento:

| Vuelo | Qué mueve | En 2 variables (x, y) |
|---|---|---|
| **Axial** | Solo **una** coordenada, elegida al azar | Solo horizontal **o** solo vertical |
| **Diagonal** | **Varias** coordenadas al azar (entre 2 y d−1) | Las dos (igual que omnidireccional) |
| **Omnidireccional** | **Todas** las coordenadas | Las dos |

En el modo **mixto** (el normal) cada vuelo sale con probabilidad 1/3. En el
plan, las ejecuciones 21, 22 y 23 obligan a usar solo uno de los tres.
""")

    with st.expander("Las 3 maneras de buscar comida (los colores de la "
                     "simulación)", expanded=True):
        st.markdown("""
En cada vuelta, cada colibrí hace **una** de las dos primeras. La tercera
pasa de vez en cuando:

| Color | Estrategia | Qué hace |
|---|---|---|
| :blue[**azul**] | **Forrajeo guiado** | Busca en su memoria la flor que **lleva más tiempo sin visitar** (si empatan, la de mejor valor) y vuela hacia ella, pudiendo pasarse o quedarse antes. *Explora.* |
| :green[**verde**] | **Forrajeo territorial** | Se queda en su zona y prueba un punto **cerca de su propia flor**. *Afina.* |
| :orange[**naranja**] | **Migración** | Cada cierto número de vueltas, el colibrí con la **peor flor** se muda a un lugar al azar del terreno. Sirve para no quedarse atorados. |

**Regla de oro:** un colibrí **solo cambia de flor si la nueva es mejor**. Si
no, se queda donde estaba (en la simulación vas a ver que muchos no se
mueven en cada vuelta).

**Explorar vs. afinar** (en el reporte: *exploración vs. explotación*).
Llegas a una ciudad nueva y buscas dónde comer. Primero **exploras**:
caminas por todos lados. Ya que encontraste una buena zona, **afinas**: te
quedas por ahí probando los lugares cercanos. En el AHA el volado entre
guiado (explorar) y territorial (afinar) lo controla `p_guided`.
""")

    with st.expander("Una vuelta, paso a paso"):
        st.markdown("""
Para **cada** colibrí, en orden:

1. **Elige cómo volar** (axial, diagonal u omnidireccional): eso dice qué
   coordenadas se pueden mover.
2. **Echa un volado** con probabilidad `p_guided` (0.5 = mitad y mitad):
   - sale guiado → mira su tabla de visitas, elige la flor que más tiempo
     lleva sin visitar y prueba un punto en esa dirección;
   - sale territorial → prueba un punto cerca de su propia flor.
3. Si el punto nuevo se salió de la zona de búsqueda, esa coordenada se
   cambia por una al azar dentro de la zona.
4. **Compara:** si el punto nuevo tiene menor valor, se muda ahí. Si no, se
   queda.
5. **Actualiza la memoria:** todas sus cuentas suben 1 (una vuelta más sin
   visitarlas) y la flor que visitó vuelve a 0. Si encontró una flor mejor,
   para **todos los demás** esa flor nueva pasa a ser "la que más tiempo
   llevan sin visitar", así que la van a querer visitar pronto.

Al terminar con todos: si toca **migración** (cada `mig_coef × npop`
vueltas; con los valores normales, cada 2 × 50 = 100), el colibrí con la
peor flor se muda al azar. Y se guarda la mejor flor de la vuelta.
""")

    with st.expander("Los ajustes del plan experimental"):
        st.markdown(f"""
| En el plan | En la app | Qué hace | Normal |
|---|---|---|---|
| `npop` | Colibríes | Cuántos colibríes buscan al mismo tiempo | {PARAMS_DEFAULT['n_colibries']} |
| `max_it` | Vueltas | Cuántas vueltas dura la corrida | {PARAMS_DEFAULT['max_iter']} |
| `p_guided` | Prob. de forrajeo guiado | En cada movimiento, qué tan seguido sale guiado en vez de territorial. 0.2 = casi siempre territorial; 0.8 = casi siempre guiado | {PARAMS_DEFAULT['p_guiado']} |
| `mig_coef` | Coeficiente de migración | La migración pasa cada `mig_coef × npop` vueltas. 0.5 → cada 25; 5 → cada 250; 0 → nunca | {PARAMS_DEFAULT['coef_migracion']} |
| `flight_mode` | Modo de vuelo | Mixto (los 3 al azar) o solo axial, diagonal u omnidireccional | mixto |
| `sigma` | Tamaño del paso | Multiplica el número al azar de las fórmulas: 0.5 = pasos de la mitad de largo | {PARAMS_DEFAULT['sigma']} |
| `seed` | Semilla | Fija el azar para poder repetir la corrida | 0 |

**Evaluaciones** (columna del plan) = `npop` al inicio + `npop` por cada
vuelta + 1 por cada migración. Ejemplo con la base: 50 + 50 × 1000 + 10 =
**50,060**.
""")

    with st.expander("Las fórmulas (para el reporte)"):
        st.markdown(r"""
| Qué | Fórmula |
|---|---|
| Forrajeo guiado | `v = x_obj + a · D · (x_i − x_obj)`, con `a ~ N(0, 1) · sigma` |
| Forrajeo territorial | `v = x_i + b · D · x_i`, con `b ~ N(0, 1) · sigma` |
| Migración | `x_peor = lb + r · (ub − lb)`, con `r` al azar entre 0 y 1 |
| Quedarse o mudarse | si `f(v) < f(x_i)` entonces `x_i = v` |

`x_i` = la flor del colibrí, `x_obj` = la flor que más tiempo lleva sin
visitar, `D` = el vector de vuelo (1 en las coordenadas que se mueven, 0 en
las demás), `N(0, 1)` = un número al azar con distribución normal.

*Zhao, Wang y Mirjalili (2022), Artificial hummingbird algorithm: A new
bio-inspired optimizer with its engineering applications. Computer Methods
in Applied Mechanics and Engineering, 388, 114194.*
""")

    with st.expander("Las 5 funciones (los terrenos)"):
        st.markdown(
            "| Función | Apodo | Variables | Zona de búsqueda | Cómo es | "
            "Mínimo en |\n|---|---|---|---|---|---|\n" + "\n".join(
                f"| {n} | {c['apodo']} | "
                f"{'2 (fija)' if c['dim'] else '10 (30 en la ej. 25)'} | "
                f"[{c['lb']:g}, {c['ub']:g}] | {c['paisaje']} | "
                f"{'(1, ..., 1)' if n == 'Rosenbrock' else '(3, 0.5)' if n == 'Beale' else '(0, ..., 0)'} |"
                for n, c in FUNCIONES.items())
            + "\n\nEn todas, el valor más bajo posible es **0**.")

    with st.expander("Cómo leer los números"):
        st.markdown("""
- Los números son el **error**: qué tan lejos quedó del mínimo real, que en
  las 5 funciones es **0**. Más cerca de 0 = mejor.
- `1.5e-15` es notación científica: 0.0000000000000015, o sea, **casi
  cero**. `3.2e+01` = 32.
- **log10(best_f)** es el exponente: `1e-15` → −15, `100` → 2. El plan pide
  poner **−16** cuando el resultado es exactamente 0. Sirve para comparar
  números muy chicos.
- **Vuelta de convergencia:** la primera vuelta en la que ya se tenía el
  resultado final (o un error menor a 1e-12). Si sale cerca del total de
  vueltas, todavía estaba mejorando al final.
- En las gráficas con **escala logarítmica**, cada raya es 10 veces más
  chica que la anterior.
""")

    with st.expander("Qué cambiamos del código AHA_python.py"):
        st.markdown("""
Usamos el código que nos dieron, con estos cambios (ninguno cambia la idea
del algoritmo):

- **Semilla:** el azar sale de un generador con semilla (antes usaba
  `torch.randperm` y `np.random` sin semilla), para que cada ejecución del
  plan se pueda repetir. Ya no hace falta instalar `torch`.
- **Vuelo axial:** el código elegía la coordenada con `ceil(rand·(d−1))`,
  que **nunca** elige la primera (en 2 variables, `x` nunca se movía en
  vuelo axial). Ahora elige cualquiera, como el código original en MATLAB
  de los autores.
- **Vuelo diagonal:** ahora mueve entre 2 y d−1 coordenadas, como en
  MATLAB (el código de Python movía entre 1 y d−2).
- **Parámetros del plan:** `p_guided` (antes fijo en 0.5), `mig_coef` (antes
  fijo en 2), `flight_mode` (antes siempre mixto) y `sigma` (antes 1).
- Se cuentan las **evaluaciones** y las **mejoras de cada estrategia**,
  que pide el plan.

Todo está en `algoritmo/aha.py`.
""")

# ------------------------------ Simulacion 3D -------------------------------
QUE_VER_3D = {
    "Rastrigin": "Es un **cartón de huevo**: lleno de hoyitos. Fíjate si "
                 "algunos colibríes se quedan atorados en hoyitos mientras "
                 "otros encuentran el hoyo del centro.",
    "Ackley": "Es una **llanura con un pozo** en el centro, y la zona es "
              "enorme (±32.768). En la parte plana casi no hay pistas de "
              "hacia dónde ir.",
    "Sphere": "Es un **tazón** sin trampas. Acércate con la rueda del mouse: "
              "al final quedan todos en un puntito.",
    "Rosenbrock": "Es el **valle del plátano**: llegan rápido al valle curvo, "
                  "pero avanzar por el fondo hasta el mínimo (1, 1) es "
                  "lentísimo porque casi no baja.",
    "Beale": "Son **valles angostos** con esquinas muy empinadas. El mínimo "
             "no está en el centro sino en (3, 0.5).",
}
VELOCIDAD_3D = {"Lenta": 300, "Normal": 150, "Rápida": 60}
CAMARAS_3D = {"Desde arriba": "arriba", "Inclinada": "inclinada",
              "Girar sola": "girar"}
LIMITE_MOVIMIENTOS = 200_000
SIM_DEFAULT = dict(sim_funcion="Sphere",
                   sim_n=PARAMS_DEFAULT["n_colibries"], sim_T=200,
                   sim_p=PARAMS_DEFAULT["p_guiado"],
                   sim_mig=PARAMS_DEFAULT["coef_migracion"],
                   sim_vuelo=PARAMS_DEFAULT["modo_vuelo"],
                   sim_sigma=PARAMS_DEFAULT["sigma"], sim_seed=0)
for _k, _v in SIM_DEFAULT.items():          # valor inicial, solo una vez
    st.session_state.setdefault(_k, _v)


def valores_por_defecto():
    st.session_state.update(SIM_DEFAULT)


@st.cache_data(show_spinner=False)
def corrida(nom, seed, n_colibries, max_iter, p_guiado, coef_migracion,
            modo_vuelo, sigma):
    """aha() en 2 variables guardando posiciones, con su tiempo."""
    cfg = FUNCIONES[nom]
    t0 = time.perf_counter()
    res = aha(cfg["f"], cfg["lb"], cfg["ub"], 2, seed=seed,
              guardar_posiciones=True, n_colibries=n_colibries,
              max_iter=max_iter, p_guiado=p_guiado,
              coef_migracion=coef_migracion, modo_vuelo=modo_vuelo,
              sigma=sigma)
    return {**res, "tiempo": time.perf_counter() - t0}


def texto_ajustes(nom, p, semilla):
    """Linea con los ajustes usados (para las capturas). En negritas lo que
    no esta en su valor normal del plan."""
    partes = [f"**{nom}**", "2 variables"]
    for k in ("n_colibries", "max_iter", "p_guiado", "coef_migracion",
              "modo_vuelo", "sigma"):
        v, dfl = p[k], PARAMS_DEFAULT[k]
        txt = f"{NOMBRE_PLAN[k]} = {v:g}" if k != "modo_vuelo" else \
            f"{NOMBRE_PLAN[k]} = {v}"
        distinto = v != dfl if k == "modo_vuelo" else not np.isclose(v, dfl)
        partes.append(f"**{txt}** (plan: {dfl if k == 'modo_vuelo' else f'{dfl:g}'})"
                      if distinto else txt)
    partes.append(f"semilla {semilla}")
    return "⚙️ Ajustes: " + " · ".join(partes)


with tab_3d:
    st.write("Los colibríes moviéndose sobre el terreno, en 3D. Elige la "
             "función y los ajustes, y dale **▶ Reproducir** abajo de la "
             "gráfica.")
    with st.container(border=True):
        st.markdown("**Ajustes del algoritmo**")
        c1, c2, c3, c4 = st.columns(4)
        nombre = c1.selectbox(
            "Función", list(FUNCIONES), key="sim_funcion",
            help="El “terreno” donde los colibríes buscan el punto más bajo.")
        n_col = c2.number_input(
            "Colibríes (npop)", 2, 300, key="sim_n",
            help="Cuántos colibríes buscan al mismo tiempo. El plan usa 50.")
        T_sim = c3.number_input(
            "Vueltas (max_it)", 1, 3000, key="sim_T",
            help="Aquí empieza en 200 para que se vea el movimiento (casi "
                 "todo pasa al principio). El plan usa 1000.")
        semilla = c4.number_input(
            "Semilla (seed)", 0, 10**6, key="sim_seed",
            help="Misma semilla = mismo resultado. Cámbiala para ver otra "
                 "corrida.")
        c1, c2, c3, c4 = st.columns(4)
        p_g = c1.slider(
            "Prob. de forrajeo guiado (p_guided)", 0.0, 1.0, step=0.05,
            key="sim_p",
            help="Qué tan seguido un colibrí sale a buscar la flor que no "
                 "visita hace más (azul), en vez de buscar cerca de la suya "
                 "(verde). El plan usa 0.5.")
        mig = c2.number_input(
            "Coef. de migración (mig_coef)", 0.0, 20.0, step=0.5,
            key="sim_mig",
            help="La migración (naranja) pasa cada mig_coef × npop vueltas. "
                 "0 = nunca. El plan usa 2.")
        vuelo = c3.selectbox(
            "Modo de vuelo (flight_mode)", MODOS_VUELO, key="sim_vuelo",
            format_func=VUELO_BONITO.get,
            help="Qué coordenadas se mueven: axial = una, diagonal = varias, "
                 "omnidireccional = todas, mixto = cualquiera de las tres "
                 "al azar.")
        sig = c4.number_input(
            "Tamaño del paso (sigma)", 0.1, 5.0, step=0.1, key="sim_sigma",
            help="Multiplica el número al azar de las fórmulas: más chico = "
                 "pasos más cortos. El plan usa 1.")
        st.button("Volver a los valores normales",
                  on_click=valores_por_defecto)

    cfg = FUNCIONES[nombre]
    params = dict(n_colibries=int(n_col), max_iter=int(T_sim),
                  p_guiado=float(p_g), coef_migracion=float(mig),
                  modo_vuelo=vuelo, sigma=float(sig))
    if cfg["dim"] is None:
        st.caption(f"Para poder dibujarla, aquí {nombre} se usa con 2 "
                   "variables (en el plan va con 10 y 30).")
    st.info(f"**{nombre}:** " + QUE_VER_3D[nombre])

    with st.container(border=True):
        st.markdown("**Cómo verla**")
        c1, c2, c3 = st.columns([1.7, 1, 1])
        cam_3d = c1.radio(
            "Cámara", list(CAMARAS_3D), horizontal=True, key="cam_3d",
            help="La vista con la que empieza. Siempre puedes girarla "
                 "arrastrando con el mouse.")
        vel_3d = c2.select_slider("Velocidad", list(VELOCIDAD_3D),
                                  value="Normal")
        opac_3d = c3.slider(
            "Terreno transparente", 0, 70, 10, step=10, format="%d%%",
            help="Súbelo para ver los colibríes que quedan detrás de una "
                 "montaña.")
        c1, c2, _ = st.columns([1.7, 1, 1])
        log_3d = c1.toggle(
            "Aplanar los picos (escala log)", value=True,
            help="Algunas funciones tienen picos altísimos que aplastan todo "
                 "lo demás. Los colibríes siguen en el mismo lugar; solo "
                 "cambia la altura del dibujo.")
        rastro_3d = c2.toggle(
            "Camino de la mejor flor", value=False,
            help="Dibuja la línea que fue siguiendo la mejor flor.")

    movimientos = params["n_colibries"] * params["max_iter"]
    if movimientos > LIMITE_MOVIMIENTOS:
        st.warning(f"Son demasiados movimientos ({movimientos:,} = colibríes "
                   "× vueltas) para la simulación: tardaría mucho. Baja los "
                   f"colibríes o las vueltas (máximo {LIMITE_MOVIMIENTOS:,}).")
    else:
        with st.spinner("Corriendo el algoritmo..."):
            res_3d = corrida(nombre, int(semilla), **params)
        with st.spinner("Preparando la simulación 3D..."):
            fig_3d = visual3d.figura_3d(
                cfg["f"], cfg["lb"], cfg["ub"], res_3d, cfg["optimo_x"](2),
                log=log_3d, ms=VELOCIDAD_3D[vel_3d],
                camara=CAMARAS_3D[cam_3d], rastro=rastro_3d,
                opacidad=1 - opac_3d / 100,
                periodo_migracion=periodo_migracion(
                    params["n_colibries"], params["coef_migracion"]),
                tiempo=res_3d["tiempo"])
        st.markdown(texto_ajustes(nombre, params, int(semilla)))
        st.iframe(visual3d.html_3d(fig_3d), height=fig_3d.layout.height + 10)
        m = res_3d["mejoras"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejor valor encontrado", num_corto(res_3d["mejor_f"]),
                  help="Lo más bajo que encontraron. Lo ideal es 0.")
        c2.metric("Evaluaciones", f"{res_3d['evaluaciones']:,}",
                  help="Cuántas veces se calculó el valor de la función.")
        c3.metric("Vuelta de convergencia", res_3d["iter_convergencia"],
                  help="Primera vuelta en la que ya tenía el resultado "
                       "final (o un error menor a 1e-12).")
        c4.metric("Tiempo del algoritmo",
                  visual3d.texto_tiempo(res_3d["tiempo"]),
                  help="Sin contar lo que tarda en dibujar. La animación va "
                       "más lento a propósito para que se pueda ver.")
        st.markdown(
            f"**Mejoras por estrategia:** :blue[guiado {m['guiado']:,}] · "
            f":green[territorial {m['territorial']:,}] · "
            f":orange[migración {m['migracion']}] (de "
            f"{res_3d['migraciones']} migraciones) — cuántas veces cada una "
            "encontró una flor mejor.")
        st.caption("Gírala arrastrando con el mouse y acércate con la rueda, "
                   "aunque se esté reproduciendo. Pasa el mouse sobre un "
                   "colibrí para ver qué hizo en esa vuelta y su valor.")

# -------------------------------- Resultados --------------------------------
COLUMNAS_PLAN = ["npop", "max_it", "p_guided", "mig_coef", "flight_mode",
                 "sigma"]


def tabla_diseno():
    """Matriz del plan (25 filas) y, por fila, las columnas que cambian."""
    filas, cambia = [], []
    for num, bloque, cambios, seed, texto in TABLA_25:
        p = {**PARAMS_DEFAULT, **{k: v for k, v in cambios.items()
                                  if k != "dim"}}
        dims = "30D (Rast/Sph/Ros) · 2D (Ack/Beale)" if "dim" in cambios \
            else "10D (Rast/Sph/Ros) · 2D (Ack/Beale)"
        filas.append({
            "Ej.": num, "Bloque": bloque,
            **{NOMBRE_PLAN[k]: (p[k] if k == "modo_vuelo" else f"{p[k]:g}")
               for k in NOMBRE_PLAN},
            "seed": seed, "Dimensión": dims,
            "Evaluaciones": f"{evaluaciones_esperadas(p['n_colibries'], p['max_iter'], p['coef_migracion']):,}",
            "Qué se evalúa": texto})
        cols = {NOMBRE_PLAN[k] for k in cambios if k in NOMBRE_PLAN}
        if "dim" in cambios:
            cols.add("Dimensión")
        if seed != 0:
            cols.add("seed")
        cambia.append(cols)
    return pd.DataFrame(filas), cambia


def texto_cambio(ejecucion, dim, seed):
    """Lo que distingue a una fila del plan, en corto."""
    cambios = TABLA_25[ejecucion - 1][2]
    if not cambios:
        return f"base, seed {seed}"
    if "dim" in cambios:
        return f"d = {dim}"
    return ", ".join(f"{NOMBRE_PLAN[k]} = {v}" for k, v in cambios.items())


def resaltar(df, cambia):
    """Pinta de azul las celdas que cambian respecto a la base."""
    def estilo(fila):
        return ["background-color: #dbeafe; font-weight: 600"
                if c in cambia[fila.name] else "" for c in fila.index]
    return df.style.apply(estilo, axis=1)


@st.cache_data(show_spinner=False)
def leer_corridas(firma):
    return pd.read_csv(CSV_CORRIDAS)


@st.cache_data(show_spinner=False)
def leer_historial(nombre, firma):
    return pd.read_csv(os.path.join(CARPETA_RES, f"historial_{nombre}.csv"))


@st.cache_data(show_spinner=False)
def zip_resultados(firma):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for arch in sorted(os.listdir(CARPETA_RES)):
            z.write(os.path.join(CARPETA_RES, arch), f"tabla_25/{arch}")
    return buf.getvalue()


def figura_convergencia(hist, ejecuciones, nombre):
    fig = go.Figure()
    for k in ejecuciones:
        y = hist[f"ej_{k}"].dropna().to_numpy()
        y = np.maximum(y, 1e-16)
        fig.add_trace(go.Scatter(x=np.arange(1, len(y) + 1), y=y,
                                 mode="lines", name=f"Ej. {k}"))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=40, b=10),
        title=f"{nombre}: mejor valor en cada vuelta",
        xaxis_title="Vuelta", yaxis_title="Mejor valor (log)",
        yaxis_type="log", legend_title="Ejecución")
    return fig


with tab_res:
    st.write("El plan experimental: **25 ejecuciones por función** (125 "
             "corridas), **una corrida por fila** con la semilla indicada. "
             "Los números salen de `resultados/tabla_25/` (se generan con "
             "`python experimentos.py`).")

    st.subheader("1. Funciones de prueba y rangos de búsqueda")
    st.dataframe(pd.DataFrame([{
        "Función": n,
        "Dimensión (d)": "2 (fija en todas)" if c["dim"]
        else "10 (y 30 en Ej. 25)",
        "Rango [lb, ub]": f"[{c['lb']:g}, {c['ub']:g}]",
        "Óptimo f(x*)": "0.0", "Cómo es": c["paisaje"]}
        for n, c in FUNCIONES.items()]), hide_index=True)

    st.subheader("2. Matriz de diseño experimental")
    st.caption("En azul, lo que cambia respecto a la base (Ej. 1). Ackley y "
               "Beale van siempre en 2 variables, así que su ejecución 25 es "
               "igual a la 1.")
    df_dis, cambia = tabla_diseno()
    st.dataframe(resaltar(df_dis, cambia), hide_index=True,
                 height=35 * (len(df_dis) + 1) + 3)

    if not os.path.exists(CSV_CORRIDAS):
        st.warning("Todavía no hay resultados. Córrelos con "
                   "`python experimentos.py` (unos 4 minutos).")
    else:
        firma = os.path.getmtime(CSV_CORRIDAS)
        df = leer_corridas(firma)

        st.subheader("3. Mejor valor de cada ejecución")
        ver_log = st.toggle(
            "Ver como log10(best_f)", value=False,
            help="El exponente del resultado: 1e-15 → −15. El plan pone −16 "
                 "cuando el resultado es exactamente 0.")
        col = "log10_best_f" if ver_log else "best_f"
        t = df.pivot(index="ejecucion", columns="funcion", values=col)
        t = t[list(FUNCIONES)].reset_index().rename(columns={"ejecucion":
                                                             "Ej."})
        t.insert(1, "Bloque", [TABLA_25[k - 1][1] for k in t["Ej."]])
        fmt = ({c: "{:.2f}" for c in FUNCIONES} if ver_log
               else {c: num_corto for c in FUNCIONES})
        st.dataframe(t.style.format(fmt), hide_index=True,
                     height=35 * (len(t) + 1) + 3)
        st.image(os.path.join(CARPETA_RES, "resultados.png"),
                 width="stretch")
        st.caption("Verde = casi 0 (muy bien), rojo = lejos del mínimo. Las "
                   "líneas negras separan los 5 bloques del plan. Ojo: con "
                   "**una sola corrida** por fila pesa mucho el azar; el "
                   "bloque 1 (5 semillas distintas) dice cuánto puede "
                   "cambiar el resultado solo por suerte.")

        st.subheader("4. Detalle por función")
        f_sel = st.radio("Función", list(FUNCIONES), horizontal=True,
                         key="res_funcion")
        d = df[df.funcion == f_sel].sort_values("ejecucion")
        detalle = pd.DataFrame({
            "Ej.": d.ejecucion, "Bloque": d.bloque,
            "Cambio": [texto_cambio(e, dd, s)
                       for e, dd, s in zip(d.ejecucion, d.dim, d.seed)],
            "d": d.dim, "best_f": d.best_f,
            "log10(best_f)": d.log10_best_f,
            "Evaluaciones": d.evaluaciones,
            "Vuelta de convergencia": d.iter_convergencia,
            "Tiempo (s)": d.tiempo_s,
            "Mejoras guiado": d.mejoras_guiado,
            "Mejoras territorial": d.mejoras_territorial,
            "Mejoras migración": d.mejoras_migracion,
            "Distancia al mínimo": d.distancia_optimo})
        st.dataframe(
            detalle.style.format({"best_f": num_corto,
                                  "log10(best_f)": "{:.2f}",
                                  "Evaluaciones": "{:,}",
                                  "Tiempo (s)": "{:.2f}",
                                  "Distancia al mínimo": num_corto}),
            hide_index=True, height=35 * (len(detalle) + 1) + 3,
            column_config={
                "Vuelta de convergencia": st.column_config.Column(
                    help="Primera vuelta donde ya tenía el resultado final "
                         "(o error < 1e-12)."),
                "Mejoras guiado": st.column_config.Column(
                    help="Cuántas veces el forrajeo guiado encontró una flor "
                         "mejor."),
                "Mejoras territorial": st.column_config.Column(
                    help="Cuántas veces el forrajeo territorial encontró una "
                         "flor mejor."),
                "Mejoras migración": st.column_config.Column(
                    help="Cuántas migraciones cayeron en un lugar mejor que "
                         "la flor que dejaron."),
                "Distancia al mínimo": st.column_config.Column(
                    help="Qué tan lejos quedó la mejor flor del verdadero "
                         "mínimo.")})

        hist = leer_historial(f_sel, firma)
        bloques = list(dict.fromkeys(b for _, b, _, _, _ in TABLA_25))
        bloque_sel = st.selectbox(
            "Curvas de convergencia: ¿qué bloque?", ["Elegir yo"] + bloques,
            index=1, key="res_bloque",
            help="Muestra las ejecuciones de un bloque del plan, o elige tú "
                 "cuáles comparar.")
        if bloque_sel == "Elegir yo":
            ejs = st.multiselect("Ejecuciones", list(range(1, 26)),
                                 default=[1, 6, 11, 16, 21],
                                 key="res_ejs")
        else:
            ejs = [n for n, b, _, _, _ in TABLA_25 if b == bloque_sel]
        if ejs:
            st.plotly_chart(figura_convergencia(hist, ejs, f_sel),
                            width="stretch")
            st.caption("Escala logarítmica. Un 0 exacto se dibuja en 1e-16 "
                       "para que quepa en la gráfica. En el bloque 3 las "
                       "curvas se enciman: todas usan la semilla 0, así que "
                       "la de 2000 vueltas hace exactamente lo mismo que la "
                       "de 100 en sus primeras 100 vueltas, y luego sigue.")
        with st.expander("best_x de cada ejecución (coordenadas de la mejor "
                         "flor)"):
            for k, bx in zip(d.ejecucion, d.best_x):
                v = np.array(json.loads(bx))
                st.text(f"Ej. {k:>2}: "
                        + np.array2string(v, precision=4, max_line_width=150))

        st.subheader("5. Mejoras por estrategia")
        st.image(os.path.join(CARPETA_RES, "mejoras.png"), width="stretch")
        st.caption("Azul = forrajeo guiado, verde = territorial, naranja = "
                   "migración. La altura total es cuántas veces algún "
                   "colibrí encontró una flor mejor en esa ejecución.")

        with st.expander("Preguntas para guiar la discusión del reporte"):
            st.markdown("""
Estas son preguntas, no respuestas: la idea es que las contesten viendo las
tablas y las gráficas.

1. **Bloque 1 (5 semillas):** ¿cuánto cambia el resultado solo por el azar?
   ¿En qué funciones casi no cambia y en cuáles sí?
2. **Bloque 2 (bandada):** ¿más colibríes siempre es mejor? Recuerden que
   también gastan más evaluaciones (columna *Evaluaciones*).
3. **Bloque 3 (vueltas):** comparen la *vuelta de convergencia* con
   `max_it`. ¿Hay funciones donde más vueltas ya no ayudan? ¿Alguna donde
   seguía mejorando al final?
4. **Bloque 4 (forrajeo y migración):** con `p_guided` = 0.2 o 0.8, ¿qué
   estrategia consiguió más mejoras? ¿Quitar la migración (Ej. 20) cambió
   algo?
5. **Bloque 5 (vuelos, paso y 30D):** ¿qué modo de vuelo funcionó mejor en
   cada función? ¿Qué pasa al subir a 30 variables (Ej. 25)?
6. **Por función:** algunas funciones llegan a **0 exacto** y otras no.
   Pista: fíjense en la fórmula del forrajeo territorial (`x_i + b·D·x_i`)
   y en **dónde** está el mínimo de cada función.
""")

        st.subheader("6. Descargas")
        st.download_button(
            "⬇️ Descargar todos los resultados (ZIP)", zip_resultados(firma),
            file_name="aha_tabla_25.zip", mime="application/zip",
            help="corridas.csv (una fila por ejecución con todo lo que pide "
                 "el plan), historial_<Función>.csv (mejor valor en cada "
                 "vuelta) y las gráficas.")
        imgs = sorted(a for a in os.listdir(CARPETA_RES)
                      if a.startswith("convergencia_"))
        with st.expander("Gráficas de convergencia para el reporte (PNG)"):
            for a in imgs:
                st.image(os.path.join(CARPETA_RES, a), width="stretch")

cargando.empty()
