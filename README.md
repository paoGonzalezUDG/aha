# Proyecto final: el algoritmo del colibrí (AHA)

Programamos en Python el **Artificial Hummingbird Algorithm** (Zhao, Wang y
Mirjalili, 2022) para encontrar el punto más bajo de 5 funciones de prueba:
Rastrigin, Ackley, Sphere, Rosenbrock y Beale. Luego corrimos el **plan
experimental de 25 ejecuciones por función** (125 corridas) cambiando sus
parámetros para ver qué efecto tiene cada uno.

**Lo que evalúa el profe es el reporte escrito** (sobre todo la discusión de
los resultados), no el código. El plan que seguimos está en la raíz:

> **[Plan Experimental y Matriz de Resultados (PDF)](Plan%20Experimental%20y%20Matriz%20de%20Resultados%20%E2%80%94%20Artificial%20Hummingbird%20Algorithm%20%28AHA%29%20V2.pdf)**:
> las funciones con sus rangos, las 25 configuraciones y qué valores hay que
> obtener de cada ejecución.

---

## Contenido

1. [Empezar](#1-empezar)
2. [Qué hay en el proyecto](#2-qué-hay-en-el-proyecto)
3. [Cómo usar la app](#3-cómo-usar-la-app)
4. [La carpeta resultados](#4-la-carpeta-resultados)
5. [Ver el algoritmo en el código](#5-ver-el-algoritmo-en-el-código)
6. [Qué cambiamos de AHA_python.py](#6-qué-cambiamos-de-aha_pythonpy)
7. [Regenerar los resultados](#7-regenerar-los-resultados)
8. [Comprobar que todo está bien (pruebas)](#8-comprobar-que-todo-está-bien-pruebas)
9. [Si algo falla](#9-si-algo-falla)

---

## 1. Empezar

Necesitas Python 3.10 o más nuevo.

**La primera vez** (descargar e instalar):

```
git clone https://github.com/paoGonzalezUDG/aha.git
cd aha
python -m pip install -r requirements.txt
```

**Para traer los cambios más recientes:** `git pull` y luego
`python -m pip install -r requirements.txt` (por si se agregó alguna
librería).

**Para abrir la app:**

```
streamlit run app.py
```

Se abre sola en el navegador (si no, entra a <http://localhost:8501>). Para
cerrarla, `Ctrl + C` en la terminal.

---

## 2. Qué hay en el proyecto

```
aha/
├── app.py               ▶ Abre la app:  streamlit run app.py
├── experimentos.py      ▶ Regenera los resultados:  python experimentos.py
│
├── algoritmo/           Lo que hay que entender
│   ├── aha.py               El algoritmo del colibrí.
│   └── funciones.py         Las 5 funciones de prueba con los rangos del plan.
├── interfaz/            Apoyo de la app (no hace falta estudiarlo)
│   └── visual3d.py          Simulación 3D interactiva.
│
├── resultados/          Tablas y gráficas del plan de 25 ejecuciones.
├── pruebas/             Revisan que todo funcione y los números sean correctos.
│
├── AHA_python.py        El código original que nos dieron (referencia).
├── Plan Experimental ... .pdf   El plan experimental.
├── README.md            Este archivo.
├── requirements.txt     Librerías que hay que instalar.
└── CLAUDE.md            Notas para el asistente de IA (se puede ignorar).
```

`README.md`, `requirements.txt`, `CLAUDE.md` y `.gitignore` se quedan en la
raíz a propósito: GitHub, `pip` y las herramientas los buscan ahí.

| Archivo | Qué hace | ¿Hay que entenderlo? |
|---|---|---|
| `algoritmo/aha.py` | Pone a cada colibrí en una flor al azar y en cada vuelta lo mueve con forrajeo guiado o territorial; cada cierto tiempo, el de la peor flor migra. Lleva la tabla de visitas y regresa la mejor flor. | **Sí, bien** |
| `algoritmo/funciones.py` | Las fórmulas de las 5 funciones, su zona de búsqueda y dónde está su mínimo. | Sí |
| `experimentos.py` | Corre las 25 ejecuciones del plan en las 5 funciones (1 corrida cada una, con su semilla) y genera la carpeta `resultados/`. | La idea general |
| `app.py` y `interfaz/visual3d.py` | La interfaz y la simulación 3D. | No |
| `AHA_python.py` | El código original. Ya no se usa (lo reemplaza `algoritmo/aha.py`), pero sirve para comparar. | Para comparar |

---

## 3. Cómo usar la app

Toda la app está escrita en lenguaje sencillo. Si es la primera vez, empieza
por **Cómo funciona**: explica las palabras básicas (colibrí, flor, tabla
de visitas, vuelta, semilla...) con ejemplos.

Colores en la simulación: **azul** = forrajeo guiado, **verde** = forrajeo
territorial, **naranja** = migración, gris = posición inicial, X amarilla =
la mejor flor, rombo rojo = el mínimo real.

La app tiene **3 secciones** arriba:

| Sección | Qué ves | Úsala para... |
|---|---|---|
| **Cómo funciona** | Glosario, las 3 formas de volar (axial, diagonal, omnidireccional), las 3 estrategias, una vuelta paso a paso, los ajustes del plan (`npop`, `max_it`, `p_guided`, `mig_coef`, `flight_mode`, `sigma`, `seed`), las fórmulas, las 5 funciones y qué cambiamos del código original. | **Entender el algoritmo** y la parte teórica del reporte. |
| **Simulación 3D** | Los colibríes moviéndose sobre el terreno en 3D, con ▶ Reproducir, cámaras, camino de la mejor flor y terreno transparente. Cada cuadro explica qué pasó (cuántos usaron cada estrategia, cuántos encontraron una flor mejor, si hubo migración). Se puede girar con el mouse mientras se reproduce. | Entender **cómo es cada función** y cómo la recorren los colibríes. |
| **Resultados (25 ejecuciones)** | Las funciones y rangos, la matriz del plan (en azul lo que cambia), el mejor valor de cada ejecución (normal o en log10), el detalle por función con todo lo que pide el plan, las curvas de convergencia, las mejoras por estrategia, preguntas para la discusión y la descarga en .zip. | La sección de resultados y la discusión del reporte. |

### Simulación 3D

1. Arriba eliges la **función** y los **ajustes del algoritmo** (cada uno
   tiene un ⓘ que explica qué hace). **Volver a los valores normales**
   regresa todo a la base del plan.
2. Las vueltas empiezan en **200** (no 1000 como el plan) porque casi todo
   el movimiento pasa al principio. La línea "⚙️ Ajustes" arriba de la
   gráfica dice con qué valores salió, para que cada captura lo diga.
3. Dale **▶ Reproducir**. Pasa el mouse sobre un colibrí para ver qué
   estrategia y qué vuelo usó en esa vuelta y si encontró una flor mejor.

Ojo: la simulación es de **una sola corrida** y en 2 variables. Los números
para el reporte salen de la sección **Resultados** (o de la carpeta
`resultados/`).

---

## 4. La carpeta resultados

Aquí están las tablas (`.csv`, se abren en Excel) y las gráficas (`.png`)
del plan: 25 ejecuciones × 5 funciones, **1 corrida por ejecución** con la
semilla del plan.

```
resultados/tabla_25/
├── corridas.csv                 Una fila por ejecución (125) con todo lo que pide el plan
├── historial_<Función>.csv      El mejor valor en cada vuelta (his_best_fit), columnas ej_1 ... ej_25
├── resultados.png               Mapa de calor: mejor valor de cada ejecución en cada función
├── convergencia_<n>_<Función>.png   Curvas de cada ejecución, un panel por bloque del plan
└── mejoras.png                  Cuántas mejoras logró cada estrategia en cada ejecución
```

Columnas de `corridas.csv` (sección 3 del plan):

| Columna | Qué es |
|---|---|
| `best_f` | Mejor valor (error final) al terminar. |
| `log10_best_f` | El mismo en log10; **−16** cuando `best_f` es exactamente 0. |
| `best_x` | Coordenadas de la mejor flor `[x1, ..., xd]`. |
| `evaluaciones` | Llamadas a la función. Coinciden con la columna *Evaluaciones* del plan (50,060 en la base). |
| `iter_convergencia` | Primera vuelta donde ya tenía `best_f` (o un error menor a 1e-12). |
| `tiempo_s` | Segundos que tardó la corrida (cambia un poco de una computadora a otra). |
| `mejoras_guiado`, `mejoras_territorial`, `mejoras_migracion` | Cuántas veces cada estrategia encontró una flor mejor. |
| `migraciones` | Cuántas migraciones hubo. |
| `distancia_optimo` | Qué tan lejos quedó la mejor flor del mínimo real. |

También se pueden ver en la app, sección **Resultados**.

---

## 5. Ver el algoritmo en el código

- `algoritmo/aha.py` es el algoritmo. Las listas `posiciones`, `mejores`,
  `acciones`, `vuelos` y `mejoro` solo sirven para la simulación; no cambian
  el resultado.
- `_vuelo()` elige qué coordenadas se mueven; el ciclo principal tiene el
  forrajeo guiado, el territorial y la migración, cada uno con un
  comentario.

Para ver qué hizo cada colibrí desde la terminal, sin abrir la app (desde la
carpeta del proyecto):

```
python -c "from algoritmo.aha import aha; from algoritmo.funciones import ackley; r = aha(ackley, -32.768, 32.768, 2, max_iter=5, seed=0, guardar_posiciones=True); print('accion:', r['acciones'][0][:5], 'vuelo:', r['vuelos'][0][:5], 'mejoro:', r['mejoro'][0][:5])"
```

`accion`: 0 = guiado, 1 = territorial, 2 = migró. `vuelo`: 0 = axial,
1 = diagonal, 2 = omnidireccional. `mejoro`: si encontró una flor mejor en
esa vuelta.

---

## 6. Qué cambiamos de AHA_python.py

Ningún cambio altera la idea del algoritmo:

- **Semilla:** el azar sale de `np.random.default_rng(seed)` para poder
  repetir cada ejecución (ya no hace falta instalar `torch`).
- **Vuelo axial:** el código original elegía la coordenada con
  `ceil(rand*(d-1))`, que nunca toca la primera (en 2 variables, `x` nunca
  se movía en vuelo axial). Ahora elige cualquiera, como el código de MATLAB
  de los autores.
- **Vuelo diagonal:** mueve entre 2 y d−1 coordenadas, como en MATLAB (el
  de Python movía entre 1 y d−2).
- **Parámetros del plan:** `p_guided` (antes fijo en 0.5), `mig_coef` (antes
  fijo en 2), `flight_mode` (antes siempre mixto) y `sigma` (antes 1).
- Se cuentan las **evaluaciones** y las **mejoras de cada estrategia**.

---

## 7. Regenerar los resultados

```
python experimentos.py            # las 125 corridas (~3 min)
python experimentos.py --figuras  # solo redibujar las PNG desde los CSV
```

Esto sobrescribe la carpeta `resultados/`. Como todo usa semillas fijas, con
el mismo código salen exactamente los mismos números (salvo `tiempo_s`).

---

## 8. Comprobar que todo está bien (pruebas)

La carpeta `pruebas/` revisa automáticamente que todo funcione y que los
números sean correctos. Conviene correrla antes de entregar o si se cambia
algo del código.

```
python -m pytest pruebas          # ~10 segundos
```

Si todo sale bien, al final dice `passed` y ningún `failed`.

| Qué revisa | Cómo |
|---|---|
| Evaluaciones del plan | La fórmula da las 25 cifras de la columna *Evaluaciones* del PDF, y el algoritmo cuenta exactamente eso. |
| Reproducible | La misma semilla da el mismo resultado; otra semilla, otro. |
| Vuelos y zona | Con los 4 modos de vuelo nadie se sale de la zona y la mejor flor nunca empeora. |
| Bug del vuelo axial | En 2 variables, el vuelo axial mueve las dos coordenadas. |
| Resultados guardados | Re-corre algunas ejecuciones y las compara con `corridas.csv`. |
| La app | Abre la app completa y revisa que no haya errores. |

---

## 9. Si algo falla

| Problema | Solución |
|---|---|
| `streamlit` no se reconoce como comando | `python -m pip install -r requirements.txt`, o usa `python -m streamlit run app.py` |
| `ModuleNotFoundError` (por ejemplo `No module named 'plotly'`) | Asegúrate de estar dentro de la carpeta del proyecto y corre `python -m pip install -r requirements.txt`. Si tienes varios Python instalados, `python -m pip` instala en el mismo que usa la app. |
| La simulación tarda | La primera vez tarda unos segundos; después queda guardada. Con muchos colibríes × vueltas (más de 200,000) la app avisa y no la dibuja. |
| La pestaña Resultados dice que faltan resultados | Corre `python experimentos.py` (~3 min). |
| El puerto 8501 está ocupado | Cierra la otra ventana de streamlit o usa `streamlit run app.py --server.port 8502` |

---

**Referencia principal:** Zhao, W., Wang, L., y Mirjalili, S. (2022).
Artificial hummingbird algorithm: A new bio-inspired optimizer with its
engineering applications. *Computer Methods in Applied Mechanics and
Engineering, 388*, 114194. https://doi.org/10.1016/j.cma.2021.114194
