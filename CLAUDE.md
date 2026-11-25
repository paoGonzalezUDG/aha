# CLAUDE.md

Proyecto escolar: AHA (Artificial Hummingbird Algorithm) en Python sobre 5
funciones (Rastrigin, Ackley, Sphere, Rosenbrock, Beale). Equipo: Paola y
Diana. Replica la idea de `c:\laragon\www\woa_proyecto` (WOA), pero solo con
lo que pidio Diana: explicacion del algoritmo, simulacion 3D y resultados
del plan de 25 ejecuciones (PDF en la raiz). Se evalua el REPORTE escrito:
ayudar a entender, no redactarlo.

Todo en espanol. Comentarios y nombres sin acentos; textos que ve el usuario
CON acentos, tono casual ("vuelta" no "iteracion", "flor" = solucion,
"afinar" = explotar). Nombres del plan (npop, max_it, p_guided, mig_coef,
flight_mode, sigma, seed) visibles junto al nombre sencillo.

## Archivos
- `algoritmo/aha.py`: `aha(f, lb, ub, dim, n_colibries, max_iter, p_guiado,
  coef_migracion, modo_vuelo, sigma, seed, guardar_posiciones)` -> dict
  (mejor_x, mejor_f, curva [incluye vuelta 0], historial = curva[1:],
  evaluaciones, iter_convergencia, mejoras{guiado, territorial, migracion},
  migraciones, posiciones, mejores, acciones [0 guiado, 1 territorial,
  2 migro], vuelos, mejoro). `evaluaciones_esperadas()` = formula del plan.
  Basado en `AHA_python.py`; diferencias documentadas en su docstring
  (rng con semilla, bug axial/diagonal corregido como en MATLAB).
- `algoritmo/funciones.py`: `FUNCIONES` con rangos del plan (Ackley
  +-32.768, Rosenbrock +-30).
- `experimentos.py`: `PARAMS_DEFAULT`, `TABLA_25` (25 filas del PDF),
  `filas_funcion()`, `NOMBRE_PLAN`; `python experimentos.py` (~3 min) genera
  `resultados/tabla_25/` (corridas.csv, historial_<F>.csv, PNG);
  `--figuras` solo redibuja.
- `interfaz/visual3d.py`: copia adaptada de woa_proyecto (`figura_3d`,
  `html_3d`; se muestra con `st.iframe`). Colores: azul guiado, verde
  territorial, naranja migracion, gris inicio.
- `app.py`: 3 pestanas (Como funciona | Simulacion 3D | Resultados).
  Ajustes de la simulacion dentro de su pestana (keys `sim_*`).

## Reglas
- Si se toca `aha.py`, regenerar resultados y correr pruebas: los CSV
  dependen del orden de llamadas a `rng`.
- Pruebas: `python -m pytest pruebas` (~10 s, incluye cargar la app).
- Entorno: Windows, `python` = 3.10 de Laragon (usar `python -m pip`),
  streamlit 1.65, plotly 6.
- Repo propio en esta carpeta (remote github.com/paoGonzalezUDG/aha.git,
  rama main). La carpeta padre `c:\laragon\www` es OTRO repo (mexware): no
  hacer commits ahi.
