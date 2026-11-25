# Proyecto final: el algoritmo del colibrí (AHA)

Programamos en Python el **Artificial Hummingbird Algorithm** (Zhao, Wang y
Mirjalili, 2021) para encontrar el punto más bajo de 5 funciones de prueba:
Rastrigin, Ackley, Sphere, Rosenbrock y Beale. Luego corrimos el **plan
experimental de 25 ejecuciones por función** (125 corridas) cambiando sus
parámetros.

## Empezar

Necesitas Python 3.10 o más nuevo.

```
python -m pip install -r requirements.txt
streamlit run app.py
```

Se abre sola en el navegador (si no, entra a <http://localhost:8501>). Para
cerrarla, `Ctrl + C` en la terminal.

## La app (3 secciones)

| Sección | Qué tiene |
|---|---|
| **Cómo funciona** | Las palabras (colibrí, flor, tabla de visitas...), las 3 formas de volar, las 3 estrategias (guiado, territorial, migración), una vuelta paso a paso, los ajustes del plan, las fórmulas y las 5 funciones |
| **Simulación 3D** | Los colibríes moviéndose sobre el terreno vuelta por vuelta, con una explicación en cada cuadro. Se pueden cambiar todos los ajustes del plan |
| **Resultados (25 ejecuciones)** | La matriz del plan, el mejor valor de cada ejecución, el detalle por función (todo lo que pide la sección 3 del plan), las curvas de convergencia, las mejoras por estrategia y la descarga en ZIP |

## Qué hay en el proyecto

| Archivo | Para qué |
|---|---|
| `app.py` | La app (Streamlit) |
| `experimentos.py` | Corre el plan de 25 ejecuciones y guarda `resultados/tabla_25/` |
| `algoritmo/aha.py` | El algoritmo AHA (basado en `AHA_python.py`) |
| `algoritmo/funciones.py` | Las 5 funciones con los rangos del plan |
| `interfaz/visual3d.py` | La simulación 3D (Plotly) |
| `pruebas/` | Pruebas automáticas |
| `AHA_python.py`, `Plan Experimental ... .pdf` | Los originales que nos dieron |

## Los resultados

`resultados/tabla_25/`:

- `corridas.csv`: una fila por ejecución (125) con `best_f`,
  `log10_best_f` (−16 si es 0 exacto), `evaluaciones`,
  `iter_convergencia`, `tiempo_s`, mejoras por estrategia, `best_x` y la
  distancia al mínimo.
- `historial_<Función>.csv`: el `his_best_fit` de cada ejecución (mejor
  valor en cada vuelta; columnas `ej_1` ... `ej_25`).
- `resultados.png`, `convergencia_<n>_<Función>.png`, `mejoras.png`:
  gráficas para el reporte.

Para volver a generarlos (unos 3 minutos):

```
python experimentos.py            # las 125 corridas
python experimentos.py --figuras  # solo redibujar las PNG
```

Las evaluaciones de cada corrida coinciden con la columna *Evaluaciones*
del plan (por ejemplo 50,060 en la base).

## Qué cambiamos de AHA_python.py

- El azar sale de `np.random.default_rng(seed)` para poder repetir cada
  ejecución (ya no hace falta `torch`).
- **Vuelo axial:** el código original nunca movía la primera coordenada
  (`ceil(rand*(d-1))`); ahora elige cualquiera, como el código de MATLAB de
  los autores. **Vuelo diagonal:** mueve entre 2 y d−1 coordenadas, como en
  MATLAB.
- Se agregaron los parámetros del plan: `p_guided`, `mig_coef`,
  `flight_mode` y `sigma`.
- Se cuentan las evaluaciones y las mejoras de cada estrategia.

## Pruebas

```
python -m pytest pruebas
```

Revisan que las evaluaciones coincidan con el plan, que la misma semilla dé
el mismo resultado, que los resultados guardados se puedan repetir y que la
app cargue sin errores.
