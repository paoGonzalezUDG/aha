"""
Simulacion 3D interactiva (Plotly) de los colibries sobre la funcion.

La funcion de 2 variables se dibuja como un terreno (superficie) y los
colibries como puntos encima. Se puede girar, hacer zoom y reproducir la
animacion vuelta por vuelta.

Colores fijos: azul = forrajeo guiado, verde = forrajeo territorial,
naranja = migracion, gris = posicion inicial.
"""
import json

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

COLORES_ACCION = np.array(["#1f77b4", "#2ca02c", "#ff7f0e"])
COLOR_INICIAL = "#bbbbbb"
ACCION_CORTA = ["guiado", "territorial", "migró"]
VUELO_CORTO = ["axial", "diagonal", "omnidireccional"]
LEYENDA_ACCIONES = ["Guiado: vuela a la flor que no visita hace más",
                    "Territorial: busca cerca de su flor",
                    "Migración: se muda a un lugar al azar"]

CAMARA_ARRIBA = dict(eye=dict(x=0, y=-0.01, z=2), up=dict(x=0, y=1, z=0),
                     projection=dict(type="orthographic"))
CAMARA_INCLINADA = dict(eye=dict(x=1.2, y=-1.2, z=0.85), up=dict(x=0, y=0, z=1),
                        projection=dict(type="perspective"))
ESCALA_ARRIBA = dict(x=1.6, y=1.6, z=0.6)
ESCALA_INCLINADA = dict(x=1, y=1, z=0.6)
VISTAS = {"Desde arriba": {"scene.camera": CAMARA_ARRIBA,
                           "scene.aspectratio": ESCALA_ARRIBA},
          "Inclinada": {"scene.camera": CAMARA_INCLINADA,
                        "scene.aspectratio": ESCALA_INCLINADA}}


def texto_tiempo(seg):
    """Segundos -> texto corto (0.003 s, 1.25 s, 2 min 5 s)."""
    if seg < 0.01:
        return f"{seg:.3f} s"
    if seg < 60:
        return f"{seg:.2f} s"
    return f"{int(seg // 60)} min {seg % 60:.0f} s"


def _altura(v, log):
    """Con log, los picos enormes se aplanan (las 5 funciones valen >= 0)."""
    return np.log10(1 + v) if log else v


def _camara_giro(k, n):
    ang = -np.pi / 4 + 2 * np.pi * k / max(n, 1)
    return dict(eye=dict(x=1.7 * np.cos(ang), y=1.7 * np.sin(ang), z=0.9),
                up=dict(x=0, y=0, z=1), projection=dict(type="perspective"))


def figura_3d(f, lb, ub, res, optimo, log=True, n_cuadros=80, ms=150,
              n_malla=100, camara="arriba", rastro=False, opacidad=0.9,
              periodo_migracion=0, tiempo=None):
    """res: resultado de aha(..., guardar_posiciones=True) en 2 variables.
    camara: "arriba", "inclinada" o "girar".
    rastro: dibuja el camino de la mejor flor.
    periodo_migracion: cada cuantas vueltas hay migracion (solo para
    explicar). tiempo: segundos que tardo aha()."""
    g = np.linspace(lb, ub, n_malla)
    Z = np.array([[f(np.array([x, y])) for x in g] for y in g])
    H = _altura(Z, log)
    elevar = 0.015 * (H.max() - H.min())
    T = len(res["posiciones"]) - 1
    n = len(res["posiciones"][0])
    cuadros = np.unique(np.linspace(0, T, min(n_cuadros, T + 1))
                        .round().astype(int))

    def colibries(t):
        P = res["posiciones"][t][:, :2]
        vals = np.array([f(p) for p in P])
        if t == 0:
            colores = [COLOR_INICIAL] * n
            texto = [f"Colibrí #{i} (inicio)<br>valor = {vals[i]:.2e}"
                     for i in range(n)]
        else:
            acc = res["acciones"][t - 1]
            vue = res["vuelos"][t - 1]
            ok = res["mejoro"][t - 1]
            colores = list(COLORES_ACCION[acc])
            texto = []
            for i in range(n):
                if acc[i] == 2:
                    que = "migró a un lugar nuevo"
                else:
                    que = (f"{ACCION_CORTA[acc[i]]}, vuelo "
                           f"{VUELO_CORTO[vue[i]]}: "
                           + ("encontró una flor mejor" if ok[i]
                              else "no mejoró, se quedó"))
                texto.append(f"Colibrí #{i} ({que})<br>"
                             f"valor = {vals[i]:.2e}")
        return go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=_altura(vals, log) + elevar,
            mode="markers", text=texto, hoverinfo="text", showlegend=False,
            marker=dict(size=5, color=colores,
                        line=dict(width=1, color="black")))

    def mejor(t):
        p = res["mejores"][t][:2]
        v = f(p)
        return go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[_altura(v, log) + 2 * elevar],
            mode="markers", name="La mejor flor hasta ahora",
            text=[f"Mejor flor<br>valor = {v:.2e}"], hoverinfo="text",
            marker=dict(size=13, color="yellow", symbol="x",
                        line=dict(width=2, color="black")))

    def camino(t):
        P = np.array([res["mejores"][k][:2] for k in range(t + 1)])
        z = _altura(np.array([f(p) for p in P]), log) + 2 * elevar
        return go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=z, mode="lines", hoverinfo="skip",
            name="Camino de la mejor flor", line=dict(color="#d4a000", width=5))

    def titulo(t):
        txt = (f"<b>Vuelta {t} de {T}</b> · mejor valor encontrado = "
               f"{res['curva'][t]:.2e}")
        if tiempo is not None and t == T and T > 0:
            txt += (f" · <b><span style='color:#c0392b'>¡Terminó! El "
                    f"algoritmo tardó {texto_tiempo(tiempo)}</span></b>")
        return txt

    ancho = ub - lb

    def explicacion(k, t):
        if t == 0:
            return ("<b>Inicio.</b> Cada colibrí aparece en una flor al azar. "
                    "La flor con el valor más bajo es la <b>mejor flor</b> "
                    "(X amarilla).<br>Un colibrí solo se cambia de flor si "
                    "la nueva es mejor. Dale <b>▶ Reproducir</b> para "
                    "verlos volar.")
        acc = res["acciones"][t - 1]
        ok = res["mejoro"][t - 1]
        c = np.bincount(acc, minlength=3)
        quien = (f"<span style='color:{COLORES_ACCION[0]}'><b>■ {c[0]} "
                 f"guiado</b></span> · <span style='color:"
                 f"{COLORES_ACCION[1]}'><b>■ {c[1]} territorial</b></span>")
        # Migraciones desde el cuadro anterior (vueltas t_ant+1 .. t)
        t_ant = cuadros[k - 1]
        mig = 0
        if periodo_migracion:
            mig = sum(1 for v in range(t_ant, t) if v % periodo_migracion == 0)
        if mig:
            quien += (f" · <span style='color:{COLORES_ACCION[2]}'><b>■ "
                      f"{'hubo migración' if mig == 1 else f'{mig} migraciones'}"
                      "</b></span> (el de la peor flor se mudó al azar)")
        mejoraron = int(ok[acc < 2].sum())
        estado = (f"En la vuelta {t}, <b>{mejoraron} de {n}</b> colibríes "
                  "encontraron una flor mejor; los demás se quedaron "
                  "donde estaban.")
        antes, ahora = res["curva"][t_ant], res["curva"][t]
        if ahora < antes:
            novedad = (f"La mejor flor bajó ({antes:.2e} → {ahora:.2e}).")
        else:
            novedad = "La mejor flor no cambió en estas vueltas."
        mejor_xy = np.asarray(res["mejores"][t][:2])
        lejos = np.linalg.norm(mejor_xy - np.asarray(optimo)[:2]) / ancho
        if lejos < 0.001:
            novedad += " <b>¡Ya llegó al mínimo real!</b>"
        elif lejos < 0.02:
            novedad += " Ya está muy cerca del mínimo real."
        P = res["posiciones"][t][:, :2]
        dist = np.median(np.linalg.norm(P - mejor_xy, axis=1)) / ancho
        if dist > 0.25:
            grupo = "Siguen repartidos por todo el terreno."
        elif dist > 0.05:
            grupo = "Se están juntando cerca de la mejor flor."
        else:
            grupo = "Casi todos están encima de la mejor flor."
        return f"{quien}<br>{estado}<br>{novedad} {grupo}"

    def caja(k, t):
        return [dict(text=explicacion(k, t), xref="paper", yref="paper",
                     x=0, y=0.205, xanchor="left", yanchor="top",
                     align="left", showarrow=False, font=dict(size=13),
                     bgcolor="#eef5fc", bordercolor="#9cc3e6", borderwidth=1,
                     borderpad=8)]

    superficie = go.Surface(
        x=g, y=g, z=H, opacity=opacidad, showscale=False,
        colorscale=[[0, "#f4f4f4"], [1, "#8a8a8a"]],
        contours_z=dict(show=True, color="#aaa", width=1, usecolormap=False,
                        start=float(H.min()), end=float(H.max()),
                        size=float(H.max() - H.min()) / 8 or 1),
        hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>")
    v_opt = f(np.asarray(optimo))
    minimo = go.Scatter3d(
        x=[optimo[0]], y=[optimo[1]], z=[_altura(v_opt, log) + 2 * elevar],
        mode="markers", name="Mínimo real (la meta)", hoverinfo="text",
        text=[f"Mínimo real ({optimo[0]:g}, {optimo[1]:g})"],
        marker=dict(size=7, color="red", symbol="diamond",
                    line=dict(width=1, color="black")))
    leyenda = [go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                            name=nom, marker=dict(size=7, color=c,
                                                  line=dict(width=1,
                                                            color="black")))
               for nom, c in zip(LEYENDA_ACCIONES + ["Inicio (al azar)"],
                                 list(COLORES_ACCION) + [COLOR_INICIAL])]

    # Trazas 1 y 2 (y 3 con rastro) son las unicas que cambian
    animadas = [colibries, mejor] + ([camino] if rastro else [])
    fig = go.Figure(data=[superficie] + [a(0) for a in animadas]
                    + [minimo] + leyenda)
    indices = list(range(1, len(animadas) + 1))

    def layout_cuadro(k, t):
        lay = dict(title_text=titulo(t), annotations=caja(k, t))
        if camara == "girar":
            lay["scene_camera"] = _camara_giro(k, len(cuadros))
        return go.Layout(**lay)

    fig.frames = [go.Frame(data=[a(t) for a in animadas], traces=indices,
                           name=str(t), layout=layout_cuadro(k, t))
                  for k, t in enumerate(cuadros)]

    if camara == "arriba":
        cam0, escala0 = CAMARA_ARRIBA, ESCALA_ARRIBA
    elif camara == "inclinada":
        cam0, escala0 = CAMARA_INCLINADA, ESCALA_INCLINADA
    else:
        cam0, escala0 = _camara_giro(0, len(cuadros)), ESCALA_INCLINADA

    reproducir = dict(frame=dict(duration=ms, redraw=True), fromcurrent=True,
                      transition=dict(duration=0))
    pausar = dict(frame=dict(duration=0, redraw=False), mode="immediate",
                  transition=dict(duration=0))
    # De arriba hacia abajo: terreno 0.21-1, explicacion, botones 0.058,
    # barra de vueltas 0.
    menus = [dict(type="buttons", direction="left", x=0.0, y=0.058,
                  xanchor="left", yanchor="bottom", showactive=False,
                  pad=dict(t=0, b=0, l=0, r=0), font=dict(size=13),
                  buttons=[dict(label="▶ Reproducir", method="animate",
                                args=[None, reproducir]),
                           dict(label="⏸ Pausa", method="animate",
                                args=[[None], pausar])])]
    if camara != "girar":
        menus.append(dict(type="buttons", direction="down", x=0.99, y=0.6,
                          xanchor="right", yanchor="top", showactive=False,
                          pad=dict(t=0, b=0, l=0, r=0),
                          buttons=[dict(label=nom, method="relayout", args=[v])
                                   for nom, v in VISTAS.items()]))
    fig.update_layout(
        title=dict(text=titulo(0), x=0.01, xanchor="left", y=0.975,
                   yanchor="top", font=dict(size=15)),
        height=820, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white", font=dict(color="#333"),
        annotations=caja(0, 0),
        legend=dict(x=0.99, xanchor="right", y=0.98, yanchor="top",
                    title=dict(text="<b>Qué es cada cosa</b>",
                               font=dict(size=14)),
                    font=dict(size=13), itemsizing="constant",
                    bgcolor="#fafafa", bordercolor="#bbb", borderwidth=1),
        scene=dict(
            domain=dict(x=[0, 0.7], y=[0.21, 1]),
            xaxis_title="x", yaxis_title="y",
            zaxis_title="altura (log)" if log else "valor de la función",
            aspectmode="manual", aspectratio=escala0, camera=cam0),
        updatemenus=menus,
        sliders=[dict(
            x=0.0, y=0.0, len=1.0, xanchor="left", yanchor="bottom",
            currentvalue=dict(visible=False),
            pad=dict(t=0, b=0, l=0, r=0), ticklen=0,
            font=dict(color="rgba(0,0,0,0)"),
            steps=[dict(method="animate", label=str(t),
                        args=[[str(t)], dict(mode="immediate",
                                             frame=dict(duration=0,
                                                        redraw=True),
                                             transition=dict(duration=0))])
                   for t in cuadros])])
    return fig


# JavaScript de la pagina: Plotly ignora el mouse mientras anima, asi que al
# agarrar la grafica (o usar la rueda) se pausa, al soltar sigue, y la camara
# elegida se guarda en el layout para que los cuadros no la regresen.
_JS_GIRAR = """
(function () {
  var gd = document.getElementById('{plot_id}');
  var opciones = __OPCIONES__;
  var corriendo = false, reanudar = false, reloj = null;
  gd.on('plotly_animating', function () { corriendo = true; });
  gd.on('plotly_animated', function () { corriendo = false; });
  gd.on('plotly_animationinterrupted', function () { corriendo = false; });

  function guardar(e) {
    for (var k in e) {
      if (k.indexOf('scene.') !== 0) continue;
      var partes = k.split('.').slice(1), o = gd.layout.scene;
      for (var i = 0; i < partes.length - 1; i++) {
        o[partes[i]] = o[partes[i]] || {};
        o = o[partes[i]];
      }
      o[partes[partes.length - 1]] = JSON.parse(JSON.stringify(e[k]));
    }
  }
  gd.on('plotly_relayouting', guardar);
  gd.on('plotly_relayout', guardar);

  function enEscena(ev) {
    return ev.target && ev.target.closest && ev.target.closest('.gl-container');
  }
  function pausar() {
    reanudar = true;
    Plotly.animate(gd, [null], {mode: 'immediate', transition: {duration: 0},
                                frame: {duration: 0, redraw: false}});
  }
  function seguir() {
    reanudar = false;
    Plotly.animate(gd, null, opciones);
  }
  function agarrar(ev) {
    if (corriendo && !reanudar && enEscena(ev)) pausar();
  }
  function soltar() {
    if (reanudar && !reloj) setTimeout(seguir, 0);
  }
  gd.addEventListener('pointerdown', agarrar, true);
  gd.addEventListener('mousedown', agarrar, true);
  gd.addEventListener('touchstart', agarrar, true);
  window.addEventListener('pointerup', soltar, true);
  window.addEventListener('mouseup', soltar, true);
  window.addEventListener('touchend', soltar, true);
  gd.addEventListener('wheel', function (ev) {
    if (!enEscena(ev)) return;
    if (corriendo && !reanudar) pausar();
    if (reanudar) {
      clearTimeout(reloj);
      reloj = setTimeout(function () { reloj = null; seguir(); }, 500);
    }
  }, true);
})();
"""


def html_3d(fig):
    """Pagina HTML completa (plotly.js adentro, funciona sin internet).
    Espera a que el iframe se vea antes de dibujar: oculto, plotly mide los
    textos en 0 y encima los botones."""
    opciones = fig.layout.updatemenus[0].buttons[0].args[1]
    js = (_JS_GIRAR.replace("__OPCIONES__", json.dumps(opciones))
          .replace("{plot_id}", "simulacion_3d"))
    spec = fig.to_json().replace("</", r"<\/")
    config = json.dumps(dict(displaylogo=False, responsive=True))
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>html, body {{margin: 0; background: white;}}</style></head>
<body>
<div id="simulacion_3d" style="width:100%; height:{fig.layout.height}px"></div>
<script type="text/javascript">{get_plotlyjs()}</script>
<script type="text/javascript">
var fig = {spec};
function dibujar() {{
  Plotly.newPlot('simulacion_3d', fig.data, fig.layout, {config})
    .then(function () {{ return Plotly.addFrames('simulacion_3d', fig.frames || []); }})
    .then(function () {{ {js} }});
}}
function seVe() {{
  var r = document.getElementById('simulacion_3d').getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}}
(function esperar() {{
  if (seVe()) {{ dibujar(); return; }}
  var listo = false;
  function intentar() {{
    if (listo || !seVe()) return;
    listo = true; clearInterval(reloj); dibujar();
  }}
  var reloj = setInterval(intentar, 250);
  window.addEventListener('resize', intentar);
}})();
</script>
</body></html>"""
