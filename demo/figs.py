"""Plotly figures for the demo app (projector-friendly: large fonts, thick lines)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from lifi.plots import COLORS, SCHEME_COLORS, SCHEME_NAMES

FONT = 17
SEQ = [[0.0, "#eef5fd"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.45, "#6da7ec"],
       [0.6, "#3987e5"], [0.75, "#256abf"], [0.88, "#184f95"], [1.0, "#0d366b"]]
STAR = "#eda100"
INK = "#1a1a1a"
ACCENT = "#1f4e9a"
GRID = "#e6e6e6"
OUT_FOV = "#d9d9d9"

# Link states in the room view: colour + marker symbol (identity never by colour alone).
STATE_STYLE = {
    "Li-Fi": (COLORS["Li-Fi only"], "circle"),
    "Wi-Fi": (COLORS["Wi-Fi only"], "square"),
    "Light blocked (on Wi-Fi)": ("#5a5a5a", "x"),
    "Switching": ("#eda100", "diamond"),
}


def _layout(fig: go.Figure, title: str = "", height: int = 520, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=FONT + 3)), height=height,
                      font=dict(size=FONT, color=INK), plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=70, r=30, t=70, b=60),
                      legend=dict(font=dict(size=FONT - 1)), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#9a9a9a", showline=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#9a9a9a", showline=True)
    return fig


def room_heatmap(g: np.ndarray, z: np.ndarray, title: str, cbar: str, zmin: float, zmax: float,
                 led: np.ndarray, probe: tuple[float, float] | None = None,
                 contour: float | None = None, room: float = 6.0) -> go.Figure:
    """Heatmap over the room. z is (n, n) with NaN where there is no signal (shown grey)."""
    fig = go.Figure(go.Heatmap(x=g, y=g, z=z, colorscale=SEQ, zmin=zmin, zmax=zmax,
                               colorbar=dict(title=dict(text=cbar, side="right")),
                               hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>%{z:.1f}<extra></extra>"))
    if contour is not None:
        fig.add_trace(go.Contour(x=g, y=g, z=np.nan_to_num(z, nan=-200), showscale=False,
                                 contours=dict(start=contour, end=contour, size=1, coloring="none",
                                               showlabels=True, labelfont=dict(size=FONT - 3, color=INK)),
                                 line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip",
                                 name=f"{contour:g} dB edge", showlegend=False))
    fig.add_trace(go.Scatter(x=[led[0]], y=[led[1]], mode="markers", name="LED",
                             marker=dict(symbol="star", size=22, color=STAR, line=dict(color=INK, width=1.5)),
                             hovertemplate="LED<extra></extra>"))
    if probe is not None:
        fig.add_trace(go.Scatter(x=[probe[0]], y=[probe[1]], mode="markers", name="Probe",
                                 marker=dict(symbol="circle-open-dot", size=22, color=ACCENT,
                                             line=dict(width=3)),
                                 hovertemplate="probe (%{x:.2f}, %{y:.2f})<extra></extra>"))
    _layout(fig, title, height=520, showlegend=False)
    fig.update_layout(plot_bgcolor=OUT_FOV)
    fig.update_xaxes(title="x (m)", range=[0, room], constrain="domain", showgrid=False)
    fig.update_yaxes(title="y (m)", range=[0, room], scaleanchor="x", showgrid=False)
    return fig


def ber_vs_distance(r: np.ndarray, ber: dict[int, np.ndarray], fov_edge: float,
                    probe_r: float | None = None) -> go.Figure:
    fig = go.Figure()
    for M, b in ber.items():
        fig.add_trace(go.Scatter(x=r, y=np.maximum(b, 1e-15), name=SCHEME_NAMES[M], mode="lines",
                                 line=dict(color=SCHEME_COLORS[M], width=3),
                                 hovertemplate=f"{SCHEME_NAMES[M]}: r=%{{x:.2f}} m, BER=%{{y:.2e}}<extra></extra>"))
    fig.add_hline(y=1e-3, line=dict(color="#6b6b6b", dash="dash", width=1.5),
                  annotation_text="BER = 1e-3", annotation_position="top left")
    if fov_edge < r[-1]:
        fig.add_vline(x=fov_edge, line=dict(color="#6b6b6b", dash="dot", width=2),
                      annotation_text="FOV edge", annotation_position="bottom left")
    if probe_r is not None:
        fig.add_vline(x=probe_r, line=dict(color=ACCENT, width=2.5),
                      annotation_text="probe", annotation_position="top right")
    _layout(fig, "BER vs horizontal distance from the LED axis", height=520)
    fig.update_yaxes(type="log", title="Bit error rate", range=[-12, 0], exponentformat="power")
    fig.update_xaxes(title="Horizontal distance from LED axis (m)")
    fig.update_layout(legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"))
    return fig


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of runs of True."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def throughput_timeline(t: np.ndarray, rates: dict[str, np.ndarray], blocked: np.ndarray,
                        t_now: float | None = None, title: str = "Speed over time") -> go.Figure:
    fig = go.Figure()
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    for a, b in _intervals(blocked):
        fig.add_vrect(x0=t[a], x1=t[b - 1] + dt, fillcolor="#eb6834", opacity=0.13, line_width=0, layer="below")
    styles = {"Wi-Fi only": dict(color=COLORS["Wi-Fi only"], width=2.5),
              "Li-Fi only": dict(color=COLORS["Li-Fi only"], width=2.5),
              "Hybrid": dict(color=INK, width=2.2, dash="dash")}
    for name, r in rates.items():
        fig.add_trace(go.Scatter(x=t, y=r, name=name, mode="lines", line=styles[name],
                                 hovertemplate=f"{name}: %{{y:.0f}} Mbps at %{{x:.1f}} s<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Light blocked (shaded)",
                             marker=dict(symbol="square", size=14, color="rgba(235,104,52,0.3)")))
    if t_now is not None:
        fig.add_vline(x=t_now, line=dict(color=ACCENT, width=2))
    _layout(fig, title, height=430)
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Speed (Mbps)", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
                      title=dict(y=0.98), margin=dict(t=95), hovermode="x unified")
    return fig


def room_view(path: np.ndarray, states: np.ndarray, t: np.ndarray, led: np.ndarray, ap: np.ndarray,
              i_now: int, room: float = 6.0, frame_every: int = 5, trail_s: float = 10.0) -> go.Figure:
    """Top-down room: LED, Wi-Fi AP, the user's last `trail_s` seconds and a marker coloured by link.

    Plotly frames (every `frame_every` steps) drive the Play button.
    """
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    n_trail = max(int(trail_s / dt), 1)

    def trail(i):
        j = max(0, i - n_trail)
        return dict(x=path[j:i + 1, 0], y=path[j:i + 1, 1], mode="lines", line=dict(color="#8a8a8a", width=2.5),
                    hoverinfo="skip", showlegend=False)

    def marker(i):
        c, sym = STATE_STYLE[states[i]]
        return dict(x=[path[i, 0]], y=[path[i, 1]], mode="markers", showlegend=False,
                    marker=dict(size=24, color=c, symbol=sym, line=dict(color=INK, width=1.5)),
                    hovertemplate=f"t = {t[i]:.1f} s<br>{states[i]}<extra></extra>")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[led[0]], y=[led[1]], mode="markers", name="Light (LED)",
                             marker=dict(symbol="star", size=24, color=STAR, line=dict(color=INK, width=1.5))))
    fig.add_trace(go.Scatter(x=[ap[0]], y=[ap[1]], mode="markers", name="Wi-Fi router", cliponaxis=False,
                             marker=dict(symbol="triangle-up", size=22, color=COLORS["Wi-Fi only"],
                                         line=dict(color=INK, width=1.5))))
    fig.add_trace(go.Scatter(**trail(i_now)))
    fig.add_trace(go.Scatter(**marker(i_now)))
    for name, (c, sym) in STATE_STYLE.items():          # legend entries for the link states
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=13, color=c, symbol=sym, line=dict(color=INK, width=1))))
    idx = list(range(0, len(t), frame_every))
    fig.frames = [go.Frame(data=[go.Scatter(**trail(i)), go.Scatter(**marker(i))], traces=[2, 3],
                           name=f"{t[i]:.1f}") for i in idx]
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
        pad=dict(b=6), font=dict(size=FONT - 3, color=INK), bgcolor="#f3f5f8", bordercolor="#c9ced6",
        buttons=[dict(label="Play", method="animate",
                      args=[None, dict(frame=dict(duration=50, redraw=False), fromcurrent=True,
                                       transition=dict(duration=0))]),
                 dict(label="Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    _layout(fig, "", height=560)
    lo, hi = min(-0.3, ap[0] - 0.3, ap[1] - 0.3), max(room + 0.3, ap[0] + 0.3, ap[1] + 0.3)
    fig.update_xaxes(title="x (m)", range=[lo, hi], constrain="domain", showgrid=False)
    fig.update_yaxes(title="y (m)", range=[lo, hi], scaleanchor="x", showgrid=False)
    fig.add_shape(type="rect", x0=0, y0=0, x1=room, y1=room, line=dict(color=INK, width=1.5), layer="below")
    fig.update_layout(legend=dict(orientation="h", x=0, y=-0.2, yanchor="top", font=dict(size=FONT - 4)),
                      margin=dict(l=60, r=20, t=50, b=130))
    return fig


def wifi_fit_plot(d: np.ndarray, rssi: np.ndarray, a: float, n: float, sigma: float) -> go.Figure:
    dd = np.logspace(np.log10(max(d.min(), 1.0) * 0.9), np.log10(d.max() * 1.1), 100)
    fit = a - 10 * n * np.log10(dd)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[dd, dd[::-1]], y=np.r_[fit + sigma, (fit - sigma)[::-1]],
                             fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0),
                             name="Typical spread (± 1 std)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dd, y=fit, mode="lines", name=f"Fit: {a:.1f} − 10·{n:.2f}·log10(d)",
                             line=dict(color=COLORS["Wi-Fi only"], width=3)))
    fig.add_trace(go.Scatter(x=d, y=rssi, mode="markers", name="Your measurements",
                             marker=dict(size=12, color=COLORS["Li-Fi only"], line=dict(color=INK, width=1)),
                             hovertemplate="d=%{x:.2f} m<br>RSSI=%{y:.1f} dBm<extra></extra>"))
    _layout(fig, "Signal strength against distance from the router", height=470)
    fig.update_xaxes(title="Distance to router (m, log scale)", type="log")
    fig.update_yaxes(title="RSSI (dBm)")
    fig.update_layout(legend=dict(x=0.98, y=0.98, xanchor="right", bgcolor="rgba(255,255,255,0.8)"))
    return fig
