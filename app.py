"""Li-Fi with Wi-Fi backup: project demo.  Run:  streamlit run app.py

Three pages: Project, Live demo, Your data (optional). The app opens on simulated data;
user-provided data is off unless the viewer enables it, and is always labelled on screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from demo import content as C
from demo import figs
from lifi.ber import ber_pam, monte_carlo_pam
from lifi.channel import lifi_rate, lifi_snr, to_db
from lifi.config import Config
from lifi.hybrid import generate_trace, hysteresis_policy, metrics
from lifi.mobility import stationary_blockage
from lifi.userdata import (DataError, build_trace, blockage_steps, fit_wifi, parse_blockage, parse_path,
                           parse_wifi, path_warnings, resample_blockage, resample_path, switching_mask)

DEFAULT = Config()
DISCLAIMER = "This is a software simulation using literature-based parameters, not a hardware measurement."
SPEED, DISCONNECTED = "Speed (Mbps)", "Time disconnected (%)"
SWITCHES, ON_LIFI = "Switches between Li-Fi and Wi-Fi", "Time on Li-Fi (%)"
INK, ACCENT = "#1a1a1a", "#1f4e9a"

st.set_page_config(page_title="Li-Fi with Wi-Fi backup", layout="wide")
# Explicit colours everywhere in custom HTML (never inherited from the theme).
st.markdown(f"""
<style>
html, body {{ font-size: 18px; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}
[data-testid="stTab"] p {{ font-size: 1.1rem; font-weight: 600; }}
.pipeline {{ display: flex; align-items: center; gap: 0.6rem; margin: 1.2rem 0 1.6rem 0;
            flex-wrap: wrap; user-select: none; -webkit-user-select: none; }}
.pipeline .step {{ flex: 1 1 0; min-width: 140px; background: #eef3fb; color: {INK};
                  border: 1.5px solid {ACCENT}; border-radius: 8px; padding: 0.75rem 0.8rem; }}
.pipeline .step .name {{ color: {INK}; font-weight: 700; font-size: 1.0rem; }}
.pipeline .step .what {{ color: #3d3d3d; font-size: 0.9rem; margin-top: 0.2rem; }}
.pipeline .arrow {{ color: #4a4a4a; font-size: 1.4rem; }}
.pipeline ::selection {{ background: transparent; }}
.userbanner {{ background: #fdecee; color: #7a0c1e; border-left: 5px solid #b3132b;
              padding: 0.55rem 0.9rem; font-weight: 600; margin: 0.3rem 0 0.9rem 0; }}
.scheme {{ color: {INK}; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.2rem; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: #474747; }}
[data-testid="stTable"] th, [data-testid="stTable"] td {{ color: {INK}; }}
[data-testid="stMetricLabel"] p {{ color: #3d3d3d; }}
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small {{ color: #474747; }}
</style>
""", unsafe_allow_html=True)


def user_banner(what: str) -> None:
    st.markdown(f'<div class="userbanner">Using user-provided data: {what}</div>', unsafe_allow_html=True)


def missing(what: str) -> None:
    st.warning(f"{what} not found. Regenerate the results with `{C.REGEN}` (about 15 s).")


# ============================================================== cached compute
@st.cache_data(show_spinner=False)
def room_maps(h_led: float, half_deg: float, fov_deg: float, p_t: float, b_mhz: float, n: int):
    cfg = DEFAULT.with_(h_led=h_led, half_angle_deg=half_deg, fov_deg=fov_deg, p_t=p_t, b_lifi=b_mhz * 1e6)
    g = np.linspace(0, cfg.room, n)
    X, Y = np.meshgrid(g, g)
    snr, p = lifi_snr(np.column_stack([X.ravel(), Y.ravel()]), cfg)
    lit = (snr > 0).reshape(X.shape)
    p_dbm = np.where(lit, to_db(p / 1e-3).reshape(X.shape), np.nan)
    snr_db = np.where(lit, to_db(snr).reshape(X.shape), np.nan)
    cov = float(100 * (to_db(snr) >= cfg.coverage_snr_db).mean())
    r = np.linspace(0, 4.2, 300)
    s_line, _ = lifi_snr(np.column_stack([cfg.led_pos[0] + r, np.full_like(r, cfg.led_pos[1])]), cfg)
    ber = {M: ber_pam(s_line, M) for M in (2, 4, 8)}
    return g, p_dbm, snr_db, cov, r, ber


@st.cache_data(show_spinner=False)
def prototype_stream_state(seed: int) -> dict:
    """Generator state after the prototype's OOK Monte Carlo draws (as in experiments.prototype_baseline)."""
    rng = np.random.default_rng(seed)
    monte_carlo_pam(np.arange(4, 17, 1.0), 2, 2_000_000, rng)
    return rng.bit_generator.state


def make_rng(seed: int, prototype_stream: bool) -> np.random.Generator:
    rng = np.random.default_rng(seed)
    if prototype_stream:
        rng.bit_generator.state = prototype_stream_state(seed)
    return rng


def summarise_run(tr, hy) -> pd.DataFrame:
    rows = []
    for name, r, sw, share in (("Wi-Fi only", tr.r_w, 0, 0.0), ("Li-Fi only", tr.r_l, 0, 100.0),
                               ("Hybrid", hy.rate, hy.switches, 100 * hy.use_lifi.mean())):
        m = metrics(r, DEFAULT)
        rows.append({"Scheme": name, SPEED: m["avg"], DISCONNECTED: m["outage"], SWITCHES: int(sw), ON_LIFI: share})
    return pd.DataFrame(rows)


def fmt_run(df: pd.DataFrame, cols=(SPEED, DISCONNECTED)) -> pd.DataFrame:
    """Display copy of a run table: chosen columns, fixed decimals."""
    out = df.set_index("Scheme")[list(cols)].copy()
    for c, nd in ((SPEED, 1), (DISCONNECTED, 2), (ON_LIFI, 1)):
        if c in out:
            out[c] = out[c].map(lambda v, nd=nd: f"{v:.{nd}f}")
    return out


def link_states(tr, hy, cost: int) -> np.ndarray:
    sw = switching_mask(hy, cost)
    st_ = np.where(hy.use_lifi, "Li-Fi", np.where(tr.blocked, "Light blocked (on Wi-Fi)", "Wi-Fi"))
    return np.where(sw, "Switching", st_)


def wifi_cfg(cfg: Config, wifi_override) -> Config:
    if wifi_override is None:
        return cfg
    ax, ay, a, n, s = wifi_override
    return cfg.with_(wifi_xy=(ax, ay), wifi_ptx_dbm=a, wifi_pl0_db=0.0, wifi_pl_exp=n, wifi_shadow_db=s)


@st.cache_data(show_spinner=False)
def run_demo(seed: int, p_in: float, up: float, down: float, cost: int, proto: bool,
             wifi_override: tuple | None = None):
    """One hybrid run (cached). wifi_override = (ap_x, ap_y, rssi_1m, n, sigma) or None."""
    cfg = wifi_cfg(DEFAULT.with_(p_block_in=p_in, thr_hi_db=up, thr_lo_db=down, switch_cost_steps=cost),
                   wifi_override)
    tr = generate_trace(cfg, make_rng(seed, proto))
    hy = hysteresis_policy(tr, up, down, cost)
    return tr, hy, link_states(tr, hy, cost), summarise_run(tr, hy), cfg.wifi_pos


@st.cache_data(show_spinner=False, max_entries=20)
def run_user(seed: int, wifi_override, path: np.ndarray | None, blocked: np.ndarray | None, steps: int | None):
    cfg = wifi_cfg(DEFAULT, wifi_override)
    tr = build_trace(cfg, np.random.default_rng(seed), path=path, blocked=blocked, steps=steps)
    hy = hysteresis_policy(tr, cfg.thr_hi_db, cfg.thr_lo_db, cfg.switch_cost_steps)
    return tr, hy, link_states(tr, hy, cfg.switch_cost_steps), summarise_run(tr, hy), cfg.wifi_pos


STATE_TEXT = {"Li-Fi": "on **Li-Fi**", "Wi-Fi": "on **Wi-Fi**",
              "Light blocked (on Wi-Fi)": "on **Wi-Fi** because the light is blocked",
              "Switching": "**switching** between links (no data for a moment)"}


def run_views(tr, hy, states, table, ap, key: str, title: str = "Speed over time") -> None:
    """Shared layout: time slider, room view + small table, speed-over-time chart."""
    t = np.arange(len(tr.r_w)) * tr.dt
    t_now = st.slider("Time (s)", 0.0, float(t[-1]), 0.0, step=0.5, key=f"{key}_t")
    i = int(min(round(t_now / tr.dt), len(t) - 1))
    left, right = st.columns([1, 1], gap="large")
    left.plotly_chart(figs.room_view(tr.path, states, t, DEFAULT.led_pos, ap, i, DEFAULT.room),
                      width="stretch", theme=None, key=f"{key}_room")
    with right:
        st.markdown(f"At {t[i]:.1f} s the user is {STATE_TEXT[states[i]]}.")
        st.table(fmt_run(table))
        sw = int(table[SWITCHES].iloc[2])
        st.caption(f"This run: {t[-1] + tr.dt:.0f} s. The hybrid switched {sw} times and spent "
                   f"{table[ON_LIFI].iloc[2]:.0f}% of the time on Li-Fi. One run is a random sample; "
                   "the Project page shows the average of 30 runs.")
    win = st.radio("Show", ["60 s around the selected time", "Whole run"], horizontal=True, key=f"{key}_win",
                   label_visibility="collapsed")
    fig = figs.throughput_timeline(t, {"Wi-Fi only": tr.r_w, "Li-Fi only": tr.r_l, "Hybrid": hy.rate},
                                   tr.blocked, t[i], title=title)
    if win.startswith("60") and t[-1] > 60:
        x0 = float(np.clip(t[i] - 30, 0, t[-1] - 60))
        fig.update_xaxes(range=[x0, x0 + 60])
    st.plotly_chart(fig, width="stretch", theme=None, key=f"{key}_speed")


# ================================================================ user data
# Uploaded files live only in this browser session's memory (Streamlit UploadedFile); nothing is
# written to disk. The widgets are drawn on the "Your data" page; their values are read here so
# that every page sees the same state in one script run.
UD_FILES = {"wifi": ("ud_wifi_file", "wifi_rssi_example.csv"),
            "path": ("ud_path_file", "user_path_example.csv"),
            "block": ("ud_block_file", "blockage_example.csv")}
st.session_state.setdefault("ud_samples", False)


@st.cache_data(show_spinner=False, max_entries=20)
def _parse(kind: str, data: bytes):
    """Parse one uploaded file. Returns (DataFrame | None, error message | None, warnings)."""
    try:
        if kind == "wifi":
            return parse_wifi(data, DEFAULT.room), None, []
        if kind == "path":
            df = parse_path(data, DEFAULT.room)
            return df, None, path_warnings(df)
        return parse_blockage(data), None, []
    except DataError as e:
        return None, str(e), []
    except Exception as e:  # noqa: BLE001 - never crash on user input
        return None, f"Could not use this file ({type(e).__name__}: {e}).", []


def user_source(kind: str) -> tuple[bytes | None, str | None]:
    """Bytes of the uploaded file (preferred) or the loaded sample, and a label for it."""
    key, sample = UD_FILES[kind]
    up = st.session_state.get(key)
    if up is not None:
        return up.getvalue(), f"uploaded file '{up.name}'"
    if st.session_state.ud_samples and (C.EXAMPLES / sample).exists():
        return (C.EXAMPLES / sample).read_bytes(), f"sample file '{sample}'"
    return None, None


UD = {}
for _k in UD_FILES:
    _b, _label = user_source(_k)
    UD[_k] = dict(label=_label, df=None, err=None, warn=[])
    if _b is not None:
        UD[_k]["df"], UD[_k]["err"], UD[_k]["warn"] = _parse(_k, _b)
AP_XY = (float(st.session_state.get("ud_ap_x", 0.0)), float(st.session_state.get("ud_ap_y", 0.0)))
WIFI_FIT, WIFI_FIT_ERR = None, None
if UD["wifi"]["df"] is not None:
    try:
        WIFI_FIT = fit_wifi(UD["wifi"]["df"], AP_XY, DEFAULT.wifi_d_min)
    except DataError as e:
        WIFI_FIT_ERR = str(e)
USE_WIFI = bool(st.session_state.get("ud_use_wifi", False)) and WIFI_FIT is not None
USE_PATH = bool(st.session_state.get("ud_use_path", False)) and UD["path"]["df"] is not None
USE_BLOCK = bool(st.session_state.get("ud_use_block", False)) and UD["block"]["df"] is not None
WIFI_OVERRIDE = ((AP_XY[0], AP_XY[1], WIFI_FIT.rssi_1m_dbm, WIFI_FIT.pl_exp, WIFI_FIT.sigma_db)
                 if USE_WIFI else None)

# Live-demo advanced settings are drawn inside an expander below the results, so their values are
# initialised here and read from session state before the run.
ADV = {"hy_up": DEFAULT.thr_hi_db, "hy_down": DEFAULT.thr_lo_db, "hy_cost": DEFAULT.switch_cost_steps,
       "hy_proto": False}
CH_KEYS = {"ch_h": DEFAULT.h_led, "ch_half": DEFAULT.half_angle_deg, "ch_fov": DEFAULT.fov_deg,
           "ch_pt": DEFAULT.p_t, "ch_b": DEFAULT.b_lifi / 1e6, "pr_x": 4.5, "pr_y": 4.5}
for _k, _v in {**ADV, **CH_KEYS}.items():
    st.session_state.setdefault(_k, _v)


def reset_channel() -> None:
    for k, v in CH_KEYS.items():
        st.session_state[k] = v


# ===================================================================== pages
pages = st.tabs(["Project", "Live demo", "Your data (optional)"])

# --------------------------------------------------------------- 1. Project
with pages[0]:
    st.title("Li-Fi with Wi-Fi backup")
    st.caption(DISCLAIMER)
    st.markdown(
        "Li-Fi sends internet data with light from a ceiling lamp. It is fast, but it stops the moment "
        "someone blocks the light or you walk out of the beam. Wi-Fi is slower but reaches the whole room, "
        "so it makes a good backup. We simulate a person walking around a 6 × 6 m room and test a system "
        "that switches between the two.")
    st.markdown(f"""
<div class="pipeline">
  <div class="step"><div class="name">Mobility + blockage</div><div class="what">user walks, people block the light</div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="name">Li-Fi channel + Wi-Fi</div><div class="what">how much signal reaches the user</div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="name">SNR</div><div class="what">signal strength against noise</div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="name">Link selector</div><div class="what">picks Li-Fi or Wi-Fi</div></div>
  <div class="arrow">→</div>
  <div class="step"><div class="name">Metrics</div><div class="what">speed and time disconnected</div></div>
</div>""", unsafe_allow_html=True)

    s6 = C.read_result_csv("E6_statistics_summary")
    if s6 is None:
        missing("Results file results/data/E6_statistics_summary.csv")
    else:
        g6 = {("Hybrid" if r.policy.startswith("Hysteresis") else r.policy): r for _, r in s6.iterrows()}
        n = int(s6.n_seeds.iloc[0])
        st.subheader(f"Average of {n} simulated runs")
        cols = st.columns(3, gap="large")
        for col, name in zip(cols, ("Wi-Fi only", "Li-Fi only", "Hybrid")):
            r = g6[name]
            col.markdown(f'<div class="scheme">{name}</div>', unsafe_allow_html=True)
            col.metric(SPEED, f"{r.avg_mean:.1f}")
            col.metric(DISCONNECTED, f"{r.outage_mean:.2f}")
        st.caption("95% confidence intervals: " + "; ".join(
            f"{k} ± {g6[k].avg_ci95:.1f} Mbps and ± {g6[k].outage_ci95:.2f} points"
            for k in ("Wi-Fi only", "Li-Fi only", "Hybrid")) +
            f". Each run is {DEFAULT.sim_time:.0f} s; 'disconnected' means less than {DEFAULT.outage_mbps:g} Mbps.")
        h, li, wi = g6["Hybrid"], g6["Li-Fi only"], g6["Wi-Fi only"]
        speed_cmp = (f"{li.avg_mean - h.avg_mean:.1f} Mbps slower than Li-Fi alone" if h.avg_mean < li.avg_mean
                     else f"{h.avg_mean - li.avg_mean:.1f} Mbps faster than Li-Fi alone")
        out_cmp = "less" if h.outage_mean < li.outage_mean else "more"
        st.markdown(f"The hybrid is {h.avg_mean / wi.avg_mean:.1f} times as fast as Wi-Fi alone but "
                    f"{speed_cmp}, and it is disconnected {out_cmp} often than Li-Fi alone "
                    f"({h.outage_mean:.2f}% against {li.outage_mean:.2f}% of the time).")

    st.write("")
    with st.expander("More experiments", expanded=False):
        exp = st.selectbox("Experiment", list(C.EXPERIMENT_TITLES.values()), key="gal_exp").split(" ")[0]
        para = C.explainer_paragraph(exp)
        if para:
            st.markdown(para)
        fig_path, csvs = C.experiment_figure(exp), C.experiment_csvs(exp)
        if fig_path is not None:
            st.image(str(fig_path), width="stretch")
        elif exp == "S1" and csvs:
            st.dataframe(pd.read_csv(csvs[0]).round(2), width="stretch", hide_index=True)
        else:
            missing(f"Figure for {exp} (results/figures/{exp}_*.png)")
        if csvs:
            dl = st.columns(max(len(csvs), 3))
            for col, p in zip(dl, csvs):
                col.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, mime="text/csv",
                                    key=f"dl_{p.name}")
        else:
            missing(f"Data for {exp} (results/data/{exp}_*.csv)")

    with st.expander("Limitations and honest notes", expanded=False):
        st.caption("Taken directly from README.md.")
        for heading in ("Known issues / possible improvements", "Known limits",
                        "Modelling assumptions and limitations", "Parameters"):
            body = C.markdown_section("README.md", heading)
            st.markdown(f"#### {heading}")
            st.markdown(body if body else f"Section '{heading}' not found in README.md.")

    with st.expander("Policy comparison (E7)", expanded=False):
        s7 = C.read_result_csv("E7_policy_comparison")
        if s7 is None:
            missing("Results file results/data/E7_policy_comparison.csv")
        else:
            st.markdown("The same 30 runs with different switching rules. A single threshold switches as soon as "
                        "the signal crosses one level; hysteresis uses a higher level to move to Li-Fi and a lower "
                        "one to move back. Ideal always picks the faster link with no switching delay, which no real "
                        "system can do.")
            t7 = pd.DataFrame({"Rule": s7.policy, SPEED: s7.avg_mean.map("{:.1f}".format),
                               DISCONNECTED: s7.outage_mean.map("{:.2f}".format),
                               SWITCHES: s7.switches_mean.map("{:.1f}".format)}).set_index("Rule")
            st.table(t7)
            st.caption("Means over 30 runs. 95% confidence intervals are in results/data/E7_policy_comparison.csv.")

# ------------------------------------------------------------- 2. Live demo
with pages[1]:
    st.header("Live demo")
    st.markdown("A person walks around the room for five minutes while others sometimes block the light. "
                "Pick a random seed and how often the light is blocked; the results update immediately.")
    BLOCK_OPTS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
    c1, c2, _ = st.columns([1, 2, 1], gap="large")
    seed = int(c1.number_input("Random seed", 0, 10_000, 42, step=1, key="hy_seed"))
    p_in = c2.select_slider("Light blocked", BLOCK_OPTS, value=DEFAULT.p_block_in, key="hy_pin",
                            format_func=lambda p: f"{100 * stationary_blockage(p, DEFAULT.p_block_out):.0f}% of the time")

    up, down = float(st.session_state.hy_up), float(st.session_state.hy_down)
    cost, proto = int(st.session_state.hy_cost), bool(st.session_state.hy_proto)
    if WIFI_OVERRIDE is not None:
        user_banner(f"fitted Wi-Fi model from the {UD['wifi']['label']} (router at {AP_XY[0]:g}, {AP_XY[1]:g} m). "
                    "Turn it off on the 'Your data' page.")
    if down >= up:
        st.error(f"The switch-back level ({down:g} dB) must be lower than the switch-to-Li-Fi level ({up:g} dB). "
                 "Change it under Advanced settings below.")
    else:
        run_views(*run_demo(seed, p_in, up, down, cost, proto, WIFI_OVERRIDE), key="hy")

    with st.expander("Advanced settings and channel explorer", expanded=False):
        st.markdown("**Switching rule.** *Hysteresis* means using two levels: move to Li-Fi when its signal "
                    "is above the first, and only move back when it falls below the second, lower one. "
                    "This stops the link flipping back and forth.")
        a1, a2, a3 = st.columns(3, gap="large")
        a1.slider("Move to Li-Fi above SNR (dB)", 5.0, 30.0, step=0.5, key="hy_up")
        a2.slider("Move back to Wi-Fi below SNR (dB)", 0.0, 25.0, step=0.5, key="hy_down")
        a3.slider("Time lost per switch (steps of 0.1 s)", 0, 10, key="hy_cost")
        st.checkbox("Replay the original prototype's seed-42 run (with all default settings this reproduces "
                    "its numbers exactly: 120.0 / 438.3 / 407.8 Mbps, 67 switches)", key="hy_proto")

        st.divider()
        st.markdown("**Channel explorer.** *SNR* (signal-to-noise ratio) is how strong the light signal is "
                    "compared with the noise, in dB. *BER* (bit error rate) is the fraction of bits received "
                    "wrongly. OOK, 4-PAM and 8-PAM send 1, 2 and 3 bits per light pulse.")
        c = st.columns([1, 1, 1, 1, 1, 0.6])
        h = c[0].slider("LED height (m)", 2.25, 4.0, step=0.05, key="ch_h")
        half = c[1].slider("Beam half-angle (°)", 15.0, 75.0, step=1.0, key="ch_half")
        fov = c[2].slider("Receiver field of view (°)", 30.0, 90.0, step=1.0, key="ch_fov")
        pt = c[3].slider("LED power (W)", 0.25, 5.0, step=0.25, key="ch_pt")
        bw = c[4].slider("Bandwidth (MHz)", 10.0, 300.0, step=10.0, key="ch_b")
        c[5].write("")
        c[5].button("Reset", on_click=reset_channel, key="ch_reset")
        p1, p2 = st.columns(2, gap="large")
        px = p1.slider("Probe position x (m)", 0.0, DEFAULT.room, step=0.1, key="pr_x")
        py = p2.slider("Probe position y (m)", 0.0, DEFAULT.room, step=0.1, key="pr_y")

        cfg2 = DEFAULT.with_(h_led=h, half_angle_deg=half, fov_deg=fov, p_t=pt, b_lifi=bw * 1e6)
        g, p_dbm, snr_db, cov, r, ber = room_maps(h, half, fov, pt, bw, DEFAULT.grid_n)
        cov0 = room_maps(DEFAULT.h_led, DEFAULT.half_angle_deg, DEFAULT.fov_deg, DEFAULT.p_t,
                         DEFAULT.b_lifi / 1e6, DEFAULT.grid_n)[3]
        snr_p, _ = lifi_snr(np.array([[px, py]]), cfg2)
        dh = float(np.hypot(px - cfg2.led_pos[0], py - cfg2.led_pos[1]))
        lit = snr_p[0] > 0
        probe = pd.DataFrame({"Value": [
            f"{cov:.1f}% of the room (default {cov0:.1f}%)",
            f"{np.hypot(dh, h - DEFAULT.h_rx):.2f} m (horizontal {dh:.2f} m)",
            f"{10 * np.log10(snr_p[0]):.1f} dB" if lit else "no signal (outside the receiver's field of view)",
            f"{float(lifi_rate(snr_p, cfg2)[0]):.0f} Mbps",
            *(f"{float(ber_pam(snr_p, M)[0]):.1e}" for M in (2, 4, 8))]},
            index=[f"Coverage (SNR ≥ {DEFAULT.coverage_snr_db:g} dB)", "Distance from probe to LED", "SNR at probe",
                   "Li-Fi speed at probe", "BER, OOK", "BER, 4-PAM", "BER, 8-PAM"])
        st.table(probe)
        f1, f2 = st.columns(2, gap="large")
        f1.plotly_chart(figs.room_heatmap(g, p_dbm, "Received light power", "dBm", -35, -5, cfg2.led_pos,
                                          (px, py), room=DEFAULT.room), width="stretch", theme=None, key="ch_pmap")
        f2.plotly_chart(figs.room_heatmap(g, snr_db, f"SNR (bandwidth {bw:g} MHz)", "dB", 0, 35, cfg2.led_pos,
                                          (px, py), contour=DEFAULT.coverage_snr_db, room=DEFAULT.room),
                        width="stretch", theme=None, key="ch_smap")
        fov_edge = float(np.tan(np.deg2rad(fov)) * (h - DEFAULT.h_rx)) if fov < 90 else np.inf
        st.plotly_chart(figs.ber_vs_distance(r, ber, fov_edge, dh if dh <= r[-1] else None),
                        width="stretch", theme=None, key="ch_ber")
        st.caption("Grey areas get no direct light. The dashed line marks the coverage edge. "
                   "Light bouncing off walls is not modelled.")

# ------------------------------------------------------- 3. Your data (optional)
TEMPLATES = {"wifi": "x_m,y_m,rssi_dbm\n", "path": "t_s,x_m,y_m\n", "block": "t_s,blocked\n"}


def load_samples() -> None:
    st.session_state.ud_samples = True


def clear_samples() -> None:
    st.session_state.ud_samples = False
    for k in ("ud_use_wifi", "ud_use_path", "ud_use_block"):
        st.session_state[k] = False


def file_step(kind: str, cols: str, help_text: str) -> None:
    key, sample = UD_FILES[kind]
    st.file_uploader(f"CSV file with columns {cols}", type=["csv", "txt"], key=key, help=help_text)
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Empty template", TEMPLATES[kind], file_name=f"template_{sample.replace('_example', '')}",
                       mime="text/csv", key=f"tpl_{kind}")
    if (C.EXAMPLES / sample).exists():
        d2.download_button("Sample file", (C.EXAMPLES / sample).read_bytes(), file_name=sample,
                           mime="text/csv", key=f"smp_{kind}")
    u = UD[kind]
    if u["err"]:
        st.error(f"{u['err']} (Source: {u['label']}.)")
    elif u["df"] is not None:
        st.caption(f"Loaded {len(u['df'])} rows from the {u['label']}.")
        for w in u["warn"]:
            st.warning(w)


with pages[2]:
    st.header("Your data")
    st.markdown("Replace parts of the simulation with your own measurements. Nothing here is used until you "
                "tick a box, and anything that uses your data is marked in red. Files stay in this browser "
                "session only and are never saved.")
    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("Load sample files", on_click=load_samples, key="ud_load",
              help="Synthetic example files from data/examples/ (not real measurements)")
    b2.button("Clear and turn everything off", on_click=clear_samples, key="ud_clear")
    if st.session_state.ud_samples:
        st.caption("Sample files loaded. They are synthetic examples made by the project, not real measurements. "
                   "An uploaded file replaces the sample of the same kind.")

    st.subheader("1. Wi-Fi measurements")
    file_step("wifi", "x_m, y_m, rssi_dbm",
              "One row per spot: position in metres inside the 6 x 6 m room and signal strength (RSSI) in dBm, "
              "for example from a Wi-Fi analyser app. At least 6 rows.")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.number_input("Router position x (m)", -20.0, 26.0, 0.0, step=0.1, key="ud_ap_x")
    c2.number_input("Router position y (m)", -20.0, 26.0, 0.0, step=0.1, key="ud_ap_y")
    if WIFI_FIT_ERR:
        st.error(WIFI_FIT_ERR)
    if WIFI_FIT is not None:
        f = WIFI_FIT
        fit_tab = pd.DataFrame({
            "Your fit": [f"{f.rssi_1m_dbm:.1f} dBm", f"{f.pl_exp:.2f}", f"{f.sigma_db:.1f} dB", f"{f.r2:.2f}",
                         f"{f.n_points}"],
            "Simulation default": [f"{DEFAULT.wifi_ptx_dbm - DEFAULT.wifi_pl0_db:g} dBm", f"{DEFAULT.wifi_pl_exp:g}",
                                   f"{DEFAULT.wifi_shadow_db:g} dB", "", ""]},
            index=["Signal at 1 m from the router", "How fast the signal fades with distance (n)",
                   "Random spread around the fit", "R² (fit quality, 1 = perfect)", "Points"])
        fl, fr = st.columns([1, 1.3], gap="large")
        fl.table(fit_tab)
        noise = -174 + 10 * np.log10(DEFAULT.b_wifi) + DEFAULT.wifi_nf_db
        fl.caption("The model is RSSI(d) = RSSI(1 m) − 10·n·log10(d). Router power and the loss over the first "
                   "metre cannot be separated from measurements alone, so their combination, the signal at 1 m, "
                   f"is fitted. Distances under 1 m count as 1 m. In this simulator Wi-Fi reaches its top speed "
                   f"({DEFAULT.max_se * DEFAULT.b_wifi / 1e6:.0f} Mbps) whenever the signal is above about "
                   f"{noise + 21:.0f} dBm.")
        if not 1.5 <= f.pl_exp <= 6:
            fl.warning(f"A fade rate n of {f.pl_exp:.2f} is outside the usual indoor range (about 1.5 to 6). "
                       "Check the router position and units.")
        fr.plotly_chart(figs.wifi_fit_plot(f.d, f.rssi, f.rssi_1m_dbm, f.pl_exp, f.sigma_db),
                        width="stretch", theme=None, key="ud_fit")
    st.checkbox("Use this Wi-Fi fit in the simulation (Live demo and results below)", value=False,
                key="ud_use_wifi", disabled=WIFI_FIT is None)

    st.subheader("2. Walking path")
    file_step("path", "t_s, x_m, y_m",
              "Time in seconds (increasing) and position in metres inside the room. "
              f"It is resampled to the {DEFAULT.dt:g} s simulation step.")
    st.checkbox("Use my path instead of the random walk", value=False, key="ud_use_path",
                disabled=UD["path"]["df"] is None)

    st.subheader("3. Blockage timeline")
    file_step("block", "t_s, blocked",
              "Time in seconds (increasing) and blocked = 1 when the light is blocked, 0 otherwise. "
              "Each value holds until the next row.")
    st.checkbox("Use my blockage timeline instead of random blockage", value=False, key="ud_use_block",
                disabled=UD["block"]["df"] is None)

    st.subheader("Results")
    used = [n for n, on in (("your Wi-Fi fit", USE_WIFI), ("your path", USE_PATH),
                            ("your blockage timeline", USE_BLOCK)) if on]
    path_arr = resample_path(UD["path"]["df"], DEFAULT.dt) if USE_PATH else None
    steps6, block_arr = None, None
    if USE_BLOCK:
        bdf = UD["block"]["df"]
        steps6 = len(path_arr) if path_arr is not None else blockage_steps(bdf, DEFAULT.dt)
        block_arr = resample_blockage(bdf, DEFAULT.dt, steps6)
    seed6 = int(st.columns([1, 3])[0].number_input("Random seed for anything still random", 0, 10_000, 42,
                                                   key="ud_seed"))
    if used:
        user_banner(", ".join(used) + ". Everything else is the default simulation.")
        if USE_BLOCK and path_arr is not None and \
                UD["block"]["df"].t_s.iloc[-1] - UD["block"]["df"].t_s.iloc[0] < (len(path_arr) - 1) * DEFAULT.dt:
            st.caption("The blockage timeline is shorter than the path, so its last value is held until the path "
                       "ends. Both start at their first timestamp.")
    else:
        st.caption("Nothing is ticked, so this is the plain simulation.")
    res6 = run_user(seed6, WIFI_OVERRIDE, path_arr, block_arr, steps6)
    if used:
        base = run_user(seed6, None, None, None, None)[3]
        cmp = pd.DataFrame({"Speed, default (Mbps)": base[SPEED].map("{:.1f}".format).values,
                            "Speed, your data (Mbps)": res6[3][SPEED].map("{:.1f}".format).values,
                            "Disconnected, default (%)": base[DISCONNECTED].map("{:.2f}".format).values,
                            "Disconnected, your data (%)": res6[3][DISCONNECTED].map("{:.2f}".format).values},
                           index=base.Scheme.values)
        st.markdown(f"Compared with the default simulation (same seed, {DEFAULT.sim_time:.0f} s random walk):")
        st.table(cmp)
    run_views(*res6, key="ud", title="Speed over time" + (" with your data" if used else ""))
