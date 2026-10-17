"""Experiments E0-E8. Each saves a CSV in <out>/data and a PNG in <out>/figures.

Every experiment function has the signature f(cfg, out, quick) -> dict (short summary).
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import plots
from .ber import ber_pam, monte_carlo_pam
from .channel import lifi_snr, lifi_snr_multi, to_db
from .config import Config
from .hybrid import (evaluate_policies, generate_trace, hysteresis_policy, metrics,
                     simulate_hybrid)
from .mobility import stationary_blockage



# ------------------------------------------------------------------ helpers
def mean_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """Sample mean and half-width of the Student-t confidence interval."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    hw = stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(hw)


def seed_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """n independent generators spawned from one master seed."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


@lru_cache(maxsize=64)
def traces(cfg: Config, n: int) -> tuple:
    """n independent traces for cfg (cached; reused across experiments for CRN)."""
    return tuple(generate_trace(cfg, r) for r in seed_rngs(cfg.seed, n))


def hyst_label(cfg: Config) -> str:
    return f"Hysteresis {cfg.thr_hi_db:g}/{cfg.thr_lo_db:g} dB"


def room_grid(cfg: Config, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.linspace(0, cfg.room, cfg.grid_n if n is None else n)
    X, Y = np.meshgrid(g, g)
    return X, Y, np.column_stack([X.ravel(), Y.ravel()])


def _csv(df: pd.DataFrame, out: Path, name: str) -> Path:
    p = Path(out) / "data" / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, float_format="%.6g")
    return p


# ------------------------------------------------------------ E0 baseline
def prototype_baseline(cfg: Config) -> tuple[dict, object, object]:
    """Reproduce the prototype's main run with one Generator(seed).

    The prototype first ran its OOK Monte Carlo (13 SNR points x 2e6 bits) on the
    same generator, so those draws are replayed first to keep the stream identical.
    """
    rng = np.random.default_rng(cfg.seed)
    monte_carlo_pam(np.arange(4, 17, 1.0), 2, 2_000_000, rng)
    return simulate_hybrid(cfg, rng)


def e0_baseline(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Prototype single-run result (seed 42) and 60 s throughput timeline."""
    res, tr, hy = prototype_baseline(cfg)
    rows = [dict(scheme=k, avg_mbps=v["avg"], outage_pct=v["outage"],
                 switches=v.get("switches", 0), lifi_share_pct=v.get("lifi_share", np.nan))
            for k, v in res.items()]
    _csv(pd.DataFrame(rows), out, "E0_baseline_seed42")

    n = int(60 / tr.dt)
    t = np.arange(n) * tr.dt
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(t, tr.r_w[:n], color=plots.COLORS["Wi-Fi only"], lw=1.6, label="Wi-Fi only")
    ax[0].plot(t, tr.r_l[:n], color=plots.COLORS["Li-Fi only"], lw=1.6, label="Li-Fi only")
    ax[0].plot(t, hy.rate[:n], color=plots.INK, lw=1.4, ls="--", label="Hybrid (15/10 dB)")
    ax[0].set_ylabel("Throughput (Mbps)")
    ax[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax[1].fill_between(t, tr.blocked[:n], step="pre", color=plots.COLORS["Li-Fi only"], alpha=0.5)
    ax[1].set_yticks([0, 1], ["no", "yes"])
    ax[1].set_ylabel("Li-Fi\nblocked")
    ax[1].set_xlabel("Time (s)")
    fig.suptitle("E0: single run, seed 42 (prototype baseline), first 60 s")
    plots.save(fig, out, "E0_baseline_timeline")
    return res


# ------------------------------------------------------------- E1 heatmaps
def e1_heatmaps(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Received optical power and SNR maps for one ceiling LED."""
    X, Y, pts = room_grid(cfg)
    snr, p = lifi_snr(pts, cfg)
    p_dbm = to_db(p / 1e-3)
    snr_db = to_db(snr)
    _csv(pd.DataFrame(dict(x_m=pts[:, 0], y_m=pts[:, 1], p_rx_dbm=p_dbm, snr_db=snr_db,
                           in_fov=snr > 0)), out, "E1_power_snr_grid")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for a, z, lab, lim in ((ax[0], p_dbm, "Received optical power (dBm)", (-30, -10)),
                           (ax[1], snr_db, f"SNR (dB), B = {cfg.b_lifi/1e6:g} MHz", (0, 30))):
        z = np.where(snr > 0, z, np.nan).reshape(X.shape)
        cm = plots.SEQ_CMAP.copy()
        cm.set_bad("#d9d9d9")
        im = a.imshow(z, origin="lower", extent=[0, cfg.room, 0, cfg.room], cmap=cm,
                      vmin=lim[0], vmax=lim[1])
        plt.colorbar(im, ax=a, label=lab)
        a.plot(*cfg.led_pos, marker="*", color="#eda100", ms=16, mec=plots.INK, ls="none")
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")
        a.grid(False)
    cs = ax[1].contour(X, Y, snr_db.reshape(X.shape), levels=[cfg.coverage_snr_db],
                       colors=plots.INK, linewidths=1.2, linestyles="--")
    ax[1].clabel(cs, fmt=f"{cfg.coverage_snr_db:g} dB", fontsize=9)
    ax[0].set_title("(a) Received optical power")
    ax[1].set_title(f"(b) SNR, dashed = {cfg.coverage_snr_db:g} dB edge")
    fig.suptitle("E1: one ceiling LED at room centre (star); grey = outside receiver FOV, no LOS signal")
    plots.save(fig, out, "E1_power_snr_heatmap")
    cov = float(100 * (snr_db >= cfg.coverage_snr_db).mean())
    return dict(coverage_pct=cov, snr_max_db=float(snr_db.max()),
                fov_pct=float(100 * (snr > 0).mean()))


# ----------------------------------------------------------- E2 BER
MC_SNR_DB = {2: np.arange(0, 15, 2.0), 4: np.arange(8, 25, 2.0), 8: np.arange(14, 31, 2.0)}


def e2_ber(cfg: Config, out: Path, quick: bool = False) -> dict:
    """BER vs distance (theory) and theory vs Monte Carlo vs SNR for OOK, 4-PAM, 8-PAM."""
    n_bits = 300_000 if quick else 3_000_000
    rng = np.random.default_rng(cfg.seed)
    r = np.linspace(0, 4.2, 300)
    line = np.column_stack([cfg.led_pos[0] + r, np.full_like(r, cfg.led_pos[1])])
    snr_line, _ = lifi_snr(line, cfg)
    dist = pd.DataFrame({"r_m": r, "snr_db": to_db(snr_line)})
    rows = []
    for M in (2, 4, 8):
        dist[f"ber_{plots.SCHEME_NAMES[M]}"] = ber_pam(snr_line, M)
        ber_mc, errs = monte_carlo_pam(MC_SNR_DB[M], M, n_bits, rng)
        for s, b, e in zip(MC_SNR_DB[M], ber_mc, errs):
            rows.append(dict(scheme=plots.SCHEME_NAMES[M], M=M, snr_db=s, n_bits=n_bits,
                             bit_errors=int(e), ber_mc=b, ber_theory=float(ber_pam(10 ** (s / 10), M))))
    mc = pd.DataFrame(rows)
    _csv(dist, out, "E2_ber_vs_distance")
    _csv(mc, out, "E2_ber_monte_carlo")

    fov_edge = np.tan(cfg.fov) * (cfg.h_led - cfg.h_rx)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for M in (2, 4, 8):
        nm, c = plots.SCHEME_NAMES[M], plots.SCHEME_COLORS[M]
        ax[0].semilogy(r, np.maximum(dist[f"ber_{nm}"], 1e-15), color=c, label=nm)
        s = np.linspace(MC_SNR_DB[M][0], MC_SNR_DB[M][-1], 200)
        ax[1].semilogy(s, ber_pam(10 ** (s / 10), M), color=c, label=f"{nm} theory")
        d = mc[(mc.M == M) & (mc.bit_errors > 0)]
        ax[1].semilogy(d.snr_db, d.ber_mc, ls="none", marker=plots.SCHEME_MARKERS[M], ms=8,
                       mfc="white", mec=c, mew=1.8, label=f"{nm} Monte Carlo")
    ax[0].axhline(1e-3, color=plots.MUTED, ls="--", lw=1)
    ax[0].axvline(fov_edge, color=plots.MUTED, ls=":", lw=1.5)
    ax[0].text(fov_edge - 0.05, 1e-10, "FOV edge", rotation=90, ha="right", color=plots.MUTED)
    ax[0].text(0.05, 1.5e-3, "BER = 1e-3", color=plots.MUTED, fontsize=9)
    ax[0].set(xlabel="Horizontal distance from LED axis (m)", ylabel="Bit error rate",
              ylim=(1e-12, 1), title=f"(a) BER vs distance, B = {cfg.b_lifi/1e6:g} MHz")
    ax[1].set(xlabel="Electrical SNR (dB)", ylabel="Bit error rate", ylim=(1e-7, 1),
              title=f"(b) Theory vs Monte Carlo ({n_bits:.0e} bits/point)")
    ax[0].legend()
    ax[1].legend(ncol=1, fontsize=8.5, loc="upper right")
    plots.save(fig, out, "E2_ber_theory_vs_mc")
    ok = mc[mc.bit_errors >= 100]
    rel = (np.abs(ok.ber_mc - ok.ber_theory) / ok.ber_theory)
    return dict(max_rel_err_100err=float(rel.max()), median_rel_err_100err=float(rel.median()),
                fov_edge_m=float(fov_edge))


# ---------------------------------------------------------- E3 sweeps
def coverage_and_corner(cfg: Config, n: int) -> tuple[float, float]:
    """Coverage % (SNR >= threshold over the room grid) and SNR (dB) at corner (0,0)."""
    _, _, pts = room_grid(cfg, n)
    snr, _ = lifi_snr(pts, cfg)
    cov = 100 * (to_db(snr) >= cfg.coverage_snr_db).mean()
    c, _ = lifi_snr(np.array([[0.0, 0.0]]), cfg)
    return float(cov), float(10 * np.log10(c[0])) if c[0] > 0 else float("nan")


E3_SWEEPS = [
    ("h_led", "LED height (m)", 1.0, np.arange(2.25, 4.01, 0.25)),
    ("half_angle_deg", "LED half-power semi-angle (deg)", 1.0, np.arange(15, 76, 5.0)),
    ("fov_deg", "Receiver FOV (deg)", 1.0, np.arange(30, 91, 5.0)),
    ("b_lifi", "Bandwidth (MHz)", 1e6, np.array([10, 20, 50, 100, 150, 200, 300.0]) * 1e6),
    ("p_t", "Transmit optical power (W)", 1.0, np.array([0.25, 0.5, 1, 2, 3, 4, 5.0])),
]


def e3_sweeps(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Coverage % and corner SNR vs LED height, semi-angle, FOV, bandwidth, power."""
    n = 61 if quick else cfg.grid_n
    rows = []
    for field, label, scale, vals in E3_SWEEPS:
        for v in vals:
            cov, corner = coverage_and_corner(cfg.with_(**{field: float(v)}), n)
            rows.append(dict(parameter=field, value=float(v) / scale, unit_label=label,
                             coverage_pct=cov, corner_snr_db=corner,
                             is_default=bool(np.isclose(v, getattr(cfg, field)))))
    df = pd.DataFrame(rows)
    _csv(df, out, "E3_parameter_sweeps")

    floor = float(np.nanmin(df.corner_snr_db)) - 3
    fig, ax = plt.subplots(2, 5, figsize=(17, 6.4), sharey="row")
    for j, (field, label, scale, _) in enumerate(E3_SWEEPS):
        d = df[df.parameter == field]
        dflt = getattr(cfg, field) / scale
        for i, col in enumerate(("coverage_pct", "corner_snr_db")):
            a = ax[i, j]
            a.plot(d.value, d[col], color=plots.COLORS["Wi-Fi only"], marker="o", ms=6)
            a.axvline(dflt, color=plots.MUTED, ls=":", lw=1.3)
            if i == 1:
                a.set_xlabel(label)
                miss = d[d[col].isna()]
                if len(miss):  # corner outside FOV: draw an x on the floor of the panel
                    a.plot(miss.value, np.full(len(miss), floor), ls="none", marker="x", ms=8,
                           color=plots.COLORS["Li-Fi only"], mew=2)
        if j == 0:
            ax[0, j].set_ylabel(f"Coverage, SNR >= {cfg.coverage_snr_db:g} dB (%)")
            ax[1, j].set_ylabel("SNR at room corner (dB)")
    ax[1, 0].plot([], [], ls="none", marker="x", color=plots.COLORS["Li-Fi only"], mew=2,
                  label="corner outside FOV\n(no LOS signal)")
    ax[1, 0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("E3: single-LED parameter sweeps (dotted line = default value)", y=1.0)
    fig.tight_layout()
    plots.save(fig, out, "E3_parameter_sweeps")
    best = {f: df[df.parameter == f].sort_values("coverage_pct").iloc[-1][["value", "coverage_pct"]].tolist()
            for f, *_ in E3_SWEEPS}
    return dict(best=best)


# --------------------------------------------------------- E4 blockage
E4_P_IN = np.array([0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1])


def _three_policy_stats(trs, cfg: Config) -> dict[str, dict[str, tuple[float, float]]]:
    acc: dict[str, dict[str, list]] = {}
    for tr in trs:
        hy = hysteresis_policy(tr, cfg.thr_hi_db, cfg.thr_lo_db, cfg.switch_cost_steps)
        for name, r, sw in (("Wi-Fi only", tr.r_w, 0), ("Li-Fi only", tr.r_l, 0),
                            ("Hybrid", hy.rate, hy.switches)):
            m = metrics(r, cfg)
            m["switches"] = sw
            for k, v in m.items():
                acc.setdefault(name, {}).setdefault(k, []).append(v)
    return {p: {k: mean_ci(v) for k, v in d.items()} for p, d in acc.items()}


def e4_blockage(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Throughput and outage vs blockage probability for Wi-Fi only, Li-Fi only, Hybrid."""
    n = 10 if quick else cfg.n_seeds
    rows = []
    for p in E4_P_IN:
        c = cfg.with_(p_block_in=float(p))
        trs = traces(c, n)
        meas = float(np.mean([t.blocked.mean() for t in trs]))
        for pol, d in _three_policy_stats(trs, c).items():
            rows.append(dict(p_block_in=p, p_block_out=c.p_block_out,
                             blocked_frac_theory=stationary_blockage(p, c.p_block_out),
                             blocked_frac_measured=meas, policy=pol, n_seeds=n,
                             avg_mbps=d["avg"][0], avg_ci95=d["avg"][1],
                             outage_pct=d["outage"][0], outage_ci95=d["outage"][1],
                             switches=d["switches"][0], switches_ci95=d["switches"][1]))
    df = pd.DataFrame(rows)
    _csv(df, out, "E4_blockage_sweep")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for pol in ("Wi-Fi only", "Li-Fi only", "Hybrid"):
        d = df[df.policy == pol]
        x = 100 * d.blocked_frac_theory
        kw = dict(color=plots.COLORS[pol], marker=plots.MARKERS[pol], capsize=3, label=pol)
        ax[0].errorbar(x, d.avg_mbps, yerr=d.avg_ci95, **kw)
        ax[1].errorbar(x, d.outage_pct, yerr=d.outage_ci95, **kw)
    dflt = 100 * stationary_blockage(cfg.p_block_in, cfg.p_block_out)
    for a in ax:
        a.axvline(dflt, color=plots.MUTED, ls=":", lw=1.3)
        a.set_xlabel(f"Long-run fraction of time Li-Fi is blocked (%)\n"
                     f"[p_in varied, p_out = {cfg.p_block_out:g}]")
    ax[0].set_ylabel("Average throughput (Mbps)")
    ax[1].set_ylabel(f"Outage, rate < {cfg.outage_mbps:g} Mbps (%)")
    ax[0].set_title(f"(a) Throughput (mean +/- 95% CI, {n} seeds)")
    ax[1].set_title("(b) Outage")
    ax[0].legend()
    plots.save(fig, out, "E4_blockage_sweep")
    return dict(df=df)


# --------------------------------------------------------- E5 hysteresis
def e5_hysteresis(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Grid over (up, down) thresholds: throughput, outage, switches."""
    n = 5 if quick else cfg.n_seeds
    trs = traces(cfg, n)
    ups = np.arange(10, 26, 1.0)
    downs = np.arange(5, 21, 1.0)
    rows = []
    for up in ups:
        for dn in downs:
            if dn >= up:
                continue
            res = [hysteresis_policy(t, up, dn, cfg.switch_cost_steps) for t in trs]
            ms = [metrics(r.rate, cfg) for r in res]
            rows.append(dict(thr_up_db=up, thr_down_db=dn, n_seeds=n,
                             avg_mbps=np.mean([m["avg"] for m in ms]),
                             outage_pct=np.mean([m["outage"] for m in ms]),
                             switches=np.mean([r.switches for r in res])))
    df = pd.DataFrame(rows)
    _csv(df, out, "E5_hysteresis_sweep")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    specs = (("avg_mbps", "Avg throughput (Mbps)"), ("outage_pct", "Outage (%)"),
             ("switches", "Switches per 300 s run"))
    for a, (col, lab) in zip(ax, specs):
        Z = df.pivot(index="thr_down_db", columns="thr_up_db", values=col).reindex(
            index=downs, columns=ups)
        cm = plots.SEQ_CMAP.copy()
        cm.set_bad("white")
        im = a.imshow(Z.values, origin="lower", cmap=cm, aspect="auto",
                      extent=[ups[0] - .5, ups[-1] + .5, downs[0] - .5, downs[-1] + .5])
        plt.colorbar(im, ax=a, label=lab)
        a.plot(cfg.thr_hi_db, cfg.thr_lo_db, marker="*", ms=18, color="#eda100",
               mec=plots.INK, ls="none", label=f"chosen {cfg.thr_hi_db:g}/{cfg.thr_lo_db:g} dB")
        a.set(xlabel="Up threshold, Wi-Fi -> Li-Fi (dB)",
              ylabel="Down threshold, Li-Fi -> Wi-Fi (dB)", title=lab)
        a.grid(False)
    ax[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"E5: hysteresis threshold sweep (mean of {n} seeds; white = invalid, down >= up)")
    fig.tight_layout()
    plots.save(fig, out, "E5_hysteresis_sweep")
    return dict(df=df)


# ------------------------------------------------ E6 / E7 multi-seed stats
def _policy_table(cfg: Config, n: int) -> pd.DataFrame:
    rows = []
    for i, tr in enumerate(traces(cfg, n)):
        for pol, m in evaluate_policies(tr, cfg).items():
            rows.append(dict(seed_idx=i, policy=pol, **m))
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for pol in order:
        d = df[df.policy == pol]
        r = dict(policy=pol, n_seeds=len(d))
        for k in ("avg", "outage", "switches", "lifi_share"):
            r[f"{k}_mean"], r[f"{k}_ci95"] = mean_ci(d[k].values)
        rows.append(r)
    return pd.DataFrame(rows)


def _bar_panel(ax, summ: pd.DataFrame, cols, labels):
    x = np.arange(len(summ))
    for a, col, lab in zip(ax, cols, labels):
        for i, (_, r) in enumerate(summ.iterrows()):
            c = plots.COLORS[plots.policy_key(r.policy)]
            a.bar(i, r[f"{col}_mean"], width=0.65, color=c, edgecolor="white", linewidth=2)
            a.errorbar(i, r[f"{col}_mean"], yerr=r[f"{col}_ci95"], color=plots.INK, capsize=4, lw=1.4)
            top = r[f"{col}_mean"] + (0 if np.isnan(r[f"{col}_ci95"]) else r[f"{col}_ci95"])
            a.annotate(f"{r[f'{col}_mean']:.1f}", (i, top), xytext=(0, 4), textcoords="offset points",
                       ha="center", va="bottom", fontsize=9, color=plots.INK)
        a.set_xticks(x, [p.replace(" (upper bound)", "\n(upper bound)").replace("Single thr. ", "Single\n")
                         .replace("Hysteresis ", "Hysteresis\n") for p in summ.policy], fontsize=9)
        a.set_ylabel(lab)
        a.margins(y=0.15)
        a.grid(axis="x", visible=False)


def e6_statistics(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Main comparison over cfg.n_seeds independent seeds, mean and 95% CI."""
    n = cfg.n_seeds
    df = _policy_table(cfg, n)
    order = ["Wi-Fi only", "Li-Fi only", hyst_label(cfg)]
    df = df[df.policy.isin(order)]
    summ = _summarise(df, order)
    _csv(df, out, "E6_per_seed")
    _csv(summ, out, "E6_statistics_summary")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    _bar_panel(ax, summ, ("avg", "outage", "switches"),
               ("Average throughput (Mbps)", f"Outage, rate < {cfg.outage_mbps:g} Mbps (%)",
                "Link switches per 300 s run"))
    rng = np.random.default_rng(0)  # jitter only, does not affect results
    for a, col in zip(ax, ("avg", "outage", "switches")):
        for i, pol in enumerate(order):
            v = df[df.policy == pol][col].values
            a.plot(i + 0.45 + rng.uniform(-0.05, 0.05, len(v)), v, ls="none", marker="o", ms=3.5,
                   mfc=plots.INK, mec="none", alpha=0.45)
    fig.suptitle(f"E6: {n} independent seeds, bars = mean, whiskers = 95% CI, dots = individual runs")
    fig.tight_layout()
    plots.save(fig, out, "E6_statistics")
    return dict(summary=summ)


def e7_policies(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Wi-Fi only, Li-Fi only, single-threshold hybrids, hysteresis hybrid, ideal bound."""
    n = cfg.n_seeds
    df = _policy_table(cfg, n)
    order = list(dict.fromkeys(df.policy))
    summ = _summarise(df, order)
    _csv(df, out, "E7_per_seed")
    _csv(summ, out, "E7_policy_comparison")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    _bar_panel(ax, summ, ("avg", "outage", "switches"),
               ("Average throughput (Mbps)", f"Outage, rate < {cfg.outage_mbps:g} Mbps (%)",
                "Link switches per 300 s run"))
    fig.suptitle(f"E7: policy comparison ({n} seeds, mean +/- 95% CI). "
                 "Ideal = best link every step, zero switching cost")
    fig.tight_layout()
    plots.save(fig, out, "E7_policy_comparison")
    return dict(summary=summ)


# ------------------------------------------------------- E8 LED grid
def e8_led_grid(cfg: Config, out: Path, quick: bool = False) -> dict:
    """2x2 LED grid, strongest-LED association, no inter-cell interference."""
    X, Y, pts = room_grid(cfg)
    q = cfg.room / 4
    leds = np.array([[q, q], [3 * q, q], [q, 3 * q], [3 * q, 3 * q]])
    cases = [("1 LED, 1 W (baseline)", cfg, np.atleast_2d(cfg.led_pos)),
             ("2x2 LEDs, 1 W each (4 W total)", cfg, leds),
             ("2x2 LEDs, 0.25 W each (1 W total)", cfg.with_(p_t=cfg.p_t / 4), leds)]
    rows, maps = [], []
    for name, c, L in cases:
        snr, _ = lifi_snr_multi(pts, c, L)
        sdb = to_db(snr)
        maps.append((name, sdb, L))
        rows.append(dict(case=name, n_leds=len(L), p_per_led_w=c.p_t, p_total_w=c.p_t * len(L),
                         coverage_pct=100 * (sdb >= c.coverage_snr_db).mean(),
                         any_signal_pct=100 * (snr > 0).mean(),
                         snr_median_db=float(np.median(sdb[snr > 0])),
                         snr_min_in_room_db=float(sdb.min()) if (snr > 0).all() else float("nan")))
    df = pd.DataFrame(rows)
    _csv(df, out, "E8_led_grid_coverage")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cm = plots.SEQ_CMAP.copy()
    cm.set_bad("#d9d9d9")
    for a, (name, sdb, L), r in zip(ax, maps, rows):
        z = np.where(sdb > -100, sdb, np.nan).reshape(X.shape)
        im = a.imshow(z, origin="lower", extent=[0, cfg.room, 0, cfg.room], cmap=cm, vmin=0, vmax=30)
        a.contour(X, Y, sdb.reshape(X.shape), levels=[cfg.coverage_snr_db], colors=plots.INK,
                  linewidths=1.2, linestyles="--")
        a.plot(L[:, 0], L[:, 1], marker="*", color="#eda100", ms=14, mec=plots.INK, ls="none")
        a.set(title=f"{name}\ncoverage {r['coverage_pct']:.1f}%", xlabel="x (m)", ylabel="y (m)")
        a.grid(False)
    fig.colorbar(im, ax=ax, label="SNR of strongest LED (dB)", shrink=0.9)
    fig.suptitle(f"E8: coverage (dashed = {cfg.coverage_snr_db:g} dB). "
                 "No inter-cell interference modelled (optimistic)", y=1.02)
    plots.save(fig, out, "E8_led_grid_coverage")
    return dict(df=df)


# --------------------------------------------- S1 known-issue fix flags
S1_VARIANTS = [("Prototype (all fixes off)", {}),
               ("IM/DD capacity bound for Li-Fi", dict(lifi_imdd_capacity=True)),
               ("Correlated Wi-Fi shadowing", dict(wifi_correlated_shadowing=True)),
               ("Wi-Fi MAC efficiency 0.6", dict(wifi_mac_overhead=True))]


def s1_fix_flags(cfg: Config, out: Path, quick: bool = False) -> dict:
    """Headline comparison re-run with each optional fix switched on (one at a time)."""
    rows = []
    for name, kw in S1_VARIANTS:
        c = cfg.with_(**kw)
        summ = _summarise(_policy_table(c, cfg.n_seeds),
                          ["Wi-Fi only", "Li-Fi only", hyst_label(c), "Ideal (upper bound)"])
        rows += [dict(variant=name, **r) for r in summ.to_dict("records")]
    df = pd.DataFrame(rows)
    _csv(df, out, "S1_fix_flag_sensitivity")
    return dict(df=df)


EXPERIMENTS = {"E0": e0_baseline, "E1": e1_heatmaps, "E2": e2_ber, "E3": e3_sweeps,
               "E4": e4_blockage, "E5": e5_hysteresis, "E6": e6_statistics,
               "E7": e7_policies, "E8": e8_led_grid,
               "S1": s1_fix_flags}


def run(names: list[str], cfg: Config, out: Path, quick: bool = False) -> dict:
    """Run experiments by name, printing timing. Returns their summaries."""
    plots.setup_style()
    out = Path(out)
    res = {}
    for nm in names:
        t0 = time.perf_counter()
        res[nm] = EXPERIMENTS[nm](cfg, out, quick)
        print(f"[{nm}] done in {time.perf_counter() - t0:5.1f} s")
    return res
