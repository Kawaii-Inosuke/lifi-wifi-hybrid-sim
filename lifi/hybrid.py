"""Hybrid Li-Fi / Wi-Fi link selection: traces, policies and metrics.

A *trace* is one random realisation (user path, blockage, shadowing) with the
per-step rates of both links. Policies are then applied to the same trace, so
different policies are compared on identical conditions (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import lifi_rate, lifi_snr, to_db
from .config import Config
from .mobility import markov_blockage, random_waypoint
from .wifi import shadowing, wifi_rate, wifi_snr


@dataclass
class Trace:
    path: np.ndarray        # (steps, 2) user position, m
    blocked: np.ndarray     # (steps,) bool, Li-Fi LOS blocked
    snr_l_db: np.ndarray    # (steps,) Li-Fi SNR in dB (blocked/outside FOV -> -120 dB)
    r_l: np.ndarray         # (steps,) Li-Fi rate, Mbps
    r_w: np.ndarray         # (steps,) Wi-Fi rate, Mbps
    dt: float               # s


@dataclass
class PolicyResult:
    rate: np.ndarray        # (steps,) delivered rate, Mbps
    use_lifi: np.ndarray    # (steps,) bool, Li-Fi selected
    switches: int


def generate_trace(cfg: Config, rng: np.random.Generator) -> Trace:
    """Draw one realisation. Draw order: path, blockage, shadowing (as the prototype)."""
    steps = cfg.steps
    path = random_waypoint(cfg, rng, steps)
    blocked = markov_blockage(steps, cfg.p_block_in, cfg.p_block_out, rng)
    shadow = shadowing(path, cfg, rng)
    snr_l, _ = lifi_snr(path, cfg)
    snr_l = np.where(blocked, 0.0, snr_l)
    return Trace(path=path, blocked=blocked, snr_l_db=to_db(snr_l),
                 r_l=lifi_rate(snr_l, cfg), r_w=wifi_rate(wifi_snr(path, cfg, shadow), cfg),
                 dt=cfg.dt)


def hysteresis_policy(tr: Trace, thr_hi_db: float, thr_lo_db: float,
                      switch_cost_steps: int) -> PolicyResult:
    """Start on Wi-Fi; go to Li-Fi when SNR > thr_hi, back to Wi-Fi when SNR < thr_lo.

    Each switch gives zero throughput for switch_cost_steps steps (including the
    switching step). thr_hi == thr_lo gives a single-threshold policy.
    """
    snr = tr.snr_l_db
    n = len(snr)
    use_lifi = np.zeros(n, bool)
    rate = np.zeros(n)
    state, penalty, switches = False, 0, 0
    for i in range(n):
        new = state
        if not state and snr[i] > thr_hi_db:
            new = True
        elif state and snr[i] < thr_lo_db:
            new = False
        if new != state:
            switches += 1
            penalty = switch_cost_steps
        state = new
        use_lifi[i] = state
        rate[i] = 0.0 if penalty > 0 else (tr.r_l[i] if state else tr.r_w[i])
        penalty = max(penalty - 1, 0)
    return PolicyResult(rate, use_lifi, switches)


def single_threshold_policy(tr: Trace, thr_db: float, switch_cost_steps: int) -> PolicyResult:
    """No hysteresis: Li-Fi if SNR > thr, Wi-Fi if SNR < thr."""
    return hysteresis_policy(tr, thr_db, thr_db, switch_cost_steps)


def ideal_policy(tr: Trace) -> PolicyResult:
    """Upper bound: pick the faster link every step with zero switching cost."""
    use = tr.r_l >= tr.r_w
    return PolicyResult(np.where(use, tr.r_l, tr.r_w), use, int(np.count_nonzero(np.diff(use))))


def metrics(rate: np.ndarray, cfg: Config) -> dict[str, float]:
    """Average throughput (Mbps) and outage (% of steps with rate < cfg.outage_mbps)."""
    return dict(avg=float(rate.mean()), outage=float(100 * (rate < cfg.outage_mbps).mean()))


def simulate_hybrid(cfg: Config, rng: np.random.Generator) -> tuple[dict, Trace, PolicyResult]:
    """Prototype comparison: Wi-Fi only, Li-Fi only, hysteresis hybrid on one trace."""
    tr = generate_trace(cfg, rng)
    hy = hysteresis_policy(tr, cfg.thr_hi_db, cfg.thr_lo_db, cfg.switch_cost_steps)
    res = {"Wi-Fi only": metrics(tr.r_w, cfg), "Li-Fi only": metrics(tr.r_l, cfg),
           "Hybrid": metrics(hy.rate, cfg)}
    res["Hybrid"]["switches"] = hy.switches
    res["Hybrid"]["lifi_share"] = float(100 * hy.use_lifi.mean())
    return res, tr, hy


def evaluate_policies(tr: Trace, cfg: Config) -> dict[str, dict[str, float]]:
    """All policies of E7 on one trace."""
    pols = {
        "Wi-Fi only": PolicyResult(tr.r_w, np.zeros(len(tr.r_w), bool), 0),
        "Li-Fi only": PolicyResult(tr.r_l, np.ones(len(tr.r_l), bool), 0),
        f"Single thr. {cfg.thr_lo_db:g} dB": single_threshold_policy(tr, cfg.thr_lo_db, cfg.switch_cost_steps),
        f"Single thr. {cfg.thr_hi_db:g} dB": single_threshold_policy(tr, cfg.thr_hi_db, cfg.switch_cost_steps),
        f"Hysteresis {cfg.thr_hi_db:g}/{cfg.thr_lo_db:g} dB": hysteresis_policy(
            tr, cfg.thr_hi_db, cfg.thr_lo_db, cfg.switch_cost_steps),
        "Ideal (upper bound)": ideal_policy(tr),
    }
    out = {}
    for name, p in pols.items():
        m = metrics(p.rate, cfg)
        m["switches"] = p.switches
        m["lifi_share"] = float(100 * p.use_lifi.mean())
        out[name] = m
    return out
