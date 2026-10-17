"""Simplified Wi-Fi link: log-distance path loss + log-normal shadowing."""
from __future__ import annotations

import numpy as np

from .channel import shannon_rate
from .config import Config


def wifi_snr(xy: np.ndarray, cfg: Config, shadow_db: np.ndarray | float = 0.0) -> np.ndarray:
    """Wi-Fi SNR (linear) at points xy given shadowing (dB, added to path loss)."""
    d = np.maximum(np.linalg.norm(np.atleast_2d(xy) - cfg.wifi_pos, axis=1), cfg.wifi_d_min)
    pl = cfg.wifi_pl0_db + 10 * cfg.wifi_pl_exp * np.log10(d) + shadow_db
    p_rx_dbm = cfg.wifi_ptx_dbm - pl
    noise_dbm = -174 + 10 * np.log10(cfg.b_wifi) + cfg.wifi_nf_db
    return 10 ** ((p_rx_dbm - noise_dbm) / 10)


def shadowing(path: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Shadowing sample (dB) per time step.

    Prototype: i.i.d. N(0, sigma) per step (one rng.normal call of length steps).
    Optional fix: AR(1) correlated over walked distance (Gudmundson model).
    """
    n = len(path)
    w = rng.normal(0, cfg.wifi_shadow_db, n)
    if not cfg.wifi_correlated_shadowing:
        return w
    step = np.r_[0.0, np.linalg.norm(np.diff(path, axis=0), axis=1)]
    rho = np.exp(-step / cfg.wifi_shadow_decorr_m)
    out = np.empty(n)
    out[0] = w[0]
    for i in range(1, n):
        out[i] = rho[i] * out[i - 1] + np.sqrt(1 - rho[i] ** 2) * w[i]
    return out


def wifi_rate(snr: np.ndarray, cfg: Config) -> np.ndarray:
    """Wi-Fi rate (Mbps); optional MAC-efficiency factor (off by default)."""
    r = shannon_rate(snr, cfg.b_wifi, cfg)
    return r * cfg.wifi_mac_efficiency if cfg.wifi_mac_overhead else r
