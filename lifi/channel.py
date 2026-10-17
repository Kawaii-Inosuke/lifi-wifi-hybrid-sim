"""Li-Fi line-of-sight (LOS) channel: Lambertian DC gain, received power, SNR, rate."""
from __future__ import annotations

import numpy as np

from .config import Config


def lambertian_order(half_angle: float) -> float:
    """Lambertian order m from the LED semi-angle at half power (radians)."""
    return -np.log(2) / np.log(np.cos(half_angle))


def lifi_gain(xy: np.ndarray, cfg: Config, led_pos: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """LOS DC channel gain H(0) and LED-receiver distance for receiver points xy (N, 2).

    LED points straight down and the photodiode straight up, so the irradiance angle
    equals the incidence angle. Gain is zero when the incidence angle exceeds the FOV.
    """
    xy = np.atleast_2d(xy)
    led = cfg.led_pos if led_pos is None else np.asarray(led_pos, float)
    dh = np.linalg.norm(xy - led, axis=1)
    dv = cfg.h_led - cfg.h_rx
    d = np.hypot(dh, dv)
    cos_phi = dv / d
    psi = np.arccos(cos_phi)
    m = lambertian_order(cfg.half_angle)
    g_conc = cfg.n_ref ** 2 / np.sin(cfg.fov) ** 2
    h = (m + 1) * cfg.a_pd / (2 * np.pi * d ** 2) * cos_phi ** m * cfg.t_s * g_conc * cos_phi
    return np.where(psi <= cfg.fov, h, 0.0), d


def lifi_snr(xy: np.ndarray, cfg: Config, bw: float | None = None,
             led_pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Electrical SNR (linear) and received optical power (W) at points xy.

    SNR = (R*Pr)^2 / (shot + thermal), shot = 2q(R*Pr + I_bg)B, thermal = 4kTB/R_L.
    """
    bw = cfg.b_lifi if bw is None else bw
    h, _ = lifi_gain(xy, cfg, led_pos)
    p_rx = cfg.p_t * h
    i_sig = cfg.resp * p_rx
    shot = 2 * cfg.q_e * (i_sig + cfg.i_bg) * bw
    thermal = 4 * cfg.k_b * cfg.temp * bw / cfg.r_load
    return i_sig ** 2 / (shot + thermal), p_rx


def lifi_snr_multi(xy: np.ndarray, cfg: Config, leds: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """SNR when the user attaches to the strongest of several LEDs (no interference).

    Returns (best SNR, index of serving LED).
    """
    snrs = np.stack([lifi_snr(xy, cfg, led_pos=p)[0] for p in np.atleast_2d(leds)])
    return snrs.max(axis=0), snrs.argmax(axis=0)


def to_db(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """10*log10(x) with a floor so zero maps to -120 dB instead of -inf."""
    return 10 * np.log10(np.maximum(x, floor))


def shannon_rate(snr: np.ndarray, bw: float, cfg: Config) -> np.ndarray:
    """Gap-adjusted Shannon rate in Mbps, spectral efficiency capped at cfg.max_se."""
    se = np.minimum(np.log2(1 + snr / 10 ** (cfg.gap_db / 10)), cfg.max_se)
    return se * bw / 1e6


def lifi_rate(snr: np.ndarray, cfg: Config) -> np.ndarray:
    """Li-Fi rate (Mbps). Prototype: gap-adjusted Shannon. Optional IM/DD bound (off)."""
    if cfg.lifi_imdd_capacity:
        # Lower bound for a real, non-negative, average-power-limited IM/DD channel
        # (Lapidoth et al. 2009): ~ B*log2(1 + e/(2*pi)*SNR).
        snr = snr * np.e / (2 * np.pi)
    return shannon_rate(snr, cfg.b_lifi, cfg)
