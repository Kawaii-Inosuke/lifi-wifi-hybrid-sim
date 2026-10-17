"""User-provided data for the demo: validation, Wi-Fi model fitting, custom traces.

New helpers only; nothing here changes the behaviour of the existing modules.
With no overrides, build_trace() returns exactly what hybrid.generate_trace() returns.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .channel import lifi_rate, lifi_snr, to_db
from .config import Config
from .hybrid import PolicyResult, Trace
from .mobility import markov_blockage, random_waypoint
from .wifi import shadowing, wifi_rate, wifi_snr

MIN_WIFI_POINTS = 6
MAX_DURATION_S = 1800.0          # longest user path / blockage timeline accepted (assumption)


class DataError(ValueError):
    """A user file is invalid. The message is meant to be shown to the user as-is."""


# ------------------------------------------------------------------ reading
def read_csv(data: bytes | str | io.IOBase, required: list[str], name: str) -> pd.DataFrame:
    """Read a CSV and check required numeric columns. Raises DataError with a friendly message."""
    try:
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        df = pd.read_csv(data, skipinitialspace=True)
    except Exception as e:  # noqa: BLE001 - any parse failure becomes a friendly error
        raise DataError(f"{name}: could not read the file as CSV ({type(e).__name__}). "
                        "Save it as comma-separated text with a header row.") from None
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{name}: missing column(s) {', '.join(missing)}. "
                        f"Expected header: {','.join(required)}. Found: {','.join(df.columns) or 'nothing'}.")
    df = df[required].copy()
    if df.empty:
        raise DataError(f"{name}: the file has a header but no data rows.")
    for c in required:
        num = pd.to_numeric(df[c], errors="coerce")
        bad = num.isna() & df[c].notna()
        if bad.any():
            row = int(np.flatnonzero(bad.values)[0]) + 2          # +2: header line, 1-based
            raise DataError(f"{name}: column '{c}' has a non-numeric value "
                            f"('{df[c].iloc[row - 2]}') on line {row}.")
        if num.isna().any():
            row = int(np.flatnonzero(num.isna().values)[0]) + 2
            raise DataError(f"{name}: column '{c}' is empty on line {row}.")
        df[c] = num.astype(float)
    return df


def _check_inside_room(x: np.ndarray, y: np.ndarray, room: float, name: str) -> None:
    out = (x < 0) | (x > room) | (y < 0) | (y > room)
    if out.any():
        i = int(np.flatnonzero(out)[0])
        hint = (" The values look like centimetres; please convert to metres."
                if max(np.abs(x).max(), np.abs(y).max()) > 10 * room else "")
        raise DataError(f"{name}: {int(out.sum())} position(s) are outside the {room:g} x {room:g} m room "
                        f"(first: line {i + 2}, x={x[i]:g}, y={y[i]:g}). "
                        f"x_m and y_m must be between 0 and {room:g} metres.{hint}")


def _check_time(t: np.ndarray, name: str) -> None:
    if len(t) < 2:
        raise DataError(f"{name}: need at least 2 rows.")
    if np.any(np.diff(t) <= 0):
        i = int(np.flatnonzero(np.diff(t) <= 0)[0])
        raise DataError(f"{name}: time t_s must be strictly increasing "
                        f"(line {i + 3}: {t[i + 1]:g} s comes after {t[i]:g} s).")
    if t[-1] - t[0] > MAX_DURATION_S:
        hint = " Is t_s in milliseconds? It must be in seconds." if t[-1] - t[0] > 1000 * 60 else ""
        raise DataError(f"{name}: covers {t[-1] - t[0]:.0f} s; the demo accepts at most "
                        f"{MAX_DURATION_S:.0f} s.{hint}")


# --------------------------------------------------------------- validators
def parse_wifi(data, room: float) -> pd.DataFrame:
    """Wi-Fi survey CSV: x_m, y_m, rssi_dbm. At least 6 points, RSSI in a plausible dBm range."""
    name = "Wi-Fi measurements"
    df = read_csv(data, ["x_m", "y_m", "rssi_dbm"], name)
    if len(df) < MIN_WIFI_POINTS:
        raise DataError(f"{name}: only {len(df)} point(s); at least {MIN_WIFI_POINTS} are needed for a fit.")
    _check_inside_room(df.x_m.values, df.y_m.values, room, name)
    r = df.rssi_dbm.values
    if np.any(r > 0):
        raise DataError(f"{name}: rssi_dbm has positive values (max {r.max():g}). RSSI must be in dBm, "
                        "which is negative for Wi-Fi (typically -30 to -90). Percent or 'bars' are not dBm.")
    if np.any(r < -120) or np.any(r > -10):
        raise DataError(f"{name}: rssi_dbm must be between -120 and -10 dBm (found {r.min():g} to {r.max():g}).")
    return df


def parse_path(data, room: float) -> pd.DataFrame:
    """User path CSV: t_s, x_m, y_m inside the room with increasing time."""
    name = "User path"
    df = read_csv(data, ["t_s", "x_m", "y_m"], name)
    _check_time(df.t_s.values, name)
    _check_inside_room(df.x_m.values, df.y_m.values, room, name)
    return df


def parse_blockage(data) -> pd.DataFrame:
    """Blockage CSV: t_s, blocked (0 or 1) with increasing time."""
    name = "Blockage timeline"
    df = read_csv(data, ["t_s", "blocked"], name)
    _check_time(df.t_s.values, name)
    b = df.blocked.values
    if not np.all(np.isin(b, (0.0, 1.0))):
        raise DataError(f"{name}: 'blocked' must be 0 or 1 (found {sorted(set(b[~np.isin(b, (0.0, 1.0))]))[:3]}).")
    return df


def path_warnings(df: pd.DataFrame) -> list[str]:
    """Soft checks that do not block use (e.g. implausible walking speed)."""
    w = []
    v = np.hypot(np.diff(df.x_m), np.diff(df.y_m)) / np.diff(df.t_s)
    if len(v) and v.max() > 3.0:
        w.append(f"Fastest segment is {v.max():.1f} m/s (faster than running). Check units: time in s, positions in m.")
    return w


# ----------------------------------------------------------------- resample
def resample_path(df: pd.DataFrame, dt: float) -> np.ndarray:
    """Linear interpolation of (x, y) onto a uniform grid t0, t0+dt, ... <= t_end."""
    t = df.t_s.values
    tt = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1) * dt
    return np.column_stack([np.interp(tt, t, df.x_m.values), np.interp(tt, t, df.y_m.values)])


def resample_blockage(df: pd.DataFrame, dt: float, steps: int) -> np.ndarray:
    """Zero-order hold onto steps samples starting at the first timestamp.

    Each sample takes the most recent value at or before its time; samples after the
    last timestamp keep the last value.
    """
    t = df.t_s.values
    tt = t[0] + np.arange(steps) * dt
    idx = np.searchsorted(t, tt + 1e-9, side="right") - 1
    return df.blocked.values[np.clip(idx, 0, len(t) - 1)].astype(bool)


def blockage_steps(df: pd.DataFrame, dt: float) -> int:
    t = df.t_s.values
    return int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1


# -------------------------------------------------------------- Wi-Fi fit
@dataclass
class WifiFit:
    rssi_1m_dbm: float       # combined Ptx - PL(1 m), dBm
    pl_exp: float            # path-loss exponent n
    sigma_db: float          # residual std (shadowing estimate), dB
    r2: float
    n_points: int
    d: np.ndarray            # distances used (clamped to d_min), m
    rssi: np.ndarray         # measured RSSI, dBm

    def apply(self, cfg: Config, ap_xy: tuple[float, float]) -> Config:
        """Config whose Wi-Fi model is this fit; all other parameters unchanged.

        wifi_snr uses Prx = Ptx - PL0 - 10 n log10(d) - X, so setting Ptx = RSSI(1 m)
        and PL0 = 0 reproduces the fitted line exactly.
        """
        return cfg.with_(wifi_xy=(float(ap_xy[0]), float(ap_xy[1])), wifi_ptx_dbm=self.rssi_1m_dbm,
                         wifi_pl0_db=0.0, wifi_pl_exp=self.pl_exp, wifi_shadow_db=self.sigma_db)


def fit_wifi(df: pd.DataFrame, ap_xy: tuple[float, float], d_min: float = 1.0) -> WifiFit:
    """Least-squares fit of RSSI = A - 10 n log10(d) (log-distance model).

    d is clamped at d_min, as in wifi.wifi_snr. Ptx and PL(1 m) cannot be separated from
    RSSI alone, so only their combination A = RSSI at 1 m is estimated.
    """
    d = np.maximum(np.hypot(df.x_m.values - ap_xy[0], df.y_m.values - ap_xy[1]), d_min)
    r = df.rssi_dbm.values.astype(float)
    if len(r) < MIN_WIFI_POINTS:
        raise DataError(f"Need at least {MIN_WIFI_POINTS} points to fit (have {len(r)}).")
    xl = np.log10(d)
    if np.ptp(xl) < 0.1:
        raise DataError("All measurement points are at almost the same distance from the AP "
                        "(or closer than 1 m), so the path-loss exponent cannot be estimated. "
                        "Measure at a wider range of distances, or check the AP position.")
    X = np.column_stack([np.ones_like(xl), -10 * xl])
    (a, n), *_ = np.linalg.lstsq(X, r, rcond=None)
    res = r - X @ np.array([a, n])
    dof = max(len(r) - 2, 1)
    ss_tot = float(((r - r.mean()) ** 2).sum())
    r2 = 1 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return WifiFit(float(a), float(n), float(np.sqrt((res ** 2).sum() / dof)), r2, len(r), d, r)


# ------------------------------------------------------------ custom trace
def build_trace(cfg: Config, rng: np.random.Generator, path: np.ndarray | None = None,
                blocked: np.ndarray | None = None, steps: int | None = None) -> Trace:
    """Like hybrid.generate_trace, with optional user path and/or blockage.

    Draw order for the random parts is unchanged (path, blockage, shadowing), so with no
    overrides and steps=None the result is identical to generate_trace(cfg, rng).
    """
    if path is not None:
        steps = len(path)
    elif blocked is not None and steps is None:
        steps = len(blocked)
    steps = cfg.steps if steps is None else steps
    if path is None:
        path = random_waypoint(cfg, rng, steps)
    if blocked is None:
        blocked = markov_blockage(steps, cfg.p_block_in, cfg.p_block_out, rng)
    elif len(blocked) != steps:
        raise ValueError(f"blocked has {len(blocked)} samples, expected {steps}; resample it first")
    else:
        blocked = np.asarray(blocked, bool)
    shadow = shadowing(path, cfg, rng)
    snr_l, _ = lifi_snr(path, cfg)
    snr_l = np.where(blocked, 0.0, snr_l)
    return Trace(path=path, blocked=blocked, snr_l_db=to_db(snr_l), r_l=lifi_rate(snr_l, cfg),
                 r_w=wifi_rate(wifi_snr(path, cfg, shadow), cfg), dt=cfg.dt)


def switching_mask(res: PolicyResult, switch_cost_steps: int) -> np.ndarray:
    """Steps that fall inside a handover gap (same rule as hybrid.hysteresis_policy)."""
    n = len(res.use_lifi)
    prev = np.r_[False, res.use_lifi[:-1]]
    mask = np.zeros(n, bool)
    for i in np.flatnonzero(res.use_lifi != prev):
        mask[i:i + switch_cost_steps] = True
    return mask
