"""All simulation parameters in one dataclass.

Every field states its unit and where the value comes from. "assumption" means the
value was chosen for this project, not taken from a specific paper. The defaults
reproduce the original prototype (lifi_sim.py) exactly.

References used below:
  [K04] T. Komine, M. Nakagawa, "Fundamental analysis for visible-light communication
        system using LED lights", IEEE Trans. Consumer Electronics, 50(1), 2004.
  [G19] Z. Ghassemlooy, W. Popoola, S. Rajbhandari, "Optical Wireless Communications:
        System and Channel Modelling with MATLAB", 2nd ed., CRC Press, 2019.
  [R02] T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., 2002.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------- geometry
    room: float = 6.0            # m, side of square room (assumption)
    h_led: float = 3.0           # m, ceiling/LED height (assumption; [K04] uses 3 m room height)
    h_rx: float = 0.85           # m, receiver (desk) height ([K04] receiver plane 0.85 m)
    led_xy: tuple[float, float] | None = None  # m, LED (x, y); None = room centre (assumption)

    # ------------------------------------------------------- Li-Fi transmitter
    p_t: float = 1.0             # W, transmitted optical power per LED (assumption)
    half_angle_deg: float = 60.0  # deg, LED semi-angle at half power (typical value [G19]; assumption)

    # ---------------------------------------------------------- Li-Fi receiver
    a_pd: float = 1e-4           # m^2, photodiode area, 1 cm^2 ([K04])
    resp: float = 0.54           # A/W, PD responsivity (typical Si PIN; [K04] uses 0.53)
    fov_deg: float = 60.0        # deg, receiver field of view (assumption)
    n_ref: float = 1.5           # -, concentrator refractive index ([K04])
    t_s: float = 1.0             # -, optical filter gain (assumption: ideal filter)

    # ------------------------------------------------------------- Li-Fi noise
    i_bg: float = 5.1e-3         # A, ambient-light photocurrent ([K04]: 5100 uA)
    r_load: float = 50.0         # ohm, load resistance (assumption)
    temp: float = 300.0          # K, receiver temperature (assumption: room temperature)
    q_e: float = 1.602e-19       # C, electron charge (physical constant)
    k_b: float = 1.381e-23       # J/K, Boltzmann constant (physical constant)

    # ------------------------------------------------------------ link rates
    b_lifi: float = 100e6        # Hz, Li-Fi modulation bandwidth (assumption)
    b_wifi: float = 20e6         # Hz, Wi-Fi channel bandwidth (IEEE 802.11 20 MHz channel)
    max_se: float = 6.0          # bit/s/Hz, spectral-efficiency cap, both links (assumption, ~64-QAM r=1)
    gap_db: float = 3.0          # dB, SNR gap to Shannon for practical coding (assumption)

    # ------------------------------------------------------------------ Wi-Fi
    wifi_xy: tuple[float, float] = (0.0, 0.0)  # m, AP position: room corner (assumption)
    wifi_ptx_dbm: float = 20.0   # dBm, AP transmit power (typical 2.4 GHz limit; assumption)
    wifi_pl0_db: float = 40.0    # dB, path loss at 1 m (free-space at 2.4 GHz is ~40 dB)
    wifi_pl_exp: float = 3.0     # -, path-loss exponent (indoor range 2-4 [R02]; assumption)
    wifi_d_min: float = 1.0      # m, minimum distance used in path loss (assumption)
    wifi_nf_db: float = 7.0      # dB, receiver noise figure (assumption)
    wifi_shadow_db: float = 4.0  # dB, log-normal shadowing std (indoor 3-6 dB typical; assumption)

    # ------------------------------------------------------ mobility/blockage
    sim_time: float = 300.0      # s, simulated time per run (assumption)
    dt: float = 0.1              # s, time step (assumption)
    speed: float = 1.0           # m/s, walking speed, random waypoint, no pause (assumption)
    p_block_in: float = 0.01     # -, P(unblocked -> blocked) per step (assumption)
    p_block_out: float = 0.10    # -, P(blocked -> unblocked) per step (assumption; mean block 1 s)

    # ------------------------------------------------------------ hybrid policy
    thr_hi_db: float = 15.0      # dB, switch Wi-Fi -> Li-Fi above this Li-Fi SNR (assumption)
    thr_lo_db: float = 10.0      # dB, switch Li-Fi -> Wi-Fi below this Li-Fi SNR (assumption)
    switch_cost_steps: int = 3   # steps, zero-throughput handover time = 0.3 s (assumption)

    # ---------------------------------------------------------------- metrics
    outage_mbps: float = 1.0     # Mbps, a step is "in outage" if rate < this (assumption)
    coverage_snr_db: float = 15.0  # dB, SNR counted as "usable" for coverage maps (assumption)
    grid_n: int = 121            # -, points per side of the room grid for maps (assumption)

    # --------------------------------------------------------------- run control
    seed: int = 42               # -, master seed (prototype value)
    n_seeds: int = 30            # -, independent runs for statistics (E6/E7)

    # ---------------------------------------- optional fixes (all OFF = prototype)
    # See README "Known issues". Turning these on changes results; regression test
    # uses the defaults.
    lifi_imdd_capacity: bool = False      # use IM/DD capacity bound B*log2(1+e/(2*pi)*SNR/gap)
    wifi_correlated_shadowing: bool = False  # AR(1) shadowing correlated over distance
    wifi_shadow_decorr_m: float = 2.0     # m, shadowing decorrelation distance (assumption)
    wifi_mac_overhead: bool = False       # multiply Wi-Fi rate by a MAC efficiency
    wifi_mac_efficiency: float = 0.6      # -, MAC/protocol efficiency when enabled (assumption)

    # -------------------------------------------------------------- derived
    @property
    def led_pos(self) -> np.ndarray:
        """LED (x, y) in metres; defaults to room centre."""
        if self.led_xy is None:
            return np.array([self.room / 2, self.room / 2])
        return np.asarray(self.led_xy, dtype=float)

    @property
    def wifi_pos(self) -> np.ndarray:
        return np.asarray(self.wifi_xy, dtype=float)

    @property
    def half_angle(self) -> float:
        return float(np.deg2rad(self.half_angle_deg))

    @property
    def fov(self) -> float:
        return float(np.deg2rad(self.fov_deg))

    @property
    def steps(self) -> int:
        return int(self.sim_time / self.dt)

    def with_(self, **kw) -> "Config":
        """Return a copy with some fields changed."""
        return replace(self, **kw)


DEFAULT = Config()

# Citation/assumption tags used to build the README parameter table.
PARAM_TABLE: list[tuple[str, str, str, str]] = [
    # (field, symbol/description, unit, source)
    ("room", "Room side length (square room)", "m", "assumption"),
    ("h_led", "LED height", "m", "assumption ([K04] room height 3 m)"),
    ("h_rx", "Receiver height", "m", "[K04]"),
    ("p_t", "LED optical transmit power", "W", "assumption"),
    ("half_angle_deg", "LED semi-angle at half power", "deg", "typical value [G19]; assumption"),
    ("a_pd", "Photodiode area", "m^2", "[K04]"),
    ("resp", "PD responsivity", "A/W", "typical Si PIN ([K04]: 0.53)"),
    ("fov_deg", "Receiver field of view", "deg", "assumption"),
    ("n_ref", "Concentrator refractive index", "-", "[K04]"),
    ("t_s", "Optical filter gain", "-", "assumption (ideal)"),
    ("i_bg", "Ambient-light photocurrent", "A", "[K04]"),
    ("r_load", "Load resistance", "ohm", "assumption"),
    ("temp", "Temperature", "K", "assumption"),
    ("b_lifi", "Li-Fi bandwidth", "Hz", "assumption"),
    ("b_wifi", "Wi-Fi bandwidth", "Hz", "IEEE 802.11 20 MHz channel"),
    ("max_se", "Spectral-efficiency cap", "bit/s/Hz", "assumption (~64-QAM)"),
    ("gap_db", "SNR gap to Shannon", "dB", "assumption"),
    ("wifi_xy", "Wi-Fi AP position", "m", "assumption (corner)"),
    ("wifi_ptx_dbm", "Wi-Fi transmit power", "dBm", "assumption (typical)"),
    ("wifi_pl0_db", "Wi-Fi path loss at 1 m", "dB", "free-space at 2.4 GHz"),
    ("wifi_pl_exp", "Wi-Fi path-loss exponent", "-", "assumption (indoor 2-4 [R02])"),
    ("wifi_nf_db", "Wi-Fi noise figure", "dB", "assumption"),
    ("wifi_shadow_db", "Wi-Fi shadowing std", "dB", "assumption (indoor 3-6 dB)"),
    ("sim_time", "Simulated time per run", "s", "assumption"),
    ("dt", "Time step", "s", "assumption"),
    ("speed", "User speed (random waypoint)", "m/s", "assumption"),
    ("p_block_in", "P(enter blockage) per step", "-", "assumption"),
    ("p_block_out", "P(leave blockage) per step", "-", "assumption"),
    ("thr_hi_db", "Hysteresis up threshold", "dB", "assumption"),
    ("thr_lo_db", "Hysteresis down threshold", "dB", "assumption"),
    ("switch_cost_steps", "Handover outage per switch", "steps (x0.1 s)", "assumption"),
    ("outage_mbps", "Outage rate threshold", "Mbps", "assumption"),
    ("coverage_snr_db", "Coverage SNR threshold", "dB", "assumption"),
    ("seed", "Master random seed", "-", "prototype"),
    ("n_seeds", "Independent runs for statistics", "-", "assumption"),
]
