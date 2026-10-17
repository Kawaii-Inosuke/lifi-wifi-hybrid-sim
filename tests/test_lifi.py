"""Unit and regression tests. Run with: python -m pytest -q"""
import math

import numpy as np
import pytest

from lifi.ber import ber_ook, ber_pam, monte_carlo_pam, qfunc
from lifi.channel import lifi_gain, lifi_snr
from lifi.config import Config
from lifi.experiments import prototype_baseline
from lifi.hybrid import Trace, hysteresis_policy

CFG = Config()


# ------------------------------------------------------------------ channel
def test_gain_matches_hand_formula():
    # Receiver 1 m east and 1 m north of the LED axis.
    x, y = 4.0, 4.0
    dh = math.sqrt(1.0 ** 2 + 1.0 ** 2)
    dv = 3.0 - 0.85
    d = math.sqrt(dh ** 2 + dv ** 2)
    cos_t = dv / d
    m = -math.log(2) / math.log(math.cos(math.radians(60)))      # = 1 for 60 deg
    g = 1.5 ** 2 / math.sin(math.radians(60)) ** 2                 # concentrator gain
    h_hand = (m + 1) * 1e-4 / (2 * math.pi * d ** 2) * cos_t ** m * 1.0 * g * cos_t
    h, dist = lifi_gain(np.array([[x, y]]), CFG)
    assert m == pytest.approx(1.0)
    assert dist[0] == pytest.approx(d)
    assert h[0] == pytest.approx(h_hand, rel=1e-12)


def test_gain_zero_outside_fov():
    dv = CFG.h_led - CFG.h_rx
    r_edge = math.tan(CFG.fov) * dv                                 # ~3.72 m
    c = CFG.led_pos
    h_in, _ = lifi_gain(np.array([[c[0] + r_edge - 0.01, c[1]]]), CFG)
    h_out, _ = lifi_gain(np.array([[c[0] + r_edge + 0.01, c[1]], [0.0, 0.0]]), CFG)
    assert h_in[0] > 0
    assert np.all(h_out == 0.0)
    snr, p = lifi_snr(np.array([[0.0, 0.0]]), CFG)
    assert snr[0] == 0.0 and p[0] == 0.0


def test_gain_falls_with_distance():
    r = np.linspace(0, 3.5, 50)
    h, _ = lifi_gain(np.column_stack([CFG.led_pos[0] + r, np.full_like(r, 3.0)]), CFG)
    assert np.all(np.diff(h) < 0)


# ---------------------------------------------------------------------- BER
@pytest.mark.parametrize("M", [2, 4, 8])
def test_ber_monotone_in_snr(M):
    snr = 10 ** (np.linspace(-5, 35, 400) / 10)
    b = ber_pam(snr, M)
    assert np.all(np.diff(b) <= 0)
    assert np.all(np.diff(b[b > 1e-300]) < 0)


def test_ook_is_q_sqrt_snr():
    snr = 10 ** (np.linspace(0, 15, 31) / 10)
    np.testing.assert_allclose(ber_ook(snr), qfunc(np.sqrt(snr)), rtol=1e-14)
    assert qfunc(0.0) == pytest.approx(0.5)


def test_higher_order_pam_worse_at_same_snr():
    snr = 10 ** (20 / 10)
    assert ber_pam(snr, 2) < ber_pam(snr, 4) < ber_pam(snr, 8)


@pytest.mark.parametrize("M,snr_db", [(2, 6.0), (2, 9.0), (4, 14.0), (4, 18.0), (8, 20.0), (8, 24.0)])
def test_monte_carlo_matches_theory(M, snr_db):
    rng = np.random.default_rng(1)
    ber, errs = monte_carlo_pam(np.array([snr_db]), M, 600_000, rng)
    theory = ber_pam(10 ** (snr_db / 10), M)
    assert errs[0] >= 200                    # enough errors for a meaningful comparison
    assert ber[0] == pytest.approx(theory, rel=0.15)


# ------------------------------------------------------------------- hybrid
def _trace(snr_db):
    n = len(snr_db)
    return Trace(path=np.zeros((n, 2)), blocked=np.zeros(n, bool), snr_l_db=np.asarray(snr_db, float),
                 r_l=np.full(n, 300.0), r_w=np.full(n, 100.0), dt=0.1)


def test_no_switch_inside_hysteresis_band():
    rng = np.random.default_rng(0)
    snr = rng.uniform(10.0, 15.0, 2000)      # always between down (10) and up (15)
    res = hysteresis_policy(_trace(snr), 15.0, 10.0, 3)
    assert res.switches == 0
    assert not res.use_lifi.any()
    # Same if already on Li-Fi: go up once, then stay inside the band.
    res2 = hysteresis_policy(_trace(np.r_[20.0, snr]), 15.0, 10.0, 3)
    assert res2.switches == 1
    assert res2.use_lifi.all()


def test_switch_penalty_and_counts():
    snr = np.r_[np.full(10, 5.0), np.full(10, 20.0), np.full(10, 5.0)]
    res = hysteresis_policy(_trace(snr), 15.0, 10.0, 3)
    assert res.switches == 2
    np.testing.assert_array_equal(res.rate[10:13], 0.0)         # 3-step handover outage
    np.testing.assert_array_equal(res.rate[13:20], 300.0)
    np.testing.assert_array_equal(res.rate[20:23], 0.0)


def test_single_threshold_switches_more_than_hysteresis():
    # SNR jitters +/-1 dB around 15 dB (the up threshold).
    snr = 15.0 + np.random.default_rng(3).uniform(-1, 1, 2000)
    single = hysteresis_policy(_trace(snr), 15.0, 15.0, 3)
    hyst = hysteresis_policy(_trace(snr), 15.0, 10.0, 3)
    assert hyst.switches == 1              # goes to Li-Fi once and stays
    assert single.switches > 100           # ping-pongs on every crossing of 15 dB


# --------------------------------------------------------------- regression
def test_regression_matches_prototype_seed42():
    """Default config + seed 42 must reproduce lifi_sim.py exactly."""
    res, _, _ = prototype_baseline(Config())
    # rounded values as printed by the prototype
    assert round(res["Wi-Fi only"]["avg"], 1) == 120.0
    assert round(res["Wi-Fi only"]["outage"], 2) == 0.00
    assert round(res["Li-Fi only"]["avg"], 1) == 438.3
    assert round(res["Li-Fi only"]["outage"], 2) == 10.73
    assert round(res["Hybrid"]["avg"], 1) == 407.8
    assert round(res["Hybrid"]["outage"], 2) == 6.60
    assert res["Hybrid"]["switches"] == 67
    assert round(res["Hybrid"]["lifi_share"], 1) == 75.0
    # full precision values from the original script (identical draw order)
    assert res["Li-Fi only"]["avg"] == pytest.approx(438.3044363710397, rel=1e-12)
    assert res["Hybrid"]["avg"] == pytest.approx(407.8285859917954, rel=1e-12)
    assert res["Li-Fi only"]["outage"] == pytest.approx(10.733333333333334, rel=1e-12)
