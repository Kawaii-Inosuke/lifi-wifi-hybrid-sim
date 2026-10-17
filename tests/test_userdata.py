"""Tests for user-provided data handling (lifi/userdata.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifi.config import Config
from lifi.hybrid import generate_trace
from lifi.userdata import (DataError, blockage_steps, build_trace, fit_wifi, parse_blockage, parse_path,
                           parse_wifi, resample_blockage, resample_path)

EX = Path(__file__).resolve().parent.parent / "data" / "examples"
ROOM = 6.0


def csv(text: str) -> bytes:
    return text.strip().encode() + b"\n"


# ------------------------------------------------------------------- fitting
@pytest.mark.parametrize("a,n,sigma,ap", [(-40.0, 2.8, 3.0, (0.0, 0.0)), (-32.0, 2.0, 1.5, (3.0, 3.0)),
                                          (-45.0, 3.5, 4.0, (-2.0, 1.0))])
def test_fit_recovers_known_parameters(a, n, sigma, ap):
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, ROOM, (400, 2))
    d = np.maximum(np.hypot(xy[:, 0] - ap[0], xy[:, 1] - ap[1]), 1.0)
    rssi = a - 10 * n * np.log10(d) + rng.normal(0, sigma, len(d))
    f = fit_wifi(pd.DataFrame({"x_m": xy[:, 0], "y_m": xy[:, 1], "rssi_dbm": rssi}), ap)
    # Allowed error = 4 standard errors of least squares for this geometry (an AP far from the
    # points makes the 1 m intercept an extrapolation, so its tolerance is wider).
    X = np.column_stack([np.ones_like(d), -10 * np.log10(d)])
    se = sigma * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    assert abs(f.rssi_1m_dbm - a) < 4 * se[0]
    assert abs(f.pl_exp - n) < 4 * se[1]
    assert se[1] < 0.2                                  # the test itself is informative
    assert f.sigma_db == pytest.approx(sigma, rel=0.12)
    assert 0 < f.r2 <= 1


def test_fit_is_exact_without_noise():
    xy = np.array([[1, 1], [2, 3], [4, 1], [5, 5], [0.5, 5.5], [3, 3], [6, 0]], float)
    d = np.maximum(np.hypot(*xy.T), 1.0)
    f = fit_wifi(pd.DataFrame({"x_m": xy[:, 0], "y_m": xy[:, 1], "rssi_dbm": -38 - 25 * np.log10(d)}), (0, 0))
    assert f.rssi_1m_dbm == pytest.approx(-38, abs=1e-9)
    assert f.pl_exp == pytest.approx(2.5, abs=1e-9)
    assert f.sigma_db == pytest.approx(0, abs=1e-9)
    assert f.r2 == pytest.approx(1.0)


def test_fit_rejects_points_all_at_same_distance():
    df = pd.DataFrame({"x_m": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], "y_m": [0.1, 0.2, 0.1, 0.3, 0.3, 0.2],
                       "rssi_dbm": [-40, -41, -42, -40, -39, -41]})
    with pytest.raises(DataError, match="same distance"):
        fit_wifi(df, (0, 0))


def test_fitted_model_drives_wifi_snr_and_leaves_defaults_alone():
    from lifi.wifi import wifi_snr
    df = pd.DataFrame({"x_m": [1, 2, 3, 4, 5, 6.0], "y_m": [0, 0, 0, 0, 0, 0.0],
                       "rssi_dbm": [-40 - 30 * np.log10(x) for x in (1, 2, 3, 4, 5, 6)]})
    f = fit_wifi(df, (0, 0))
    cfg = f.apply(Config(), (0, 0))
    noise = -174 + 10 * np.log10(cfg.b_wifi) + cfg.wifi_nf_db
    got = 10 * np.log10(wifi_snr(np.array([[4.0, 0.0]]), cfg)[0])
    assert got == pytest.approx(-40 - 30 * np.log10(4) - noise)
    assert Config() == Config()                       # the default is not mutated
    assert Config().wifi_ptx_dbm == 20.0 and Config().wifi_pl0_db == 40.0 and Config().wifi_pl_exp == 3.0
    others = {k: v for k, v in vars(cfg).items() if not k.startswith("wifi_")}
    assert others == {k: v for k, v in vars(Config()).items() if not k.startswith("wifi_")}


# ---------------------------------------------------------------- validators
@pytest.mark.parametrize("text,match", [
    ("x_m,y_m\n1,2\n", "missing column"),
    ("x_m,y_m,rssi_dbm\n", "no data rows"),
    ("x_m,y_m,rssi_dbm\n1,1,-40\n2,2,-45\n3,3,abc\n4,4,-50\n5,5,-52\n1,5,-48\n", "non-numeric"),
    ("x_m,y_m,rssi_dbm\n1,1,-40\n2,2,-45\n3,3,-47\n", "at least 6"),
    ("x_m,y_m,rssi_dbm\n1,1,-40\n2,2,-45\n3,3,-47\n4,4,-50\n5,5,-52\n7,5,-48\n", "outside"),
    ("x_m,y_m,rssi_dbm\n100,100,-40\n200,200,-45\n300,300,-47\n400,400,-50\n500,500,-52\n100,500,-48\n",
     "centimetres"),
    ("x_m,y_m,rssi_dbm\n1,1,40\n2,2,45\n3,3,47\n4,4,50\n5,5,52\n1,5,48\n", "positive"),
    ("x_m,y_m,rssi_dbm\n1,1,-400\n2,2,-45\n3,3,-47\n4,4,-50\n5,5,-52\n1,5,-48\n", "between -120 and -10"),
    ("x_m,y_m,rssi_dbm\n1,1,\n2,2,-45\n3,3,-47\n4,4,-50\n5,5,-52\n1,5,-48\n", "empty"),
])
def test_wifi_validator_rejects_bad_files(text, match):
    with pytest.raises(DataError, match=match):
        parse_wifi(csv(text), ROOM)


@pytest.mark.parametrize("text,match", [
    ("t,x,y\n0,1,1\n1,2,2\n", "missing column"),
    ("t_s,x_m,y_m\n0,1,1\n", "at least 2"),
    ("t_s,x_m,y_m\n0,1,1\n2,2,2\n1,3,3\n", "strictly increasing"),
    ("t_s,x_m,y_m\n0,1,1\n1,2,2\n1,3,3\n", "strictly increasing"),
    ("t_s,x_m,y_m\n0,1,1\n1,-1,2\n", "outside"),
    ("t_s,x_m,y_m\n0,1,1\n5000000,2,2\n", "milliseconds"),
])
def test_path_validator_rejects_bad_files(text, match):
    with pytest.raises(DataError, match=match):
        parse_path(csv(text), ROOM)


@pytest.mark.parametrize("text,match", [
    ("t_s,blocked\n0,0\n1,2\n", "0 or 1"),
    ("t_s,blocked\n0,0\n1,yes\n", "non-numeric"),
    ("t_s\n0\n1\n", "missing column"),
    ("t_s,blocked\n1,0\n0,1\n", "strictly increasing"),
])
def test_blockage_validator_rejects_bad_files(text, match):
    with pytest.raises(DataError, match=match):
        parse_blockage(csv(text))


def test_garbage_bytes_give_friendly_error():
    with pytest.raises(DataError):
        parse_wifi(b"\x00\x01\x02\xff\xfe", ROOM)


def test_header_whitespace_and_case_are_tolerated():
    df = parse_path(csv(" T_S , X_M , Y_M \n0,1,1\n1,2,2\n"), ROOM)
    assert list(df.columns) == ["t_s", "x_m", "y_m"]


def test_example_files_are_valid():
    assert len(parse_wifi((EX / "wifi_rssi_example.csv").read_bytes(), ROOM)) >= 6
    assert len(parse_path((EX / "user_path_example.csv").read_bytes(), ROOM)) >= 2
    assert len(parse_blockage((EX / "blockage_example.csv").read_bytes())) >= 2


# ----------------------------------------------------------------- resampling
def test_resample_path_linear_interpolation():
    df = pd.DataFrame({"t_s": [10.0, 11.0], "x_m": [1.0, 2.0], "y_m": [3.0, 3.0]})
    p = resample_path(df, 0.1)
    assert len(p) == 11
    np.testing.assert_allclose(p[:, 0], np.linspace(1, 2, 11))
    np.testing.assert_allclose(p[:, 1], 3.0)


def test_resample_blockage_zero_order_hold():
    df = pd.DataFrame({"t_s": [0.0, 0.25, 0.5], "blocked": [0.0, 1.0, 0.0]})
    assert blockage_steps(df, 0.1) == 6
    b = resample_blockage(df, 0.1, 8)                  # 2 extra samples hold the last value
    np.testing.assert_array_equal(b, [0, 0, 0, 1, 1, 0, 0, 0])


# ------------------------------------------------------------ custom traces
def test_build_trace_without_overrides_equals_generate_trace():
    cfg = Config()
    a, b = generate_trace(cfg, np.random.default_rng(11)), build_trace(cfg, np.random.default_rng(11))
    for k in ("path", "blocked", "snr_l_db", "r_l", "r_w"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))


def test_build_trace_uses_user_path_and_blockage():
    cfg = Config()
    path = np.column_stack([np.linspace(1, 5, 50), np.full(50, 3.0)])
    blocked = np.zeros(50, bool)
    blocked[10:20] = True
    tr = build_trace(cfg, np.random.default_rng(0), path=path, blocked=blocked)
    np.testing.assert_array_equal(tr.path, path)
    np.testing.assert_array_equal(tr.blocked, blocked)
    assert np.all(tr.r_l[10:20] == 0) and np.all(tr.r_l[:10] > 0)
    with pytest.raises(ValueError):
        build_trace(cfg, np.random.default_rng(0), path=path, blocked=blocked[:40])
