"""Smoke tests for the Streamlit demo (streamlit.testing.v1.AppTest). Run: python -m pytest -q"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Project", "Live demo", "Your data (optional)"]
BANNER = "Using user-provided data"


@pytest.fixture
def at():
    a = AppTest.from_file(APP, default_timeout=60)
    a.run()
    return a


def _banners(at):
    return [m.value for m in at.markdown if "userbanner" in m.value and BANNER in m.value]


def _tables(at):
    return [t.value.to_dict() for t in at.table]


def test_app_loads_without_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == PAGES


def test_every_page_renders_content(at):
    for i, tab in enumerate(at.tabs):
        assert len(tab.children) > 0, f"page {PAGES[i]} is empty"
    assert any("not a hardware measurement" in c.value for c in at.caption)
    assert not any("footer" in m.value for m in at.markdown)          # the old fixed footer is gone


def test_project_page_headline_metrics_and_expanders(at):
    assert [m.label for m in at.metric] == ["Speed (Mbps)", "Time disconnected (%)"] * 3
    labels = [e.label for e in at.expander]
    for name in ("More experiments", "Limitations and honest notes", "Policy comparison (E7)",
                 "Advanced settings and channel explorer"):
        assert name in labels


def test_down_threshold_must_be_below_up(at):
    at.slider(key="hy_down").set_value(20.0).run()
    assert not at.exception
    assert any("must be lower than" in e.value for e in at.error)


def test_channel_reset_restores_defaults(at):
    at.slider(key="ch_h").set_value(3.5).run()
    assert at.slider(key="ch_h").value == pytest.approx(3.5)
    at.button(key="ch_reset").click().run()
    assert at.slider(key="ch_h").value == pytest.approx(3.0)
    assert not at.exception


def test_blockage_control_and_prototype_replay(at):
    at.select_slider(key="hy_pin").set_value(0.05).run()
    assert not at.exception
    at.select_slider(key="hy_pin").set_value(0.01).run()
    at.checkbox(key="hy_proto").check().run()
    assert not at.exception
    runs = [t.value for t in at.table if list(t.value.index) == ["Wi-Fi only", "Li-Fi only", "Hybrid"]]
    speeds = runs[0]["Speed (Mbps)"].tolist()                 # first run table = live demo
    assert speeds == ["120.0", "438.3", "407.8"]             # the prototype's seed-42 numbers


def test_experiment_gallery_every_experiment(at):
    n = len(at.selectbox(key="gal_exp").options)
    assert n == 10                                   # E0-E8 and S1
    for i in range(n):
        at.selectbox(key="gal_exp").select_index(i).run()
        assert not at.exception, i
        assert not at.warning, [w.value for w in at.warning]   # no "missing figure/CSV" messages


def test_opens_with_simulated_data_only(at):
    assert not _banners(at)
    for k in ("ud_use_wifi", "ud_use_path", "ud_use_block"):
        assert at.checkbox(key=k).value is False
        assert at.checkbox(key=k).disabled            # nothing to use until a file is provided
    assert at.session_state.ud_samples is False


def test_sample_data_end_to_end(at):
    at.button(key="ud_load").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not _banners(at)                            # loading alone changes nothing
    for k in ("ud_use_wifi", "ud_use_path", "ud_use_block"):
        assert not at.checkbox(key=k).disabled
        at.checkbox(key=k).check().run()
        assert not at.exception, (k, [e.value for e in at.exception])
    b = _banners(at)
    assert len(b) == 2                                 # Live demo page + Your data page
    assert any("Wi-Fi fit" in x and "your path" in x and "blockage" in x for x in b)
    assert any("Compared with the default simulation" in m.value for m in at.markdown)
    at.button(key="ud_clear").click().run()
    assert not at.exception
    assert not _banners(at)


def test_wifi_fit_only_affects_demo_when_enabled(at):
    at.button(key="ud_load").click().run()
    before = _tables(at)
    at.checkbox(key="ud_use_wifi").check().run()
    assert not at.exception
    assert any("fitted Wi-Fi model" in x for x in _banners(at))
    at.checkbox(key="ud_use_wifi").uncheck().run()
    assert not _banners(at)
    assert _tables(at) == before                       # back to the simulated numbers
