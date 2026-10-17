"""Read precomputed results and documentation for the demo app (never recomputes E6/E7)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
FIGS = ROOT / "results" / "figures"
EXAMPLES = ROOT / "data" / "examples"
REGEN = "python -m lifi run --all"

EXPERIMENT_TITLES = {
    "E0": "E0 - Prototype baseline (seed 42) and timeline",
    "E1": "E1 - Received power and SNR heatmaps",
    "E2": "E2 - BER vs distance, theory vs Monte Carlo",
    "E3": "E3 - Parameter sweeps (coverage, corner SNR)",
    "E4": "E4 - Blockage sweep",
    "E5": "E5 - Hysteresis threshold sweep",
    "E6": "E6 - 30-seed statistics (headline numbers)",
    "E7": "E7 - Policy comparison",
    "E8": "E8 - 2x2 LED grid coverage",
    "S1": "S1 - Sensitivity to optional fixes (table only)",
}


def read_result_csv(name: str) -> pd.DataFrame | None:
    """results/data/<name>.csv or None if missing."""
    p = DATA / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def experiment_csvs(exp: str) -> list[Path]:
    return sorted(DATA.glob(f"{exp}_*.csv"))


def experiment_figure(exp: str) -> Path | None:
    hits = sorted(FIGS.glob(f"{exp}_*.png"))
    return hits[0] if hits else None


def headline_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format an E6/E7 summary CSV as 'mean ± CI' strings (numbers come from the CSV)."""
    def pm(m, c, nd):
        return f"{m:.{nd}f} ± {c:.{nd}f}"
    return pd.DataFrame({
        "Scheme": summary.policy,
        "Avg throughput (Mbps)": [pm(m, c, 1) for m, c in zip(summary.avg_mean, summary.avg_ci95)],
        "Outage (%)": [pm(m, c, 2) for m, c in zip(summary.outage_mean, summary.outage_ci95)],
        "Switches / 300 s": [pm(m, c, 1) for m, c in zip(summary.switches_mean, summary.switches_ci95)],
        "Time on Li-Fi (%)": [pm(m, c, 1) for m, c in zip(summary.lifi_share_mean, summary.lifi_share_ci95)],
    })


def _read(name: str) -> str | None:
    p = ROOT / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def markdown_section(doc: str, heading: str) -> str | None:
    """Body of the '## heading' section of a markdown file (until the next '## ')."""
    text = _read(doc)
    if text is None:
        return None
    m = re.search(rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
    return m.group(1).strip() if m else None


def explainer_paragraph(exp: str) -> str | None:
    """The EXPLAINER.md paragraph that starts with '**E3:' (or '**S1:')."""
    text = _read("EXPLAINER.md")
    if text is None:
        return None
    for para in re.split(r"\n\s*\n", text):
        if para.lstrip().startswith(f"**{exp}:"):
            return para.strip()
    return None
