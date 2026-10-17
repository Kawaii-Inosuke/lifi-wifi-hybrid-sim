"""Shared figure style and helpers: 300 dpi, consistent colours/markers per policy."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Fixed categorical order: a policy keeps the same colour in every figure.
_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
COLORS = {
    "Wi-Fi only": _SLOTS[0],
    "Li-Fi only": _SLOTS[1],
    "Hybrid": _SLOTS[2],
    "Hysteresis": _SLOTS[2],
    "Single lo": _SLOTS[3],
    "Single hi": _SLOTS[4],
    "Ideal": _SLOTS[5],
}
MARKERS = {"Wi-Fi only": "s", "Li-Fi only": "o", "Hybrid": "D", "Hysteresis": "D",
           "Single lo": "v", "Single hi": "^", "Ideal": "*"}
SCHEME_COLORS = {2: _SLOTS[0], 4: _SLOTS[1], 8: _SLOTS[2]}
SCHEME_MARKERS = {2: "o", 4: "s", 8: "^"}
SCHEME_NAMES = {2: "OOK", 4: "4-PAM", 8: "8-PAM"}

# Single-hue sequential ramp (light = low, dark = high) for heatmaps.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                 "#184f95", "#0d366b"])
INK = "#2b2b2b"
MUTED = "#6b6b6b"


def policy_key(name: str) -> str:
    """Map a policy label to a colour/marker key."""
    if name.startswith("Hysteresis") or name == "Hybrid":
        return "Hysteresis"
    if name.startswith("Single"):
        return "Single lo" if name.endswith(" 10 dB") else "Single hi"
    if name.startswith("Ideal"):
        return "Ideal"
    return name


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.dpi": 100,
    })


def save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    """Save a figure as results/figures/<name>.png at 300 dpi and close it."""
    p = Path(outdir) / "figures" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p
