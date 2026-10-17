"""Regenerate the SYNTHETIC example files in data/examples/ (not real measurements).

Run: python -m demo.make_examples
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "examples"

# Known "true" values used to synthesise the Wi-Fi survey (the fit should recover them).
TRUE_AP = (0.0, 0.0)
TRUE_RSSI_1M = -40.0      # dBm
TRUE_N = 2.8
TRUE_SIGMA = 3.0          # dB


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    # 1. Wi-Fi survey: 20 points spread over the room.
    xy = rng.uniform(0.3, 5.7, (20, 2))
    d = np.maximum(np.hypot(xy[:, 0] - TRUE_AP[0], xy[:, 1] - TRUE_AP[1]), 1.0)
    rssi = TRUE_RSSI_1M - 10 * TRUE_N * np.log10(d) + rng.normal(0, TRUE_SIGMA, len(d))
    pd.DataFrame({"x_m": xy[:, 0].round(2), "y_m": xy[:, 1].round(2), "rssi_dbm": rssi.round(1)}
                 ).to_csv(OUT / "wifi_rssi_example.csv", index=False)

    # 2. User path: walk a loop around the desk area at ~0.8 m/s, then cross the room, 1 Hz samples.
    loop = [[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]
    corners = np.array(loop + loop[1:] + [[3, 3], [5.5, 0.5]], float)
    seg = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    s_knots = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s_knots[-1] / 0.8, 1.0)
    s = t * 0.8
    path = np.column_stack([np.interp(s, s_knots, corners[:, 0]), np.interp(s, s_knots, corners[:, 1])])
    pd.DataFrame({"t_s": t, "x_m": path[:, 0].round(3), "y_m": path[:, 1].round(3)}
                 ).to_csv(OUT / "user_path_example.csv", index=False)

    # 3. Blockage timeline: same duration, three people walking past.
    tb = np.arange(0, t[-1] + 1, 0.5)
    blocked = np.zeros(len(tb), int)
    for a, b in ((8, 10), (22, 26.5), (40, 41)):
        blocked[(tb >= a) & (tb < b)] = 1
    pd.DataFrame({"t_s": tb, "blocked": blocked}).to_csv(OUT / "blockage_example.csv", index=False)
    print(f"wrote examples to {OUT}")


if __name__ == "__main__":
    main()
