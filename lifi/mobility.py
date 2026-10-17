"""User mobility (random waypoint) and Li-Fi blockage (2-state Markov chain)."""
from __future__ import annotations

import numpy as np

from .config import Config


def random_waypoint(cfg: Config, rng: np.random.Generator, steps: int | None = None
                    ) -> np.ndarray:
    """Positions (steps, 2) of a user walking at cfg.speed between uniform waypoints, no pause."""
    steps = cfg.steps if steps is None else steps
    hop = cfg.speed * cfg.dt
    pos = rng.uniform(0, cfg.room, 2)
    wp = rng.uniform(0, cfg.room, 2)
    out = np.empty((steps, 2))
    for i in range(steps):
        v = wp - pos
        dist = np.linalg.norm(v)
        if dist < hop:
            pos, wp = wp, rng.uniform(0, cfg.room, 2)
        else:
            pos = pos + v / dist * hop
        out[i] = pos
    return out


def markov_blockage(steps: int, p_in: float, p_out: float, rng: np.random.Generator
                    ) -> np.ndarray:
    """Boolean blockage state per step. Starts unblocked.

    Unblocked -> blocked with prob p_in, blocked -> unblocked with prob p_out.
    One uniform draw per step from step 1 on (same stream as the prototype's loop).
    """
    u = rng.random(steps - 1)
    blocked = np.zeros(steps, bool)
    for i in range(1, steps):
        blocked[i] = (u[i - 1] < p_in) if not blocked[i - 1] else (u[i - 1] > p_out)
    return blocked


def stationary_blockage(p_in: float, p_out: float) -> float:
    """Long-run fraction of time blocked: p_in / (p_in + p_out)."""
    return p_in / (p_in + p_out) if p_in + p_out > 0 else 0.0
