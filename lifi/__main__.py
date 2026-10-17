"""Command line: python -m lifi run --all [--quick]   |   python -m lifi run --exp E3"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import Config
from .experiments import EXPERIMENTS, run
from .summary import write_summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lifi", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run experiments")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="run E0-E8 and write the summary")
    g.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS), metavar="EX",
                   help=f"one or more of {', '.join(EXPERIMENTS)}")
    r.add_argument("--quick", action="store_true", help="fewer seeds/bits/grid points (<1 min)")
    r.add_argument("--out", default="results", help="output folder (default: results)")
    r.add_argument("--seed", type=int, default=Config.seed, help="master seed (default 42)")
    sub.add_parser("summary", help="rebuild results_summary.md from existing CSVs").add_argument(
        "--out", default="results")
    a = ap.parse_args(argv)

    if a.cmd == "summary":
        print(f"wrote {write_summary(Config(), Path(a.out))}")
        return
    cfg = Config(seed=a.seed)
    t0 = time.perf_counter()
    names = list(EXPERIMENTS) if a.all else a.exp
    run(names, cfg, Path(a.out), quick=a.quick)
    if a.all:
        print(f"wrote {write_summary(cfg, Path(a.out), quick=a.quick)}")
    print(f"total {time.perf_counter() - t0:.1f} s{'  (quick mode)' if a.quick else ''}")


if __name__ == "__main__":
    main()
