#!/usr/bin/env python3
"""Run all preprocessing steps required for Figure 3."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"

EPIGEPT_CMDS = [
    [sys.executable, str(CODE_DIR / "epigept_tf_expression.py")],
    [sys.executable, str(CODE_DIR / "epigept_process_predictions.py")],
    [sys.executable, str(CODE_DIR / "epigept_vs_corgi.py")],
]

BORZOI_CMD = [sys.executable, str(CODE_DIR / "bench_grt_borzoi.py")]


def _run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-epigept",
        action="store_true",
        help="Skip the EpiGePT preprocessing steps (requires outputs to already exist).",
    )
    parser.add_argument(
        "--skip-borzoi",
        action="store_true",
        help="Skip the Corgi vs Borzoi benchmark.",
    )
    args = parser.parse_args(argv)

    if not args.skip_epigept:
        print("\n==> EpiGePT preprocessing")
        for cmd in EPIGEPT_CMDS:
            _run(cmd)
    else:
        print("Skipping EpiGePT preprocessing as requested.")

    if not args.skip_borzoi:
        print("\n==> Corgi vs Borzoi benchmark")
        _run(BORZOI_CMD)
    else:
        print("Skipping Corgi vs Borzoi benchmark as requested.")


if __name__ == "__main__":
    main(sys.argv[1:])
