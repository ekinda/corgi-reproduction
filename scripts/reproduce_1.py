#!/usr/bin/env python3
"""Orchestrate the minimal steps required to regenerate Figure 2 inputs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"

BENCHMARK_CMDS = [
    [sys.executable, str(CODE_DIR / "bench_grt_tta_crosscell.py")],
    [sys.executable, str(CODE_DIR / "bench_grt_tta_crossboth.py")],
    [sys.executable, str(CODE_DIR / "bench_grt_tta_crosssequence.py")],
]

POSTPROCESS_CMDS = [
    [sys.executable, str(CODE_DIR / "gene_level_correlations_crosscell.py")],
    [sys.executable, str(CODE_DIR / "gene_level_correlations_crossboth.py")],
    [sys.executable, str(CODE_DIR / "gene_level_correlations_crosssequence.py")],
    [sys.executable, str(CODE_DIR / "gene_level_correlations_process.py")],
]


def run_steps(cmds: list[list[str]], description: str) -> None:
    print(f"\n==> {description}")
    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Run the benchmarking passes only (skip gene-level aggregation).",
    )
    args = parser.parse_args()

    run_steps(BENCHMARK_CMDS, "Benchmarking (cross-cell, cross-both, cross-sequence)")

    if not args.skip_postprocess:
        run_steps(POSTPROCESS_CMDS, "Post-processing (gene-level aggregates)")
    else:
        print("Skipping post-processing as requested.")


if __name__ == "__main__":
    main()
