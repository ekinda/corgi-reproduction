#!/usr/bin/env python3
"""Orchestrate the minimal steps required to regenerate Figure 2 inputs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"

POSTPROCESS_CMDS = [
    [sys.executable, str(CODE_DIR / "gene_level_correlations_crosscell.py")],
    [sys.executable, str(CODE_DIR / "gene_level_correlations_crossboth.py")],
    [sys.executable, str(CODE_DIR / "gene_level_correlations_crosssequence.py")],
    [sys.executable, str(CODE_DIR / "gene_level_correlations_fixed.py")],
]


def run_steps(cmds: list[list[str]], description: str) -> None:
    print(f"\n==> {description}")
    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main() -> None:
    run_steps(POSTPROCESS_CMDS, "Post-processing (gene-level aggregates)")
   
if __name__ == "__main__":
    main()
