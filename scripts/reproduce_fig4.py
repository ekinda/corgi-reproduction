#!/usr/bin/env python3
"""Run the Avocado preprocessing, training, and evaluation steps required for Figure 4."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"

PREPARE_CMD = [sys.executable, str(CODE_DIR / "avocado_prepare_data2.py")]
TRAIN_CMD = [sys.executable, str(CODE_DIR / "avocado_train2.py")]
VALIDATION_CMD = [sys.executable, str(CODE_DIR / "avocado_validation.py")]
TEST_CMD = [sys.executable, str(CODE_DIR / "avocado_test.py")]


def _extend_command(base: list[str], extra: str | None) -> list[str]:
    if extra:
        base = [*base, *shlex.split(extra)]
    return base


def _run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prepare", action="store_true", help="Skip avocado_prepare_data2.py.")
    parser.add_argument("--skip-train", action="store_true", help="Skip avocado_train2.py.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip avocado_validation.py.")
    parser.add_argument("--skip-test", action="store_true", help="Skip avocado_test.py.")
    parser.add_argument("--prepare-args", default="", help="Extra arguments to pass to avocado_prepare_data2.py.")
    parser.add_argument("--train-args", default="", help="Extra arguments to pass to avocado_train2.py.")
    parser.add_argument("--validation-args", default="", help="Extra arguments to pass to avocado_validation.py.")
    parser.add_argument("--test-args", default="", help="Extra arguments to pass to avocado_test.py.")
    args = parser.parse_args(argv)

    if not args.skip_prepare:
        _run(_extend_command(PREPARE_CMD.copy(), args.prepare_args))
    else:
        print("Skipping Avocado data preparation as requested.")

    if not args.skip_train:
        _run(_extend_command(TRAIN_CMD.copy(), args.train_args))
    else:
        print("Skipping Avocado training as requested.")

    if not args.skip_validation:
        _run(_extend_command(VALIDATION_CMD.copy(), args.validation_args))
    else:
        print("Skipping Avocado validation inference as requested.")

    if not args.skip_test:
        _run(_extend_command(TEST_CMD.copy(), args.test_args))
    else:
        print("Skipping Avocado test inference as requested.")


if __name__ == "__main__":
    main(sys.argv[1:])
