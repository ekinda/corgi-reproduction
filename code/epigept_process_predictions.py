#!/usr/bin/env python3
"""Convert downloaded EpiGePT prediction CSVs into BigWig files."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = REPO_ROOT / "processed_data"

ASSAYS = [
    "DNase",
    "CTCF",
    "H3K27ac",
    "H3K4me3",
    "H3K36me3",
    "H3K27me3",
    "H3K9me3",
    "H3K4me1",
]
TISSUES = [124, 192, 213, 277, 323]

DEFAULT_RAW_DIR = DATA_DIR / "epigept" / "raw_predictions"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "figure3" / "corgi_vs_epigept" / "processed_predictions"
DEFAULT_CHROM_SIZES = DATA_DIR / "hg38.chrom.sizes"


def _bedgraph_from_predictions(raw: pd.DataFrame, assay: str) -> list[str]:
    series = raw[assay].to_dict()
    return ["\t".join(key.replace(":", "\t").split("_")) + f"\t{value}" for key, value in series.items()]


def _process_tissue(
    tissue: int,
    assays: list[str],
    raw_dir: Path,
    output_dir: Path,
    chrom_sizes: Path,
) -> None:
    frames = []
    for part in range(4):
        csv_path = raw_dir / f"tissue_{tissue}_bed_{part}.csv"
        if not csv_path.exists():
            continue
        frames.append(pd.read_csv(csv_path, index_col=0))
    if not frames:
        raise FileNotFoundError(f"No raw prediction CSVs found for tissue {tissue} in {raw_dir}.")

    merged = pd.concat(frames, axis=0)
    for assay in assays:
        bedgraph_lines = _bedgraph_from_predictions(merged, assay)
        stem = output_dir / f"tissue_{tissue}_{assay}"
        bedgraph_path = stem.with_suffix(".bedgraph")
        bigwig_path = stem.with_suffix(".bw")
        bedgraph_path.parent.mkdir(parents=True, exist_ok=True)
        bedgraph_path.write_text("\n".join(bedgraph_lines), encoding="utf-8")
        subprocess.run(
            ["bedGraphToBigWig", str(bedgraph_path), str(chrom_sizes), str(bigwig_path)],
            check=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chrom-sizes", type=Path, default=DEFAULT_CHROM_SIZES)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tissue in TISSUES:
        _process_tissue(tissue, ASSAYS, args.raw_dir, args.output_dir, args.chrom_sizes)


if __name__ == "__main__":
    main()
