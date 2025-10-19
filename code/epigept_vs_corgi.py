#!/usr/bin/env python3
"""Compare EpiGePT and Corgi predictions against ENCODE ground truth on chr8 regions."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

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

DEFAULT_EPI_DIR = PROCESSED_DIR / "figure3" / "corgi_vs_epigept" / "processed_predictions"
DEFAULT_CORGI_DIR = PROCESSED_DIR / "figure2" / "cross_both_tta"
DEFAULT_CORGI_EASYTEST_DIR = DATA_DIR / "figure3" / "tissues_124_192"
DEFAULT_REGIONS = DATA_DIR / "epigept" / "chr8_test_regions.bed"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "figure3" / "corgi_vs_epigept" / "comparison"
DEFAULT_SUMMARY = PROCESSED_DIR / "figure3" / "corgi_vs_epigept" / "comparison" / "results.csv"


def _run_bwtool(regions: Path, bigwig: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bwtool",
        "summary",
        str(regions),
        str(bigwig),
        str(output),
        "-keep-bed",
        "-header",
    ]
    subprocess.run(cmd, check=True)


def _collect_statistics(output_dir: Path, summary_csv: Path) -> None:
    records: list[dict[str, object]] = []
    for tissue in TISSUES:
        for assay in ASSAYS:
            encode_path = output_dir / f"tissue_{tissue}_{assay}_encode.tsv"
            corgi_path = output_dir / f"tissue_{tissue}_{assay}_corgi.tsv"
            epi_path = output_dir / f"tissue_{tissue}_{assay}_epigept.tsv"
            if not encode_path.exists() or not corgi_path.exists() or not epi_path.exists():
                continue
            encode_df = pd.read_csv(encode_path, sep="\t")
            corgi_df = pd.read_csv(corgi_path, sep="\t")
            epi_df = pd.read_csv(epi_path, sep="\t")

            records.append(
                {
                    "tissue": tissue,
                    "assay": assay,
                    "model": "corgi",
                    "pearson": pearsonr(encode_df["mean"], corgi_df["mean"])[0],
                    "spearman": spearmanr(encode_df["mean"], corgi_df["mean"])[0],
                }
            )
            records.append(
                {
                    "tissue": tissue,
                    "assay": assay,
                    "model": "epigept",
                    "pearson": pearsonr(encode_df["mean"], epi_df["mean"])[0],
                    "spearman": spearmanr(encode_df["mean"], epi_df["mean"])[0],
                }
            )
    if not records:
        raise RuntimeError("No bwtool summaries were generated; ensure the BigWig inputs are present.")

    df = pd.DataFrame.from_records(records)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)


def _summarise_assay(tissue: int, assay: str, regions: Path, epi_dir: Path, corgi_dir: Path, corgi_alt_dir: Path, output_dir: Path) -> None:
    epi_bw = epi_dir / f"tissue_{tissue}_{assay}.bw"
    if not epi_bw.exists():
        raise FileNotFoundError(f"Missing EpiGePT BigWig: {epi_bw}")
    epi_out = output_dir / f"tissue_{tissue}_{assay}_epigept.tsv"
    _run_bwtool(regions, epi_bw, epi_out)

    if tissue in (124, 192):
        prefix = corgi_alt_dir / f"tissue{tissue}_{assay.lower()}"
    else:
        prefix = corgi_dir / f"tissue{tissue}_{assay.lower()}"
    corgi_bw = prefix.parent / f"{prefix.name}_grt.bw"
    encode_bw = prefix.parent / f"{prefix.name}_encode.bw"
    if not corgi_bw.exists() or not encode_bw.exists():
        raise FileNotFoundError(f"Missing Corgi or ENCODE BigWig for tissue {tissue} assay {assay}")

    corgi_out = output_dir / f"tissue_{tissue}_{assay}_corgi.tsv"
    encode_out = output_dir / f"tissue_{tissue}_{assay}_encode.tsv"
    _run_bwtool(regions, corgi_bw, corgi_out)
    _run_bwtool(regions, encode_bw, encode_out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epigept-dir", type=Path, default=DEFAULT_EPI_DIR)
    parser.add_argument("--corgi-dir", type=Path, default=DEFAULT_CORGI_DIR)
    parser.add_argument("--corgi-alt-dir", type=Path, default=DEFAULT_CORGI_EASYTEST_DIR)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tissue in TISSUES:
        for assay in ASSAYS:
            # Skip combinations lacking ground truth to avoid bwtool errors.
            try:
                _summarise_assay(tissue, assay, args.regions, args.epigept_dir, args.corgi_dir, args.corgi_alt_dir, args.output_dir)
            except FileNotFoundError:
                print(f"Skipping tissue {tissue} assay {assay} due to missing data.")

    _collect_statistics(args.output_dir, args.summary)


if __name__ == "__main__":
    main()
