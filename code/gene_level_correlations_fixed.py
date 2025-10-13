#!/usr/bin/env python3
"""Compute corrected gene-level correlations and emit correlations_fixed.pk files for Figure 2 outputs."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG2_DIR = REPO_ROOT / "processed_data" / "figure2"

DATASETS = {
    "cross_celltype": {
        "directory": FIG2_DIR / "gene_level_cross_celltype",
        "true_suffix": "_encode",
        "pred_suffix": "_grt",
        "true_file": "all_expressions_encode.csv",
        "pred_file": "all_expressions_grt.csv",
    },
    "cross_both": {
        "directory": FIG2_DIR / "gene_level_cross_both",
        "true_suffix": "_encode",
        "pred_suffix": "_grt",
        "true_file": "all_expressions_encode.csv",
        "pred_file": "all_expressions_grt.csv",
    },
    "cross_region": {
        "directory": FIG2_DIR / "gene_level_cross_region",
        "true_suffix": "_encode",
        "pred_suffix": "_grt",
        "true_file": "all_expressions_encode.csv",
        "pred_file": "all_expressions_grt.csv",
    },
}


def _strip_suffix(value: str, suffix: str) -> str:
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def _load_gene_list(path: str | None) -> set[str] | None:
    if not path:
        return None
    gene_path = Path(path)
    if not gene_path.exists():
        raise FileNotFoundError(f"Gene list file not found: {gene_path}")

    if gene_path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if gene_path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(gene_path, sep=sep)
        return set(df.iloc[:, 0].astype(str))

    with gene_path.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def _filter_genes(df: pd.DataFrame, genes: set[str] | None) -> pd.DataFrame:
    if genes is None:
        return df
    subset = df.loc[df.index.intersection(genes)]
    if subset.empty:
        raise ValueError("Gene filter removed all rows; please verify the supplied gene list.")
    return subset


def _compute_dataset(name: str, config: dict[str, Path | str], gene_filter: set[str] | None) -> None:
    directory = Path(config["directory"])
    true_path = directory / config["true_file"]
    pred_path = directory / config["pred_file"]

    if not true_path.exists() or not pred_path.exists():
        print(f"[{name}] Skipping: missing inputs ({true_path} or {pred_path}).")
        return

    true_df = pd.read_csv(true_path, index_col=0)
    pred_df = pd.read_csv(pred_path, index_col=0)

    true_df = _filter_genes(true_df, gene_filter)
    pred_df = _filter_genes(pred_df, gene_filter)

    true_suffix = config["true_suffix"]
    pred_suffix = config["pred_suffix"]

    correlations = {}
    r_values: list[float] = []

    for true_col in true_df.columns:
        base = _strip_suffix(true_col, true_suffix)
        pred_col = f"{base}{pred_suffix}"
        if pred_col not in pred_df.columns:
            continue
        x = pred_df[pred_col]
        y = true_df[true_col]
        mask = x.notna() & y.notna()
        if not mask.any():
            continue
        r = pearsonr(x[mask], y[mask])[0]
        correlations[true_col] = float(r)
        r_values.append(float(r))

    if not correlations:
        print(f"[{name}] No overlapping column pairs found; ensure post-processing scripts ran successfully.")
        return

    out_path = directory / "correlations_fixed.pk"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(correlations, handle)

    mean_r = float(np.mean(r_values))
    print(f"[{name}] Saved {len(correlations)} correlations (mean r={mean_r:.3f}) to {out_path}.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(DATASETS.keys()),
        default=sorted(DATASETS.keys()),
        help="Specific datasets to process (default: all).",
    )
    parser.add_argument(
        "--coding-genes",
        default=None,
        help="Optional path to a text/CSV/TSV file listing gene symbols to retain.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    gene_filter = _load_gene_list(args.coding_genes)

    for name in args.datasets:
        _compute_dataset(name, DATASETS[name], gene_filter)


if __name__ == "__main__":
    main()
