#!/usr/bin/env python3
"""Recreate the TF expression inputs and BED tiling used for the EpiGePT comparisons."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qnorm


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = REPO_ROOT / "processed_data"

DEFAULT_REFERENCE = DATA_DIR / "epigept" / "reference_TPM_tf_expr.csv"
DEFAULT_TF_NPY = DATA_DIR / "tf_expression.npy"
DEFAULT_TF_LIST = DATA_DIR / "trans_regulators_final_hgnc.txt"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "figure3" / "corgi_vs_epigept"
DEFAULT_BED_SOURCE = DATA_DIR / "figure3" / "fold3_notf_merged.bed"
DEFAULT_TILES_PATH = DEFAULT_OUTPUT_DIR / "fold3_notf_epigept.bed"
DEFAULT_CHR8_TEMPLATE = DEFAULT_OUTPUT_DIR / "fold3_notf_epigept_chr8_{index}.bed"

TISSUES = [124, 192, 213, 277, 323, 59]
SEQ_LENGTH = 128_000
STRIDE = 128_000


def _read_reference(reference_csv: Path) -> pd.DataFrame:
    if not reference_csv.exists():
        raise FileNotFoundError(
            f"Reference TF expression matrix not found at {reference_csv}. "
            "Please place the file provided with the EpiGePT download there."
        )
    return pd.read_csv(reference_csv, header=0, sep="\t", index_col=0)


def _load_tf_expression(tf_npy: Path) -> np.ndarray:
    if not tf_npy.exists():
        raise FileNotFoundError(
            f"Corgi TF expression array not found at {tf_npy}. "
            "Copy the pretraining TF expression file into this location before rerunning."
        )
    arr = np.load(tf_npy)
    return ((arr - 1.0) ** 10) - 0.1  # undo log transform used during pretraining


def _load_tf_list(tf_list_path: Path) -> list[str]:
    if not tf_list_path.exists():
        raise FileNotFoundError(
            f"TF symbol list not found at {tf_list_path}. "
            "Copy the trans_regulators_final_hgnc.txt file into place."
        )
    return tf_list_path.read_text(encoding="utf-8").strip().split()


def _quantile_normalise(
    sample: np.ndarray,
    reference_index: pd.Index,
    genes: list[str],
    positions: list[int],
    target: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(index=reference_index, columns=["1"], dtype=float)
    out.loc[genes, "1"] = sample[positions]
    out = out.fillna(0.0)
    out = qnorm.quantile_normalize(out, target=target)
    return out


def _write_quantile_normalised_profiles(
    tissues: list[int],
    expressions: np.ndarray,
    reference_index: pd.Index,
    genes: list[str],
    positions: list[int],
    target: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for tissue in tissues:
        qn = _quantile_normalise(expressions[tissue], reference_index, genes, positions, target)
        outfile = output_dir / f"tissue_{tissue}_qn.csv"
        qn.to_csv(outfile)


def _parse_bed(bed_path: Path) -> list[tuple[str, int, int]]:
    if not bed_path.exists():
        raise FileNotFoundError(
            f"Prediction BED template not found at {bed_path}. Copy fold3_notf_merged.bed into place."
        )
    df = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2], names=["chr", "start", "end"])
    return [(row.chr, int(row.start), int(row.end)) for row in df.itertuples(index=False)]


def _tile_region(start: int, end: int, seq_len: int, stride: int) -> list[tuple[int, int]]:
    tiles: list[tuple[int, int]] = []
    position = start
    while position + seq_len <= end:
        tiles.append((position, position + seq_len))
        position += stride
    return tiles


def _tile_regions(regions: list[tuple[str, int, int]], seq_len: int, stride: int) -> list[tuple[str, int, int]]:
    tiled: list[tuple[str, int, int]] = []
    for chrom, start, end in regions:
        for tile_start, tile_end in _tile_region(start, end, seq_len, stride):
            tiled.append((chrom, tile_start, tile_end))
    return tiled


def _write_tiling_outputs(
    regions: list[tuple[str, int, int]],
    seq_len: int,
    stride: int,
    combined_path: Path,
    chr8_template: Path,
) -> None:
    tiles = _tile_regions(regions, seq_len, stride)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text("\n".join(f"{c}\t{s}\t{e}" for c, s, e in tiles), encoding="utf-8")

    chr8_tiles = [tile for tile in tiles if tile[0] == "chr8"]
    chunk = 50
    for idx in range((len(chr8_tiles) + chunk - 1) // chunk):
        start = idx * chunk
        piece = chr8_tiles[start:start + chunk]
        if not piece:
            continue
        out_path = Path(str(chr8_template).format(index=idx))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(f"{c}\t{s}\t{e}" for c, s, e in piece), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tf-expression", type=Path, default=DEFAULT_TF_NPY)
    parser.add_argument("--tf-list", type=Path, default=DEFAULT_TF_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bed-source", type=Path, default=DEFAULT_BED_SOURCE)
    parser.add_argument("--combined-bed", type=Path, default=DEFAULT_TILES_PATH)
    parser.add_argument("--chr8-template", type=Path, default=DEFAULT_CHR8_TEMPLATE)
    args = parser.parse_args(argv)

    reference_df = _read_reference(args.reference)
    ref_target = qnorm.quantile_normalize(reference_df).mean(axis=1).to_numpy()
    tf_array = _load_tf_expression(args.tf_expression)
    tf_symbols = _load_tf_list(args.tf_list)

    intersect = {gene: idx for idx, gene in enumerate(tf_symbols) if gene in reference_df.index}
    if not intersect:
        raise RuntimeError("No TF overlap found between the reference matrix and Corgi regulators.")
    genes = list(intersect.keys())
    positions = [intersect[g] for g in genes]

    _write_quantile_normalised_profiles(
        TISSUES,
        tf_array,
        reference_df.index,
        genes,
        positions,
        ref_target,
        args.output_dir,
    )

    regions = _parse_bed(args.bed_source)
    _write_tiling_outputs(regions, SEQ_LENGTH, STRIDE, args.combined_bed, args.chr8_template)


if __name__ == "__main__":
    main(sys.argv[1:])
