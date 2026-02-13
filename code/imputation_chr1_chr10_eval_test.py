"""Quick smoke test for imputation on a small subset of tissues/windows.

This is a reduced version of imputation_chr1_chr10_eval.py to verify runtime.
It limits tissues and window counts and can optionally cap selected bins.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pyfaidx import Fasta

from corgi.config_corgiplus import config_corgiplus
from corgi.utils import load_experiment_mask
from corgiplus_delta_bench_tta import load_model, pred_tta
from epigept_inference_wrapper import EpigeptInferenceWrapper

SEQ_LENGTH = 524_288
CORGI_FULL_BINS = 8192
CORGI_TARGET_BINS = 6144
CORGI_CROP_BINS = (CORGI_FULL_BINS - CORGI_TARGET_BINS) // 2
BIN_SIZE = 64
EPIGEPT_BP = 128_000
EPIGEPT_OUT_BINS = 1000
EPIGEPT_OUT_BINS_64 = EPIGEPT_OUT_BINS * 2

CHANNELS_INTEREST = [
    "dnase",
    "atac",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
    "h3k9ac",
    "h3k9me3",
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k79me2",
    "ctcf",
    "wgbs",
    "cage",
    "rampage",
]


def read_ids(path: Path) -> List[int]:
    with path.open() as handle:
        return [int(x) for x in handle.read().strip().split() if x.strip()]


def load_experiments(path: Path) -> List[str]:
    with path.open() as handle:
        return [x.strip() for x in handle.read().strip().split() if x.strip()]


def load_bed(path: Path) -> List[Tuple[str, int, int]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, start_s, end_s, *_ = line.rstrip().split("\t")
            rows.append((chrom, int(start_s), int(end_s)))
    return rows


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def intervals_to_bins(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out = []
    for start, end in intervals:
        start_bin = start // BIN_SIZE
        end_bin = int(np.ceil(end / BIN_SIZE))
        out.append((start_bin, end_bin))
    return out


def build_bin_index(chrom_bins: int, selected_bins: List[int]) -> np.ndarray:
    idx = np.full(chrom_bins, -1, dtype=np.int64)
    for out_i, bin_i in enumerate(selected_bins):
        idx[bin_i] = out_i
    return idx


def build_selected_bins(interval_bins: List[Tuple[int, int]]) -> List[int]:
    bins: List[int] = []
    for start_bin, end_bin in interval_bins:
        bins.extend(range(start_bin, end_bin))
    return bins


def windows_from_intervals(
    interval_bins: List[Tuple[int, int]],
    window_bins: int,
    chrom_bins: int,
) -> List[int]:
    windows: List[int] = []
    for start_bin, end_bin in interval_bins:
        cur = start_bin
        while cur + window_bins <= end_bin:
            windows.append(cur)
            cur += window_bins
        if cur < end_bin:
            center = int((cur + end_bin - window_bins) // 2)
            center = max(start_bin, min(center, end_bin - window_bins))
            windows.append(center)
    for start_bin in windows:
        if start_bin < 0 or start_bin + window_bins > chrom_bins:
            raise ValueError("Window extends beyond chromosome bins")
    return windows


def one_hot_encode(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base not in mapping:
            continue
        onehot[i, mapping[base]] = 1.0
    return onehot


def count_invalid_bases(seq: str) -> int:
    mapping = {"A", "C", "G", "T", "a", "c", "g", "t"}
    return sum(1 for base in seq if base not in mapping)


def slice_bins_with_pad(arr: np.ndarray, start: int, end: int, label: str = "") -> np.ndarray:
    left = max(0, -start)
    right = max(0, end - arr.shape[0])
    start_clip = max(start, 0)
    end_clip = min(end, arr.shape[0])
    sliced = arr[start_clip:end_clip]
    if left == 0 and right == 0:
        return sliced
    if label:
        logging.info(
            "Padding %s with zeros: start=%d end=%d (left=%d right=%d)",
            label,
            start,
            end,
            left,
            right,
        )
    if arr.ndim == 1:
        return np.pad(sliced, (left, right), mode="constant")
    if arr.ndim == 2:
        return np.pad(sliced, ((left, right), (0, 0)), mode="constant")
    raise ValueError("slice_bins_with_pad only supports 1D or 2D arrays")


def get_padded_sequence(genome: Fasta, chrom: str, start_bp: int, end_bp: int) -> str:
    chrom_len = len(genome[chrom])
    left = max(0, -start_bp)
    right = max(0, end_bp - chrom_len)
    start_clip = max(start_bp, 0)
    end_clip = min(end_bp, chrom_len)
    core = genome[chrom][start_clip:end_clip].seq
    if left == 0 and right == 0:
        return core
    logging.info(
        "Padding DNA with Ns for %s:%d-%d (left=%d right=%d)",
        chrom,
        start_bp,
        end_bp,
        left,
        right,
    )
    return ("N" * left) + core + ("N" * right)


def get_channel_indices(experiments: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(experiments)}


def build_combined(pred: np.ndarray, idx_map: Dict[str, int], name: str) -> np.ndarray:
    if name == "cage":
        return pred[idx_map["cage_plus"]] + pred[idx_map["cage_minus"]]
    if name == "rampage":
        return pred[idx_map["rampage_plus"]] + pred[idx_map["rampage_minus"]]
    raise ValueError("Unknown combined channel")


def channel_available(mask_row: np.ndarray, idx_map: Dict[str, int], name: str) -> bool:
    if name in ("cage", "rampage"):
        if name == "cage":
            return mask_row[idx_map["cage_plus"]] == 1 and mask_row[idx_map["cage_minus"]] == 1
        return mask_row[idx_map["rampage_plus"]] == 1 and mask_row[idx_map["rampage_minus"]] == 1
    return mask_row[idx_map[name]] == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissues", type=Path, default=Path("/project/deeprna/data/revision/test_samples.txt"))
    parser.add_argument("--train-tissues", type=Path, default=Path("/project/deeprna/data/revision/training_samples.txt"))
    parser.add_argument("--experiments", type=Path, default=Path("/project/deeprna/data/experiments_final.txt"))
    parser.add_argument("--mask-path", type=Path, default=Path("/project/deeprna_data/pretraining_data_final2/experiment_mask.npy"))
    parser.add_argument("--tf-expression", type=Path, default=Path("/project/deeprna_data/pretraining_data_final2/tf_expression.npy"))
    parser.add_argument("--tf-expression-epigept", type=Path, default=Path("/project/deeprna/epigept_directory/motifdata/tf_expression_epigept.npy"))
    parser.add_argument("--dna-fasta", type=Path, default=Path("/project/deeprna_data/borzoi/hg38.ml.fa"))
    parser.add_argument("--regions-bed", type=Path, default=Path("/project/deeprna/data/hg38_sequence_folds_tfexcluded34_merged.bed"))
    parser.add_argument("--binmap", type=Path, default=Path("/project/deeprna_data/revision_data_qn_binned/chrom_bin_map.npz"))
    parser.add_argument("--tissue-binned-dir", type=Path, default=Path("/project/deeprna_data/revision_data_qn_binned"))
    parser.add_argument("--mean-baseline-binned", type=Path, default=Path("/project/deeprna_data/revision_data_qn_binned/mean_baseline_binned.npz"))
    parser.add_argument("--corgi-ckpt", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/big_corgi.pt"))
    parser.add_argument("--corgiplus-rna-qn", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/corgiplus_rna_qn.pt"))
    parser.add_argument("--corgiplus-rna-nofilm", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/corgiplus_rna_nofilm.pt"))
    parser.add_argument("--corgiplusplus-rna", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/corgiplusplus_rna.pt"))
    parser.add_argument("--epigept-ckpt", type=Path, default=Path("/project/deeprna/logs/TensorBoard/epigept_corgi/checkpoints/epigept-epoch=12-val_loss=0.4338.ckpt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tissues", type=int, default=2)
    parser.add_argument("--max-corgi-windows", type=int, default=2)
    parser.add_argument("--max-epi-windows", type=int, default=2)
    parser.add_argument("--max-bins-per-chrom", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tissues = read_ids(args.tissues)
    if args.max_tissues > 0:
        tissues = tissues[: args.max_tissues]
    train_tissues = read_ids(args.train_tissues)

    experiments = load_experiments(args.experiments)
    idx_map = get_channel_indices(experiments)
    mask = load_experiment_mask(args.mask_path)

    tf_expr = np.load(args.tf_expression).astype(np.float32)
    tf_mean = tf_expr[train_tissues].mean(axis=0)
    tf_expr_epi = np.load(args.tf_expression_epigept).astype(np.float32)

    binmap = np.load(args.binmap)
    chroms = [str(c) for c in binmap["chroms"]]
    bin_counts = binmap["bin_counts"].astype(int)
    bin_size = int(binmap["bin_size"][0])
    if bin_size != BIN_SIZE:
        raise ValueError(f"Expected bin size {BIN_SIZE}, got {bin_size}")

    chrom_bins = {chrom: bin_counts[i] for i, chrom in enumerate(chroms)}

    bed_rows = load_bed(args.regions_bed)
    bed_chr = {"chr1": [], "chr10": []}
    for chrom, start, end in bed_rows:
        if chrom in bed_chr:
            bed_chr[chrom].append((start, end))
    interval_bins = {chrom: intervals_to_bins(bed_chr[chrom]) for chrom in bed_chr}

    selected_bins = {chrom: build_selected_bins(interval_bins[chrom]) for chrom in interval_bins}
    if args.max_bins_per_chrom > 0:
        for chrom in selected_bins:
            selected_bins[chrom] = selected_bins[chrom][: args.max_bins_per_chrom]
    total_bins = sum(len(selected_bins[chrom]) for chrom in selected_bins)
    logging.info("Selected %d total bins for test run", total_bins)

    bin_chrom = []
    bin_start = []
    bin_end = []
    bin_index = []
    for chrom in ("chr1", "chr10"):
        for b in selected_bins[chrom]:
            bin_chrom.append(chrom)
            bin_start.append(b * BIN_SIZE)
            bin_end.append((b + 1) * BIN_SIZE)
            bin_index.append(b)
    bin_chrom = np.array(bin_chrom, dtype=object)
    bin_start = np.array(bin_start, dtype=np.int64)
    bin_end = np.array(bin_end, dtype=np.int64)
    bin_index = np.array(bin_index, dtype=np.int64)

    out_index = {chrom: build_bin_index(chrom_bins[chrom], selected_bins[chrom]) for chrom in selected_bins}

    genome = Fasta(str(args.dna_fasta))
    mean_baseline = np.load(args.mean_baseline_binned)

    corgi_model = load_model(str(args.corgi_ckpt), "big_corgi", device)
    corgiplus_rna_qn = load_model(str(args.corgiplus_rna_qn), "corgiplus_rna", device)
    corgiplus_rna_nofilm = load_model(str(args.corgiplus_rna_nofilm), "corgiplus_rna_nofilm", device)
    corgiplusplus_rna = load_model(str(args.corgiplusplus_rna), "corgiplusplus_rna", device)

    epigept_wrapper = EpigeptInferenceWrapper(
        checkpoint_path=args.epigept_ckpt,
        pfm_dir=Path("/project/deeprna/epigept_directory/motif_pfms"),
        tf_csv=Path("/project/deeprna/epigept_directory/tf_expression.csv"),
        device=device,
    )

    corgi_windows_by_chrom = {}
    for chrom in ("chr1", "chr10"):
        windows = windows_from_intervals(interval_bins[chrom], CORGI_TARGET_BINS, chrom_bins[chrom])
        if args.max_corgi_windows > 0:
            windows = windows[: args.max_corgi_windows]
        corgi_windows_by_chrom[chrom] = windows

    epi_windows_by_chrom = {}
    for chrom in ("chr1", "chr10"):
        windows = windows_from_intervals(interval_bins[chrom], EPIGEPT_OUT_BINS_64, chrom_bins[chrom])
        if args.max_epi_windows > 0:
            windows = windows[: args.max_epi_windows]
        epi_windows_by_chrom[chrom] = windows

    results: Dict[str, np.ndarray] = {}

    for tissue_id in tissues:
        logging.info("Processing tissue %d", tissue_id)
        tissue_path = args.tissue_binned_dir / f"tissue_{tissue_id}_binned.npz"
        tissue_npz = np.load(tissue_path)
        channel_indices = tissue_npz["channel_indices"].astype(int)
        local_map = {int(idx): i for i, idx in enumerate(channel_indices)}
        tissue_mask = mask[tissue_id]

        for ch in CHANNELS_INTEREST:
            if not channel_available(tissue_mask, idx_map, ch):
                continue
            for model_name in (
                "mean_baseline",
                "corgi_avg_tf",
                "corgi_tissue_tf",
                "corgiplus_rna_qn",
                "corgiplus_rna_nofilm",
                "corgiplusplus_rna",
                "epigept",
            ):
                key = f"{ch}__t{tissue_id}__{model_name}"
                results[key] = np.full(total_bins, np.nan, dtype=np.float32)

        for chrom in ("chr1", "chr10"):
            chrom_bins_n = chrom_bins[chrom]
            out_idx = out_index[chrom]

            mb = mean_baseline[chrom]
            if mb.shape[0] != chrom_bins_n:
                raise ValueError(f"Mean baseline bins mismatch for {chrom}")

            for ch in CHANNELS_INTEREST:
                if not channel_available(tissue_mask, idx_map, ch):
                    continue
                if ch in ("cage", "rampage"):
                    plus = f"{ch}_plus"
                    minus = f"{ch}_minus"
                    pred_mb = mb[:, idx_map[plus]] + mb[:, idx_map[minus]]
                else:
                    pred_mb = mb[:, idx_map[ch]]

                key_mb = f"{ch}__t{tissue_id}__mean_baseline"
                sel = out_idx >= 0
                results[key_mb][out_idx[sel]] = pred_mb[sel].astype(np.float32)

        for chrom in ("chr1", "chr10"):
            chrom_bins_n = chrom_bins[chrom]
            out_idx = out_index[chrom]
            corgi_windows = corgi_windows_by_chrom[chrom]

            for start_bin in corgi_windows:
                input_start = start_bin - CORGI_CROP_BINS
                input_end = start_bin + CORGI_TARGET_BINS + CORGI_CROP_BINS
                start_bp = input_start * BIN_SIZE
                end_bp = input_end * BIN_SIZE
                seq = get_padded_sequence(genome, chrom, start_bp, end_bp)
                if len(seq) != SEQ_LENGTH:
                    raise ValueError("DNA length mismatch for corgi window")
                n_count = count_invalid_bases(seq)
                if n_count > 0:
                    logging.info("Found %d ambiguous bases in %s:%d-%d", n_count, chrom, start_bp, end_bp)
                dna_onehot = one_hot_encode(seq)
                dna_t = torch.from_numpy(dna_onehot).unsqueeze(0).to(device)

                tissue_chrom = tissue_npz[chrom]
                if tissue_chrom.shape[0] != chrom_bins_n:
                    raise ValueError(f"Tissue bins mismatch for {chrom}")
                mb = mean_baseline[chrom]
                if mb.shape[0] != chrom_bins_n:
                    raise ValueError(f"Mean baseline bins mismatch for {chrom}")

                rna_plus_idx = idx_map["rna_total_plus"]
                rna_minus_idx = idx_map["rna_total_minus"]
                if rna_plus_idx not in local_map or rna_minus_idx not in local_map:
                    raise ValueError("RNA tracks missing from tissue data")

                tissue_chunk = slice_bins_with_pad(
                    tissue_chrom,
                    input_start,
                    input_end,
                    label=f"tissue_bins:{chrom}:{input_start}-{input_end}",
                )
                mb_chunk = slice_bins_with_pad(
                    mb,
                    input_start,
                    input_end,
                    label=f"mean_baseline:{chrom}:{input_start}-{input_end}",
                )
                rna_tracks = np.stack(
                    [
                        tissue_chunk[:, local_map[rna_plus_idx]],
                        tissue_chunk[:, local_map[rna_minus_idx]],
                    ],
                    axis=0,
                ).astype(np.float32)
                mean_baseline_chunk = mb_chunk.T.astype(np.float32)

                aux_rna = torch.from_numpy(np.concatenate([rna_tracks, mean_baseline_chunk], axis=0)).unsqueeze(0).to(device)
                delta_rna = rna_tracks - mean_baseline_chunk[[rna_plus_idx, rna_minus_idx], :]
                aux_plus = torch.from_numpy(np.concatenate([rna_tracks, mean_baseline_chunk, delta_rna], axis=0)).unsqueeze(0).to(device)
                trans_tissue = torch.from_numpy(tf_expr[tissue_id]).unsqueeze(0).to(device)
                trans_avg = torch.from_numpy(tf_mean).unsqueeze(0).to(device)

                pred_corgi_avg = pred_tta(corgi_model, dna_t, trans_avg, "big_corgi").cpu().numpy()[0]
                pred_corgi_tis = pred_tta(corgi_model, dna_t, trans_tissue, "big_corgi").cpu().numpy()[0]
                pred_rna_qn = pred_tta(corgiplus_rna_qn, dna_t, trans_tissue, "corgiplus_rna", aux=aux_rna).cpu().numpy()[0]
                pred_rna_nofilm = pred_tta(corgiplus_rna_nofilm, dna_t, None, "corgiplus_rna_nofilm", aux=aux_rna).cpu().numpy()[0]
                pred_plus = pred_tta(corgiplusplus_rna, dna_t, trans_tissue, "corgiplusplus_rna", aux=aux_plus).cpu().numpy()[0]

                out_slice = out_idx[start_bin : start_bin + CORGI_TARGET_BINS]
                sel = out_slice >= 0
                for ch in CHANNELS_INTEREST:
                    if not channel_available(tissue_mask, idx_map, ch):
                        continue
                    if ch in ("cage", "rampage"):
                        val_corgi_avg = build_combined(pred_corgi_avg, idx_map, ch)
                        val_corgi_tis = build_combined(pred_corgi_tis, idx_map, ch)
                        val_rna_qn = build_combined(pred_rna_qn, idx_map, ch)
                        val_rna_nofilm = build_combined(pred_rna_nofilm, idx_map, ch)
                        val_plus = build_combined(pred_plus, idx_map, ch)
                    else:
                        idx = idx_map[ch]
                        val_corgi_avg = pred_corgi_avg[idx]
                        val_corgi_tis = pred_corgi_tis[idx]
                        val_rna_qn = pred_rna_qn[idx]
                        val_rna_nofilm = pred_rna_nofilm[idx]
                        val_plus = pred_plus[idx]

                    results[f"{ch}__t{tissue_id}__corgi_avg_tf"][out_slice[sel]] = val_corgi_avg[sel]
                    results[f"{ch}__t{tissue_id}__corgi_tissue_tf"][out_slice[sel]] = val_corgi_tis[sel]
                    results[f"{ch}__t{tissue_id}__corgiplus_rna_qn"][out_slice[sel]] = val_rna_qn[sel]
                    results[f"{ch}__t{tissue_id}__corgiplus_rna_nofilm"][out_slice[sel]] = val_rna_nofilm[sel]
                    results[f"{ch}__t{tissue_id}__corgiplusplus_rna"][out_slice[sel]] = val_plus[sel]

        for chrom in ("chr1", "chr10"):
            chrom_bins_n = chrom_bins[chrom]
            out_idx = out_index[chrom]
            epi_windows = epi_windows_by_chrom[chrom]

            for start_bin in epi_windows:
                start_bp = start_bin * BIN_SIZE
                end_bp = start_bp + EPIGEPT_BP
                seq = genome[chrom][start_bp:end_bp].seq
                if len(seq) != EPIGEPT_BP:
                    raise ValueError("DNA length mismatch for epigept window")
                dna_onehot = one_hot_encode(seq)
                epi_pred = epigept_wrapper(dna_onehot, tf_expr_epi[tissue_id]).cpu().numpy()
                epi_pred_64 = np.repeat(epi_pred, 2, axis=0)

                out_slice = out_idx[start_bin : start_bin + EPIGEPT_OUT_BINS_64]
                sel = out_slice >= 0
                for ch in CHANNELS_INTEREST:
                    if not channel_available(tissue_mask, idx_map, ch):
                        continue
                    if ch in ("cage", "rampage"):
                        val = build_combined(epi_pred_64.T, idx_map, ch)
                    else:
                        val = epi_pred_64[:, idx_map[ch]]
                    results[f"{ch}__t{tissue_id}__epigept"][out_slice[sel]] = val[sel]

    meta = {
        "bin_chrom": bin_chrom,
        "bin_start": bin_start,
        "bin_end": bin_end,
        "bin_index": bin_index,
        "channels": np.array(CHANNELS_INTEREST, dtype=object),
        "tissues": np.array(tissues, dtype=np.int32),
    }
    meta.update(results)
    np.savez(args.out, **meta)
    logging.info("Saved output to %s", args.out)


if __name__ == "__main__":
    main()