"""Region-based imputation evaluation using non-overlapping 524kb regions.

Builds NPZ with:
- corgi_family_preds: (n_models, n_regions, n_tissues, 6144, 22)
- epigept_preds: (1, n_regions, n_tissues, 3000, 22)
- mean_baseline: (1, n_regions, 6144, 22)
- ground_truth: (1, n_regions, n_tissues, 6144, 22)
Plus metadata (region coords, indices, tissues, model names, experiments).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from corgi.utils import load_experiment_mask
from corgiplus_delta_bench_tta import load_model, rc_auxiliary
from benchmark_utils import dna_rc, predictions_rc, shift_dna
from epigept_inference_wrapper import EpigeptInferenceWrapper

SEQ_LENGTH = 524_288
CROP_BINS = 1024
TARGET_BINS = 6144
AUX_BINS = 8192
BIN_SIZE = 64
EPIGEPT_BP = 128_000
EPIGEPT_BINS = 1000
EPIGEPT_TOTAL_BINS = 3000

MODEL_SPECS = [
    ("corgi_avg_baseline", "big_corgi"),
    ("corgi", "big_corgi"),
    ("corgi_impute", "corgiplus_rna"),
    ("corgi_impute_nofilm", "corgiplus_rna_nofilm"),
    ("corgiplus", "corgiplusplus_rna"),
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
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            chrom, start_s, end_s, *_ = line.rstrip().split("\t")
            rows.append((chrom, int(start_s), int(end_s), idx))
    return rows


def select_nonoverlapping(regions: List[Tuple[str, int, int, int]]) -> List[Tuple[str, int, int, int]]:
    by_chr: Dict[str, List[Tuple[str, int, int, int]]] = {"chr1": [], "chr10": []}
    for chrom, start, end, idx in regions:
        if chrom in by_chr:
            by_chr[chrom].append((chrom, start, end, idx))
    selected: List[Tuple[str, int, int, int]] = []
    for chrom in ("chr1", "chr10"):
        cur = sorted(by_chr[chrom], key=lambda x: (x[1], x[2]))
        last_end = None
        for item in cur:
            _, start, end, _ = item
            if last_end is None or start >= last_end:
                selected.append(item)
                last_end = end
    return selected


def parse_shifts(text: str) -> List[int]:
    if not text:
        return [0]
    return [int(x) for x in text.split(",") if x.strip()]


def pred_tta_custom(
    model: torch.nn.Module,
    dna_seq: torch.Tensor,
    trans_reg: torch.Tensor | None,
    model_type: str,
    shifts: List[int],
    use_rc: bool,
    aux: torch.Tensor | None = None,
) -> torch.Tensor:
    device_type = dna_seq.device.type
    aux_rc = rc_auxiliary(aux, model_type) if (aux is not None and use_rc) else None

    def forward(seq_in: torch.Tensor, aux_in: torch.Tensor | None) -> torch.Tensor:
        if model_type == "big_corgi":
            pred = model(seq_in, trans_reg)
            return torch.clamp(pred, min=0.0)
        if model_type == "corgiplus_rna":
            return model(seq_in, aux_in.permute(0, 2, 1), trans_reg)
        if model_type == "corgiplus_rna_nofilm":
            return model(seq_in, aux_in.permute(0, 2, 1), None)
        if model_type == "corgiplusplus_rna":
            return model(seq_in, aux_in.permute(0, 2, 1), trans_reg)
        raise ValueError(f"Unsupported model type: {model_type}")

    preds: List[torch.Tensor] = []
    with torch.no_grad():
        with torch.autocast(device_type, dtype=torch.bfloat16):
            for shift in shifts:
                seq_shift = shift_dna(dna_seq, shift)
                pred = forward(seq_shift, aux)
                preds.append(pred)

                if use_rc:
                    seq_rc = dna_rc(seq_shift)
                    pred_rc = forward(seq_rc, aux_rc)
                    pred_rc = predictions_rc(pred_rc)
                    preds.append(pred_rc)

    preds = [p.float() for p in preds]
    out = torch.stack(preds, dim=0).mean(dim=0)
    out_len = out.shape[-1]
    if out_len == TARGET_BINS:
        return out.squeeze(0)
    if out_len > TARGET_BINS:
        start = (out_len - TARGET_BINS) // 2
        end = start + TARGET_BINS
        out = out[..., start:end]
        return out.squeeze(0)
    raise ValueError(f"Model output bins ({out_len}) smaller than TARGET_BINS ({TARGET_BINS})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", type=Path, default=Path("/project/deeprna/data/hg38_sequence_folds_tfexcluded34.bed"))
    parser.add_argument("--tissues", type=Path, default=Path("/project/deeprna/data/revision/test_samples.txt"))
    parser.add_argument("--train-tissues", type=Path, default=Path("/project/deeprna/data/revision/training_samples.txt"))
    parser.add_argument("--experiments", type=Path, default=Path("/project/deeprna/data/experiments_final.txt"))
    parser.add_argument("--mask-path", type=Path, default=Path("/project/deeprna_data/pretraining_data_final2/experiment_mask.npy"))
    parser.add_argument("--tf-expression", type=Path, default=Path("/project/deeprna_data/pretraining_data_final2/tf_expression.npy"))
    parser.add_argument("--tf-expression-epigept", type=Path, default=Path("/project/deeprna/epigept_directory/motifdata/tf_expression_epigept.npy"))
    parser.add_argument("--dna-onehot", type=Path, default=Path("/project/deeprna_data/pretraining_data_final2/dna_onehot.npy"))
    parser.add_argument("--tissue-npy-dir", type=Path, default=Path("/project/deeprna_data/revision_data_qn_parallel"))
    parser.add_argument("--mean-baseline", type=Path, default=Path("/project/deeprna/data/revision/training_baseline_signal_qn.npy"))
    parser.add_argument("--corgi-ckpt", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/big_corgi.pt"))
    parser.add_argument("--corgiplus-rna-qn", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/corgiplus_rna_qn.pt"))
    parser.add_argument("--corgiplus-rna-nofilm", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/corgiplus_rna_nofilm.pt"))
    parser.add_argument("--corgiplusplus-rna", type=Path, default=Path("/project/deeprna_data/models/revision/imputation/corgiplusplus_rna.pt"))
    parser.add_argument("--epigept-ckpt", type=Path, default=Path("/project/deeprna/logs/TensorBoard/epigept_corgi/checkpoints/epigept-epoch=12-val_loss=0.4338.ckpt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tta-shifts", type=str, default="-2,0,2")
    parser.add_argument("--no-rc", action="store_true")
    parser.add_argument("--tissue-start", type=int, default=None, help="Start index in tissue list (inclusive)")
    parser.add_argument("--tissue-end", type=int, default=None, help="End index in tissue list (exclusive)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tissues = read_ids(args.tissues)
    if args.tissue_start is not None or args.tissue_end is not None:
        start = args.tissue_start if args.tissue_start is not None else 0
        end = args.tissue_end if args.tissue_end is not None else len(tissues)
        tissues = tissues[start:end]
        logging.info("Using tissue index slice [%d:%d] (%d tissues)", start, end, len(tissues))
    train_tissues = read_ids(args.train_tissues)

    experiments = load_experiments(args.experiments)
    idx_map = {name: idx for idx, name in enumerate(experiments)}
    mask = load_experiment_mask(args.mask_path)

    tf_expr = np.load(args.tf_expression).astype(np.float32)
    tf_mean = tf_expr[train_tissues].mean(axis=0)
    tf_expr_epi = np.load(args.tf_expression_epigept).astype(np.float32)

    bed_rows = load_bed(args.bed)
    logging.info("Loaded %d total BED regions", len(bed_rows))
    bed_rows = [r for r in bed_rows if r[0] in ("chr1", "chr10")]
    logging.info("Filtered to chr1/chr10: %d regions", len(bed_rows))
    bed_rows = sorted(bed_rows, key=lambda x: (x[0], x[1], x[2]))
    selected_regions = select_nonoverlapping(bed_rows)
    logging.info("Selected %d non-overlapping regions", len(selected_regions))

    if not selected_regions:
        raise ValueError("No non-overlapping regions selected for chr1/chr10")

    region_coords = [(c, s, e) for c, s, e, _ in selected_regions]
    region_indices = [idx for _, _, _, idx in selected_regions]
    chr1_count = sum(1 for c, _, _ in region_coords if c == "chr1")
    chr10_count = sum(1 for c, _, _ in region_coords if c == "chr10")
    logging.info("Non-overlapping regions per chrom: chr1=%d chr10=%d", chr1_count, chr10_count)

    mean_baseline = np.load(args.mean_baseline, mmap_mode="r")
    if mean_baseline.ndim != 3 or mean_baseline.shape[1] != AUX_BINS:
        raise ValueError("Expected mean baseline shape (n_regions, 8192, 22)")
    logging.info("Mean baseline shape: %s", mean_baseline.shape)

    dna_onehot = np.load(args.dna_onehot, mmap_mode="r")
    if dna_onehot.ndim != 3 or dna_onehot.shape[1] != SEQ_LENGTH or dna_onehot.shape[2] != 4:
        raise ValueError("Expected dna_onehot shape (n_regions, 524288, 4)")
    logging.info("DNA one-hot shape: %s", dna_onehot.shape)

    shifts = parse_shifts(args.tta_shifts)
    use_rc = not args.no_rc

    # Models
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

    n_regions = len(region_coords)
    n_tissues = len(tissues)
    n_models = len(MODEL_SPECS)
    logging.info("n_regions=%d n_tissues=%d n_models=%d", n_regions, n_tissues, n_models)
    logging.info("TTA shifts: %s  RC: %s", shifts, use_rc)

    corgi_preds = np.zeros((n_models, n_regions, n_tissues, TARGET_BINS, 22), dtype=np.float32)
    mean_baseline_out = np.zeros((1, n_regions, TARGET_BINS, 22), dtype=np.float32)
    gt_out = np.full((1, n_regions, n_tissues, TARGET_BINS, 22), np.nan, dtype=np.float32)
    epigept_out = np.zeros((1, n_regions, n_tissues, EPIGEPT_TOTAL_BINS, 22), dtype=np.float32)

    # Precompute mean baseline (center 6144 bins) per region
    mean_baseline_out[0] = mean_baseline[region_indices, CROP_BINS:CROP_BINS + TARGET_BINS, :]

    model_names = [name for name, _ in MODEL_SPECS]

    for t_i, tissue_id in enumerate(tissues):
        logging.info("Processing tissue %d (%d/%d)", tissue_id, t_i + 1, n_tissues)
        tissue_path = args.tissue_npy_dir / f"tissue_{tissue_id}.npy"
        tissue_arr = np.load(tissue_path, mmap_mode="r")
        if tissue_arr.ndim != 3 or tissue_arr.shape[1] != AUX_BINS:
            raise ValueError("Expected tissue array shape (n_regions, 8192, n_avail)")

        avail = np.where(mask[tissue_id] == 1)[0]
        if tissue_arr.shape[2] != len(avail):
            raise ValueError("Tissue channel count does not match experiment mask")
        avail_map = {int(exp_idx): local_idx for local_idx, exp_idx in enumerate(avail)}

        rna_plus_idx = idx_map["rna_total_plus"]
        rna_minus_idx = idx_map["rna_total_minus"]
        if rna_plus_idx not in avail_map or rna_minus_idx not in avail_map:
            raise ValueError("RNA tracks missing for corgiplus inputs")
        rna_plus_local = avail_map[rna_plus_idx]
        rna_minus_local = avail_map[rna_minus_idx]

        trans_tissue = torch.from_numpy(tf_expr[tissue_id]).unsqueeze(0).to(device)
        trans_avg = torch.from_numpy(tf_mean).unsqueeze(0).to(device)
        tf_epi = tf_expr_epi[tissue_id]

        # Ground truth
        logging.info("  Loading ground truth")
        for r_i, region_idx in enumerate(region_indices):
            gt_slice = tissue_arr[region_idx, CROP_BINS:CROP_BINS + TARGET_BINS, :]
            for local_idx, global_idx in enumerate(avail):
                gt_out[0, r_i, t_i, :, global_idx] = gt_slice[:, local_idx]

        # Model inference
        logging.info("  Running model inference")
        for r_i, (chrom, start, end) in enumerate(region_coords):
            if r_i % 100 == 0:
                logging.info("    Region %d/%d: %s:%d-%d", r_i + 1, n_regions, chrom, start, end)
            if (end - start) != SEQ_LENGTH:
                raise ValueError("Region length does not match 524288 bp")
            region_idx = region_indices[r_i]
            dna_region = np.ascontiguousarray(dna_onehot[region_idx], dtype=np.float32)
            dna_t = torch.from_numpy(dna_region).unsqueeze(0).to(device)

            region_mb = mean_baseline[region_idx]  # (8192, 22)
            region_tissue = tissue_arr[region_idx]
            rna_tracks = np.stack(
                [
                    region_tissue[:, rna_plus_local],
                    region_tissue[:, rna_minus_local],
                ],
                axis=0,
            ).astype(np.float32)
            mean_baseline_chunk = region_mb.T.astype(np.float32)

            aux_rna = torch.from_numpy(np.concatenate([rna_tracks, mean_baseline_chunk], axis=0)).unsqueeze(0).to(device)
            delta_rna = rna_tracks - mean_baseline_chunk[[rna_plus_idx, rna_minus_idx], :]
            aux_plus = torch.from_numpy(np.concatenate([rna_tracks, mean_baseline_chunk, delta_rna], axis=0)).unsqueeze(0).to(device)

            # Model outputs
            pred_corgi_avg = pred_tta_custom(corgi_model, dna_t, trans_avg, "big_corgi", shifts, use_rc)
            pred_corgi_tis = pred_tta_custom(corgi_model, dna_t, trans_tissue, "big_corgi", shifts, use_rc)
            pred_rna_qn = pred_tta_custom(corgiplus_rna_qn, dna_t, trans_tissue, "corgiplus_rna", shifts, use_rc, aux=aux_rna)
            pred_rna_nofilm = pred_tta_custom(corgiplus_rna_nofilm, dna_t, None, "corgiplus_rna_nofilm", shifts, use_rc, aux=aux_rna)
            pred_plus = pred_tta_custom(corgiplusplus_rna, dna_t, trans_tissue, "corgiplusplus_rna", shifts, use_rc, aux=aux_plus)

            preds = [pred_corgi_avg, pred_corgi_tis, pred_rna_qn, pred_rna_nofilm, pred_plus]
            for m_i, pred in enumerate(preds):
                corgi_preds[m_i, r_i, t_i] = pred.transpose(1, 0).cpu().numpy().astype(np.float32)

            # Epigept: central 384000 bp -> 3 chunks of 128000
            center_start = (SEQ_LENGTH - 384_000) // 2
            epi_bins = []
            for chunk_i in range(3):
                s = center_start + chunk_i * EPIGEPT_BP
                e = s + EPIGEPT_BP
                chunk_onehot = np.ascontiguousarray(dna_region[s:e], dtype=np.float32)
                if chunk_onehot.shape[0] != EPIGEPT_BP:
                    raise ValueError("Epigept chunk length mismatch")
                epi_pred = epigept_wrapper(chunk_onehot, tf_epi).cpu().numpy()
                epi_bins.append(epi_pred)
            epi_concat = np.concatenate(epi_bins, axis=0)  # (3000, 22)
            epigept_out[0, r_i, t_i] = epi_concat.astype(np.float32)

    meta = {
        "region_coords": np.array(region_coords, dtype=object),
        "region_indices": np.array(region_indices, dtype=np.int32),
        "tissues": np.array(tissues, dtype=np.int32),
        "model_names": np.array(model_names, dtype=object),
        "experiments": np.array(experiments, dtype=object),
        "mean_baseline": mean_baseline_out,
        "ground_truth": gt_out,
        "corgi_family_preds": corgi_preds,
        "epigept_preds": epigept_out,
        "tta_shifts": np.array(shifts, dtype=np.int32),
        "tta_rc": np.array([use_rc], dtype=bool),
    }

    np.savez(args.out, **meta)
    logging.info("Saved output to %s", args.out)


if __name__ == "__main__":
    main()
