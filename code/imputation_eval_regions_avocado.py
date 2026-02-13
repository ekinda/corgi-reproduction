#!/usr/bin/env python
"""Avocado predictions for non-overlapping 524kb regions (chr1/chr10).

Outputs NPZ with:
- avocado_preds: (1, n_regions, n_tissues, 6144, 22)
- region_coords, region_indices, tissues, experiments
"""
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import numpy as np

# Theano/keras backend setup (override via env if desired)
os.environ.setdefault("KERAS_BACKEND", "theano")
os.environ.setdefault(
    "THEANO_FLAGS",
    "device=cuda,floatX=float32,optimizer=fast_run,gpuarray.preallocate=0.8,exception_verbosity=high",
)

import theano  # noqa: E402
from avocado import Avocado  # noqa: E402

SEQ_LENGTH = 524288
BIN_SIZE = 64
CROP_BINS = 1024
TARGET_BINS = 6144

RAW_ASSAYS = [
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
]
COMBINED_ASSAYS = ["cage", "rampage"]


def read_ids(path):
    handle = open(path, "r")
    try:
        return [int(x) for x in handle.read().strip().split() if x.strip()]
    finally:
        handle.close()


def load_experiments(path):
    handle = open(path, "r")
    try:
        return [x.strip() for x in handle.read().strip().split() if x.strip()]
    finally:
        handle.close()


def load_experiment_mask(mask_path):
    data = np.load(mask_path)
    tissue_mask_dict = {}
    for t_id, mask in enumerate(data):
        tissue_mask_dict[t_id] = mask
    return tissue_mask_dict


def load_bed(path):
    rows = []
    handle = open(path, "r")
    try:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            parts = line.rstrip().split("\t")
            rows.append((parts[0], int(parts[1]), int(parts[2]), idx))
    finally:
        handle.close()
    return rows


def select_nonoverlapping(regions):
    by_chr = {"chr1": [], "chr10": []}
    for chrom, start, end, idx in regions:
        if chrom in by_chr:
            by_chr[chrom].append((chrom, start, end, idx))
    selected = []
    for chrom in ("chr1", "chr10"):
        cur = sorted(by_chr[chrom], key=lambda x: (x[1], x[2]))
        last_end = None
        for item in cur:
            _, start, end, _ = item
            if last_end is None or start >= last_end:
                selected.append(item)
                last_end = end
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", default="/project/deeprna/data/hg38_sequence_folds_tfexcluded34.bed")
    parser.add_argument("--tissues", default="/project/deeprna/data/revision/test_samples.txt")
    parser.add_argument("--experiments", default="/project/deeprna/data/experiments_final.txt")
    parser.add_argument("--mask-path", default="/project/deeprna_data/pretraining_data_final2/experiment_mask.npy")
    parser.add_argument("--avocado-dir", default="/project/deeprna/data/revision/avocado")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tissue-start", type=int, default=None)
    parser.add_argument("--tissue-end", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    experiments = load_experiments(args.experiments)
    idx_map = dict((name, i) for i, name in enumerate(experiments))
    mask = load_experiment_mask(args.mask_path)

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

    tissues = read_ids(args.tissues)
    if args.tissue_start is not None or args.tissue_end is not None:
        start = args.tissue_start if args.tissue_start is not None else 0
        end = args.tissue_end if args.tissue_end is not None else len(tissues)
        tissues = tissues[start:end]
        logging.info("Using tissue index slice [%d:%d] (%d tissues)", start, end, len(tissues))

    n_regions = len(region_coords)
    n_tissues = len(tissues)

    avocado_preds = np.full((1, n_regions, n_tissues, TARGET_BINS, 22), np.nan, dtype=np.float32)

    avocado_chr1 = Avocado.load(os.path.join(args.avocado_dir, "avocado_chr1_epoch_81"))
    avocado_chr10 = Avocado.load(os.path.join(args.avocado_dir, "avocado_chr10_epoch_67"))

    for t_i, tissue_id in enumerate(tissues):
        logging.info("Processing tissue %d (%d/%d)", tissue_id, t_i + 1, n_tissues)
        tissue_mask = mask[tissue_id]
        available_assays = []
        for assay in RAW_ASSAYS:
            if assay in idx_map and tissue_mask[idx_map[assay]] == 1:
                available_assays.append(assay)
        for assay in COMBINED_ASSAYS:
            if assay == "cage":
                needed = ["cage_plus", "cage_minus"]
            else:
                needed = ["rampage_plus", "rampage_minus"]
            if all(tissue_mask[idx_map[name]] == 1 for name in needed):
                available_assays.append(assay)
        logging.info("Available assays for tissue %d: %s", tissue_id, ", ".join(available_assays))

        for chrom in ("chr1", "chr10"):
            model = avocado_chr1 if chrom == "chr1" else avocado_chr10

            # Preload assay predictions for this tissue+chrom
            pred_cache = {}
            for assay in available_assays:
                pred_cache[assay] = model.predict(tissue_id, assay)

            # Assign region slices
            for r_i, (r_chrom, start, end) in enumerate(region_coords):
                if r_chrom != chrom:
                    continue
                if (end - start) != SEQ_LENGTH:
                    raise ValueError("Region length does not match 524288 bp")
                start_bin = (start // BIN_SIZE) + CROP_BINS
                end_bin = start_bin + TARGET_BINS

                for assay, pred in pred_cache.items():
                    if pred.shape[0] < end_bin:
                        raise ValueError("Avocado prediction bins shorter than requested window")
                    slice_vals = pred[start_bin:end_bin].astype(np.float32)

                    if assay in COMBINED_ASSAYS:
                        if assay == "cage":
                            idx_plus = idx_map["cage_plus"]
                            idx_minus = idx_map["cage_minus"]
                        else:
                            idx_plus = idx_map["rampage_plus"]
                            idx_minus = idx_map["rampage_minus"]
                        avocado_preds[0, r_i, t_i, :, idx_plus] = slice_vals
                        avocado_preds[0, r_i, t_i, :, idx_minus] = slice_vals
                    else:
                        avocado_preds[0, r_i, t_i, :, idx_map[assay]] = slice_vals

    meta = {
        "region_coords": np.array(region_coords, dtype=object),
        "region_indices": np.array(region_indices, dtype=np.int32),
        "tissues": np.array(tissues, dtype=np.int32),
        "experiments": np.array(experiments, dtype=object),
        "avocado_preds": avocado_preds,
        "note": np.array([
            "Avocado outputs for cage/rampage are mapped to both plus and minus channels."
        ], dtype=object),
    }

    np.savez(args.out, **meta)
    logging.info("Saved output to %s", args.out)


if __name__ == "__main__":
    main()
