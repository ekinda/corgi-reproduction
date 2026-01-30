"""Avocado-only imputation on chr1/chr10 bins (Python 3, no pathlib/f-strings).

Outputs an NPZ with keys like: channel__t{tissue}__avocado
and bin metadata (bin_chrom/bin_start/bin_end/channels/tissues).
"""
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys

import numpy as np

# Theano/keras backend setup (override via env if desired)
os.environ.setdefault("KERAS_BACKEND", "theano")
os.environ.setdefault(
    "THEANO_FLAGS",
    "device=cuda,floatX=float32,optimizer=fast_run,gpuarray.preallocate=0.8,exception_verbosity=high",
)

import theano  # noqa: E402
from avocado import Avocado  # noqa: E402

def load_experiment_mask(mask_path):
    """
    Loads the available experiment mask from a numpy array
    and converts it to a dictionary of form {tissue_id:mask, where mask is a numpy array}
    """
    data = np.load(mask_path)

    tissue_mask_dict = {}
    for t_id, mask in enumerate(data):
        tissue_mask_dict[t_id] = mask
    return tissue_mask_dict

BIN_SIZE = 64

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


def load_bed(path):
    rows = []
    handle = open(path, "r")
    try:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip().split("\t")
            rows.append((parts[0], int(parts[1]), int(parts[2])))
    finally:
        handle.close()
    return rows


def merge_intervals(intervals):
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


def intervals_to_bins(intervals):
    out = []
    for start, end in intervals:
        start_bin = start // BIN_SIZE
        end_bin = int(np.ceil(float(end) / float(BIN_SIZE)))
        out.append((start_bin, end_bin))
    return out


def build_selected_bins(interval_bins):
    bins = []
    for start_bin, end_bin in interval_bins:
        bins.extend(list(range(start_bin, end_bin)))
    return bins


def build_bin_index(chrom_bins, selected_bins):
    idx = np.full(chrom_bins, -1, dtype=np.int64)
    for out_i, bin_i in enumerate(selected_bins):
        idx[bin_i] = out_i
    return idx


def channel_available(mask_row, idx_map, name):
    if name == "cage":
        return mask_row[idx_map["cage_plus"]] == 1 and mask_row[idx_map["cage_minus"]] == 1
    if name == "rampage":
        return mask_row[idx_map["rampage_plus"]] == 1 and mask_row[idx_map["rampage_minus"]] == 1
    return mask_row[idx_map[name]] == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissues", default="/project/deeprna/data/revision/test_samples.txt")
    parser.add_argument("--experiments", default="/project/deeprna/data/experiments_final.txt")
    parser.add_argument("--mask-path", default="/project/deeprna_data/pretraining_data_final2/experiment_mask.npy")
    parser.add_argument("--regions-bed", default="/project/deeprna/data/hg38_sequence_folds_tfexcluded34_merged.bed")
    parser.add_argument("--binmap", default="/project/deeprna_data/revision_data_qn_binned/chrom_bin_map.npz")
    parser.add_argument("--avocado-dir", default="/project/deeprna/data/revision/avocado")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    tissues = read_ids(args.tissues)
    experiments = load_experiments(args.experiments)
    idx_map = dict((name, i) for i, name in enumerate(experiments))
    mask = load_experiment_mask(args.mask_path)

    binmap = np.load(args.binmap)
    chroms = [str(c) for c in binmap["chroms"]]
    bin_counts = binmap["bin_counts"].astype(int)
    bin_size = int(binmap["bin_size"][0])
    if bin_size != BIN_SIZE:
        raise ValueError("Expected bin size {0}, got {1}".format(BIN_SIZE, bin_size))

    chrom_bins = dict((chrom, bin_counts[i]) for i, chrom in enumerate(chroms))

    bed_rows = load_bed(args.regions_bed)
    bed_chr = {"chr1": [], "chr10": []}
    for chrom, start, end in bed_rows:
        if chrom in bed_chr:
            bed_chr[chrom].append((start, end))

    interval_bins = dict((chrom, intervals_to_bins(bed_chr[chrom])) for chrom in bed_chr)

    selected_bins = dict((chrom, build_selected_bins(interval_bins[chrom])) for chrom in interval_bins)
    total_bins = sum(len(selected_bins[chrom]) for chrom in selected_bins)

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

    out_index = dict((chrom, build_bin_index(chrom_bins[chrom], selected_bins[chrom])) for chrom in selected_bins)

    avocado_chr1 = Avocado.load(os.path.join(args.avocado_dir, "avocado_chr1_epoch_81"))
    avocado_chr10 = Avocado.load(os.path.join(args.avocado_dir, "avocado_chr10_epoch_67"))

    results = {}
    for tissue_id in tissues:
        logging.info("Processing tissue %d", tissue_id)
        tissue_mask = mask[tissue_id]
        for ch in CHANNELS_INTEREST:
            if not channel_available(tissue_mask, idx_map, ch):
                continue
            key = "{0}__t{1}__avocado".format(ch, tissue_id)
            results[key] = np.full(total_bins, np.nan, dtype=np.float32)

        for chrom in ("chr1", "chr10"):
            chrom_bins_n = chrom_bins[chrom]
            out_idx = out_index[chrom]
            avocado_model = avocado_chr1 if chrom == "chr1" else avocado_chr10

            for ch in CHANNELS_INTEREST:
                if not channel_available(tissue_mask, idx_map, ch):
                    continue
                if ch == "cage" or ch == "rampage":
                    pred = avocado_model.predict(tissue_id, ch)
                else:
                    pred = avocado_model.predict(tissue_id, ch)

                if pred.shape[0] != chrom_bins_n:
                    raise ValueError("Avocado output bins mismatch")

                key = "{0}__t{1}__avocado".format(ch, tissue_id)
                sel = out_idx >= 0
                results[key][out_idx[sel]] = pred[sel].astype(np.float32)

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