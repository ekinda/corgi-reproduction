#!/usr/bin/env python
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
import glob
import logging
import numpy as np
import subprocess
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
import h5py
from collections import defaultdict
from benchmark_utils import transform_params, special_atac_params, transform_softclip, transform_scale, transform_scale_special
from benchmark_utils import undo_squashed_scale, parse_bed_file_with_coords, one_hot_encode, load_genome, tile_region, tile_regions, crop_center
from benchmark_utils import dna_rc, predictions_rc, shift_dna, process_coverage, fast_bin

# --- Global parameters ---
EXPORT_TO_BIGWIG = True
PRED_SEQ_LENGTH = 524288      # tile length for predictions (bp)
PRED_STRIDE = 393216          # stride for predictions (bp)
CROP = 65536                 # offset to crop from both left and right

# Tiling for ground truth:
GT_SEQ_LENGTH = 393216        # tile length for ground truth extraction (bp)
BINSIZE = 64                # bin size in bp
TARGET_BINS = 6144          # number of bins required
assert GT_SEQ_LENGTH // BINSIZE == TARGET_BINS

# File paths
BED_FILE = REPO_ROOT / "data" / "figure2" / "fold3_notf_merged.bed"
GENOME_FASTA = REPO_ROOT / "data" / "hg38.ml.fa"
GT_DATA_DIR = REPO_ROOT / "data" / "ground_truth"  # TODO: point to directory containing per-tissue .h5 files
TF_EXP_FILE = REPO_ROOT / "data" / "figure2" / "tf_expression.npy"
CHROM_SIZES_FILE = REPO_ROOT / "data" / "hg38.chrom.sizes"
OUTDIR = REPO_ROOT / "data" / "figure2" / "cross_both_tta"
# (expecting GT files to be found in: GT_DATA_DIR/{tissue_id}/*.h5,
# and that for each experiment the filename starts with the borzoi_track_id)

# Model checkpoint
GRT_CHECKPOINT = REPO_ROOT / "data" / "corgi_model.pt"  # TODO: replace with packaged checkpoint path if distributed via pip

# TODO: confirm tissue IDs capture the cross-both evaluation split.
tissues = [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
           214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
           323, 324, 422, 442, 443, 473, 474, 515, 517]

regular_tracks = ['dnase','atac','h3k4me1','h3k4me2','h3k4me3','h3k9ac','h3k9me3','h3k27ac','h3k27me3','h3k36me3','h3k79me2','ctcf','rna_10x','wgbs']
stranded_tracks = ['cage', 'rampage', 'rna_total', 'rna_polya']

with open(REPO_ROOT / 'data' / 'experiments_final.txt', 'r') as f:
    experiments_list = f.read().strip().split()

exp_name_to_channel_id = {exp:i for i,exp in enumerate(experiments_list)}

# --- Model loading functions ---
def load_grt_model(device):
    """
    Load the GRT model.
    (Adjust this to import your actual model class.)
    """
    from models import GRT_v3_Pretraining2
    from config import config_molgen as config
    model = GRT_v3_Pretraining2(config).to(device)
    ckpt = torch.load(GRT_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def predict_grt_tile(model, seq_tensor, tf_exp, tissue_id, device, undo_squashed_scale=False):
    """
    Run GRT model on a one-hot encoded sequence tile.
    Returns a tensor of shape (num_channels, L_out), then crop center to TARGET_BINS.
    """
    preds = []
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16):
            for shift in [-2, 0, 2]:
                seq = shift_dna(seq_tensor, shift)
                pred = model(seq, tf_exp)  # expected shape: (batch, channels, L_out)
                preds.append(pred)

                seq_rc = dna_rc(seq)
                pred = predictions_rc(model(seq_rc, tf_exp))
                preds.append(pred)

    preds = [x.float() for x in preds]
    outputs = torch.cat(preds).mean(dim=0)
    if undo_squashed_scale:
        outputs = undo_squashed_scale(outputs, tissue_id, device)
    outputs_cropped = crop_center(outputs, TARGET_BINS)  # shape: (batch, channels, TARGET_BINS)
    return outputs_cropped.squeeze(0)  # remove batch dimension

def extract_gt_tile(tissue_id, chrom, tile_start, track_name):
    """
    Extract ground truth for a given prediction tile.
    For a prediction tile defined by (chrom, tile_start, tile_end) of length 524288 bp,
    extract the central GT_SEQ_LENGTH bp region (from tile_start + CROP to tile_start + CROP + GT_SEQ_LENGTH)
    and bin it into TARGET_BINS (6144) bins (averaging every BINSIZE=64 bp).
    """
    params = transform_params.get(track_name, {})
    if track_name.lower() == 'atac' and 464 <= tissue_id <= 480:
        params = special_atac_params
    sum_stat = params.get('sum_stat', 'mean')

    gt_file = os.path.join(GT_DATA_DIR, str(tissue_id), f"{track_name}.h5") # Not squashed scale!
    if not os.path.exists(gt_file):
        #logging.warning(f"GT file {gt_file} does not exist. Skipping.")
        return None
    # Compute GT extraction coordinates from the prediction tile.
    gt_start = tile_start + CROP
    gt_end = gt_start + GT_SEQ_LENGTH
    with h5py.File(gt_file, 'r') as hf:
        if chrom not in hf:
            logging.warning(f"Chromosome {chrom} not found in {gt_file}.")
            return None
        data = hf[chrom][gt_start:gt_end]
        if len(data) < GT_SEQ_LENGTH:
            logging.warning(f"Data length {len(data)} is shorter than expected {GT_SEQ_LENGTH} in {gt_file} for tile {chrom}:{gt_start}-{gt_end}.")
            return None
        num_bins = GT_SEQ_LENGTH // BINSIZE
        data = data[:num_bins * BINSIZE]
        data = process_coverage(data, params)
        binned = fast_bin(data, sum_stat, n_bins=TARGET_BINS, bin_size=BINSIZE)
        return binned  # numpy array of shape (6144,)

def export_bedgraph_and_bigwig(df, model_type, outdir, chrom_sizes, tracks_to_export='all'):
        """
        For each unique tissue and experiment in the DataFrame,
        export a bedGraph and convert it to a BigWig file.
        Only processes experiments in allowed_exps and (for predictions) only
        if ground truth entries are available.
        The bedGraph is constructed by exploding each tile row into individual
        bins. For each tile row, the binned region is assumed to start at
        tile_start + CROP (the central 196608 bp region) with bins of size BINSIZE.
        """
        # Iterate over unique (tissue, experiment) pairs.
        for (tissue, experiment), group in df.groupby(["tissue", "experiment"]):
            # Only process allowed experiments (compare in lower-case).
            if tracks_to_export is not 'all':
                if experiment.lower() not in tracks_to_export:
                    continue
            # For prediction models (GRT or Borzoi), export only if GT is available.
            #if model_type != "encode" and not ((gt_df["tissue"] == tissue) & (gt_df["experiment"] == experiment)).any():
            #    continue

            bedgraph_lines = []
            for _, row in group.iterrows():
                chrom = row["chr"]
                tile_start = int(row["start"])
                # The binned region corresponding to the predictions is the central region:
                region_start = tile_start + CROP
                # 'values' contains a comma-separated list of 3072 scores.
                bin_scores = row["values"].split(",")
                for i, score in enumerate(bin_scores):
                    bin_start = region_start + i * BINSIZE
                    bin_end = bin_start + BINSIZE
                    bedgraph_lines.append(f"{chrom}\t{bin_start}\t{bin_end}\t{score}")
            # Create file paths.
            bg_path = os.path.join(outdir, f"tissue{tissue}_{experiment}_{model_type}.bedgraph")
            bw_path = os.path.join(outdir, f"tissue{tissue}_{experiment}_{model_type}.bw")
            # Write bedGraph file.
            with open(bg_path, "w") as f:
                f.write("\n".join(bedgraph_lines))
            # Convert bedGraph to BigWig.
            subprocess.run(["bedGraphToBigWig", bg_path, chrom_sizes, bw_path], check=True)
            print(f"Exported {model_type} BigWig for tissue {tissue}, experiment {experiment} to {bw_path}")
            # os.remove(bg_path)

# --- Main processing ---
def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load BED regions.
    bed_regions = parse_bed_file_with_coords(BED_FILE)
    logging.info(f"Parsed {len(bed_regions)} regions from BED file.")

    # Tile regions for predictions.
    pred_tiles = tile_regions(bed_regions, PRED_SEQ_LENGTH, PRED_STRIDE, drop_last=True)
    logging.info(f"Generated {len(pred_tiles)} prediction tiles.")

    # Load genome.
    genome = load_genome(GENOME_FASTA)
    logging.info(f"Genome loaded.")

    # Load models.
    grt_model = load_grt_model(device)
    logging.info(f"Model loaded from {GRT_CHECKPOINT}")

    # Load TF expression
    tf_expression = torch.from_numpy(np.load(TF_EXP_FILE)).float()
    logging.info(f"Loaded TF expression from {TF_EXP_FILE}")

    # We will accumulate tile–level outputs (each a 6144–length array) in lists of dictionaries.
    grt_rows = []
    gt_rows = []
    # Also accumulate per tissue & experiment the predictions (for correlations later)
    grt_accum = defaultdict(list)     # key: (tissue_id, experiment) -> list of numpy arrays
    gt_accum = defaultdict(list)

    # Process tissue–by–tissue.
    for tissue_id in tissues:
        logging.info(f"Processing Tissue {tissue_id}.")
        tf_exp = tf_expression[tissue_id].unsqueeze(0).to(device)

        # --- Process prediction tiles (524kb window) ---
        for (chrom, tile_start, tile_end) in pred_tiles:
            # Get sequence for the full 524kb tile.
            try:
                seq = genome[chrom][tile_start:tile_end].seq
            except Exception as e:
                logging.warning(f"Error fetching sequence for {chrom}:{tile_start}-{tile_end}: {e}")
                continue
            # One-hot encode and convert to tensor.
            seq_encoded = one_hot_encode(seq)
            seq_tensor = torch.from_numpy(seq_encoded).unsqueeze(0).to(device)  # shape: (1, L, 4)

            # GRT prediction
            grt_pred = predict_grt_tile(grt_model, seq_tensor, tf_exp, tissue_id, device)  # shape: (num_channels, TARGET_BINS).

            # Save per–tile prediction for each channel.
            for exp_name in regular_tracks:
                # Check ground truth signal, if it is not available, skip
                gt_array = extract_gt_tile(tissue_id, chrom, tile_start, exp_name)
                if gt_array is None:
                    continue
                channel_id = exp_name_to_channel_id[exp_name]
                grt_array = grt_pred[channel_id].cpu().numpy()
                assert len(grt_array) == len(gt_array)

                grt_row = {
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in grt_array])
                }
                grt_rows.append(grt_row)
                grt_accum[(tissue_id, exp_name)].append(grt_array)

                gt_row = {
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_array])
                }
                gt_rows.append(gt_row)
                gt_accum[(tissue_id, exp_name)].append(gt_array)

            # For stranded tracks, we save the originals and the sum of the tracks
            for exp_name in stranded_tracks:
                plus_name = exp_name+'_plus'
                minus_name = exp_name+'_minus'

                gt_array_plus = extract_gt_tile(tissue_id, chrom, tile_start, plus_name)
                gt_array_minus = extract_gt_tile(tissue_id, chrom, tile_start, minus_name)
                if gt_array_plus is None or gt_array_minus is None:
                    continue
                
                # Plus
                channel_id_plus = exp_name_to_channel_id[plus_name]
                grt_array_plus = grt_pred[channel_id_plus].cpu().numpy()
                assert len(grt_array_plus) == len(gt_array_plus)
                grt_row = {
                    "tissue": tissue_id,
                    "experiment": plus_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in grt_array_plus])
                }
                grt_rows.append(grt_row)
                grt_accum[(tissue_id, plus_name)].append(grt_array_plus)
                gt_row = {
                    "tissue": tissue_id,
                    "experiment": plus_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_array_plus])
                }
                gt_rows.append(gt_row)
                gt_accum[(tissue_id, plus_name)].append(gt_array_plus)

                # Minus
                channel_id_minus = exp_name_to_channel_id[minus_name]
                grt_array_minus = grt_pred[channel_id_minus].cpu().numpy()
                assert len(grt_array_minus) == len(gt_array_minus)
                grt_row = {
                    "tissue": tissue_id,
                    "experiment": minus_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in grt_array_minus])
                }
                grt_rows.append(grt_row)
                grt_accum[(tissue_id, minus_name)].append(grt_array_minus)
                gt_row = {
                    "tissue": tissue_id,
                    "experiment": minus_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_array_minus])
                }
                gt_rows.append(gt_row)
                gt_accum[(tissue_id, minus_name)].append(gt_array_minus)

                # Total
                grt_array = grt_array_minus + grt_array_plus
                gt_array = gt_array_minus + gt_array_plus
                grt_row = {
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in grt_array])
                }
                grt_rows.append(grt_row)
                grt_accum[(tissue_id, exp_name)].append(grt_array)
                gt_row = {
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_array])
                }
                gt_rows.append(gt_row)
                gt_accum[(tissue_id, exp_name)].append(gt_array)

    grt_df = pd.DataFrame(grt_rows)
    gt_df = pd.DataFrame(gt_rows)

    # --- Compute correlations per tissue and experiment ---
    def safe_corr(a, b):
        # Compute correlations if arrays are nonempty and have nonzero std.
        if a.size == 0 or b.size == 0 or np.std(a)==0 or np.std(b)==0:
            return float('nan'), float('nan')
        return pearsonr(a, b)[0], spearmanr(a, b)[0]

    corr_results = []
    for key in grt_accum.keys():
        tissue_id, exp_name = key
        # Concatenate along axis=0 (i.e. tile concatenation)
        grt_concat = np.concatenate(grt_accum[key])
        if len(gt_accum[key]) > 0:
            gt_concat = np.concatenate(gt_accum[key])
        else:
            continue
        
        grt_pearson, grt_spearman = safe_corr(grt_concat, gt_concat)
        corr_results.append({
            "tissue": tissue_id,
            "experiment": exp_name,
            "model": "GRT",
            "pearson": round(grt_pearson, 3),
            "spearman": round(grt_spearman, 3)
        })

    corr_df = pd.DataFrame(corr_results)
    os.makedirs(OUTDIR, exist_ok=True)
    corr_df.to_csv(f"{OUTDIR}/correlation_results.csv", index=False)
    logging.info("Correlation results written to correlation_results.csv")
    
    # Export BigWigs for predictions and ground truth.
    if EXPORT_TO_BIGWIG:
        export_bedgraph_and_bigwig(grt_df, "grt", OUTDIR, CHROM_SIZES_FILE)
        export_bedgraph_and_bigwig(gt_df, "encode", OUTDIR, CHROM_SIZES_FILE)
    
if __name__ == "__main__":
    main()
