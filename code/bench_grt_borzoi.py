#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath('/project/deeprna/scripts/GRT_v3'))
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

from borzoi_pytorch import Borzoi
from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks

from benchmark_utils import transform_params, special_atac_params, transform_softclip, transform_scale, transform_scale_special
from benchmark_utils import undo_squashed_scale, parse_bed_file_with_coords, one_hot_encode, load_genome, tile_region, tile_regions, crop_center
from benchmark_utils import dna_rc, predictions_rc, shift_dna, process_coverage, fast_bin

# --- Global parameters ---
# Tiling for model predictions:
PRED_SEQ_LENGTH = 524288      # tile length for predictions (bp)
PRED_STRIDE = 196608          # stride for predictions (bp)
CROP = 163840                 # offset to crop the central 196608 bp region from a 524288 bp tile

# Tiling for ground truth:
GT_SEQ_LENGTH = 196608        # tile length for ground truth extraction (bp)
BINSIZE = 64                # bin size in bp
TARGET_BINS = 3072          # number of bins required (3072*64 = 196608 bp)
assert GT_SEQ_LENGTH // BINSIZE == TARGET_BINS

# File paths
BED_FILE = "/project/deeprna/data/borzoi_fold3_merged.bed"
TRACKS_CSV = "/project/deeprna/data/training_tissues_borzoi_matches_extended.csv"
GENOME_FASTA = "/project/deeprna_data/borzoi/hg38.ml.fa"
GT_DATA_DIR = "/project/deeprna_data/pretraining_data_merged" 
TF_EXP_FILE = '/project/deeprna_data/pretraining_data_final2/tf_expression.npy'
CHROM_SIZES_FILE = "/project/deeprna/data/hg38/hg38.chrom.sizes"
OUTDIR = "/project/deeprna/benchmark/grt_vs_borzoi"

# Model checkpoint
GRT_CHECKPOINT = '/project/deeprna/models/grt_v3_pretraining/adaptive_mn/grt_epoch_4_2025-03-25_20:34.pt'
BORZOI_NUM = 4     # For Borzoi, we load the ensemble of 4 replicates

# --- Model loading functions ---
def load_borzoi_ensemble(device):
    """
    Load a Borzoi ensemble of BORZOI_NUM models.
    (Adjust this to your actual Borzoi model loading method.)
    """
    ensemble = []
    for i in range(BORZOI_NUM):
        model = Borzoi.from_pretrained(f'johahi/borzoi-replicate-{i}')
        model = model.to(device)
        model.eval()
        ensemble.append(model)
    return ensemble

def predict_borzoi_tile(ensemble, seq_tensor, device, targets_df):
    """
    Run Borzoi ensemble on a one-hot encoded sequence tile.
    Here we assume that the ensemble returns output with 32 bp resolution covering a 196608 bp region,
    i.e. 6144 bins per track. We then average pool (kernel=2, stride=2) to get TARGET_BINS = 3072 bins.
    Outputs all tracks.
    """
    # Assume seq_tensor is of shape (1, L, 4). Borzoi might expect shape (L, 4) so adjust accordingly.
    seq_sample = seq_tensor.squeeze(0).float().to(device)  # shape: (L, 4)
    # Get predictions from each model and average them
    # The function below returns a numpy array; adjust if necessary.
    pred_np = predict_tracks(ensemble, seq_sample, list(range(7611)))  # shape: (1, 4, 6144, num_tracks=7611)
    mean_pred = np.mean(pred_np, axis=1)  # shape: (1, 6144, num_tracks)
    mean_pred = np.squeeze(mean_pred, axis=0)  # shape: (6144, num_tracks)
    pooled = mean_pred.reshape(3072, 2, mean_pred.shape[1]).mean(axis=1)  # shape: (3072, num_tracks)
    #unsquashed = undo_squashed_borzoi(pooled, targets_df)  # expects a NumPy array
    # Transpose the array so that channels (tracks) come first: shape becomes (num_tracks, 3072)
    final = pooled.transpose(1, 0)
    return final # (num_tracks=7611, 3072)

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
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs = model(seq_tensor, tf_exp)  # expected shape: (batch, channels, L_out)
    outputs = outputs.float()
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

# --- Main processing ---
def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the tracks CSV.
    tracks_df = pd.read_csv(TRACKS_CSV)
    logging.info(f"Loaded {len(tracks_df)} rows from grt-vs-borzoi CSV.")

    # Load BED regions.
    bed_regions = parse_bed_file_with_coords(BED_FILE)
    logging.info(f"Parsed {len(bed_regions)} regions from BED file.")

    # Tile regions for predictions.
    pred_tiles = tile_regions(bed_regions, PRED_SEQ_LENGTH, PRED_STRIDE, drop_last=True)
    logging.info(f"Generated {len(pred_tiles)} prediction tiles.")

    # Load genome.
    genome = load_genome(GENOME_FASTA)

    # Load models.
    grt_model = load_grt_model(device)
    borzoi_ensemble = load_borzoi_ensemble(device)
    logging.info(f"GRT model loaded from {GRT_CHECKPOINT}, Borzoi 4 model ensemble loaded.")

    # Load TF expression
    tf_expression = torch.from_numpy(np.load(TF_EXP_FILE)).float()
    logging.info(f"Loaded TF expression from {TF_EXP_FILE}")

    with open('/project/deeprna/data/experiments_final.txt', 'r') as f:
        experiments_list = f.read().strip().split()
    exp_name_to_channel_id = {exp:i for i,exp in enumerate(experiments_list)}

    # We will accumulate tile–level outputs (each a 3072–length array) in lists of dictionaries.
    grt_rows = []
    borzoi_rows = []
    gt_rows = []
    # Also accumulate per tissue & experiment the predictions (for correlations later)
    grt_accum = defaultdict(list)     # key: (tissue_id, experiment) -> list of numpy arrays
    borzoi_accum = defaultdict(list)
    gt_accum = defaultdict(list)

    # Borzoi
    logging.info(f"Beginning Borzoi calculations.")
    targets_df = pd.read_csv('/project/deeprna/data/targets_human.txt', sep='\t', index_col=0)
    for (chrom, tile_start, tile_end) in pred_tiles:
        # Get sequence for the full 524kb tile.
        try:
            seq = genome[chrom][tile_start:tile_end].seq
        except Exception as e:
            logging.warning(f"Error fetching sequence for {chrom}:{tile_start}-{tile_end}: {e}")
            continue
        # One-hot encode and convert to tensor.
        seq_encoded = one_hot_encode(seq)
        seq_tensor = torch.from_numpy(seq_encoded).unsqueeze(0).permute(0,2,1).to(device)  # shape: (1, L, 4)
        borzoi_pred = predict_borzoi_tile(borzoi_ensemble, seq_tensor, device, targets_df) # (7611, 3072)

        for idx, row in tracks_df.iterrows():
            exp_name = row["channel_name"]
            track_id = row['borzoi_track_id']
            tissue_id = row['tissue_id']
            borzoi_row = {
                "tissue": row['tissue_id'],
                "experiment": exp_name,
                "chr": chrom,
                "start": tile_start,
                "end": tile_end,
                "values": ",".join([f"{v:.3f}" for v in borzoi_pred[track_id]])
            }
            borzoi_rows.append(borzoi_row)
            borzoi_accum[(tissue_id, exp_name)].append(borzoi_pred[track_id])
    logging.info(f"Borzoi done.")

    # Process tissue–by–tissue based on the CSV.
    for tissue_id, group in tracks_df.groupby("tissue_id"):
        tissue_id = int(tissue_id)
        # For this tissue, get channels in the order provided.
        channels = group.sort_values("channel_id").to_dict("records")
        channels_count = len(channels)
        logging.info(f"Processing Tissue {tissue_id} with {channels_count} channels.")

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
            # For GRT, predictions:
            grt_pred = predict_grt_tile(grt_model, seq_tensor, tf_exp, tissue_id, device)  
            # grt_pred shape: (num_channels, TARGET_BINS).

            # Save per–tile prediction for each channel.
            for ch in channels:
                exp_name = ch["channel_name"]
                channel_id = exp_name_to_channel_id[exp_name]
                # GRT row
                grt_row = {
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in grt_pred[channel_id].cpu().numpy()])
                }
                grt_rows.append(grt_row)
                grt_accum[(tissue_id, exp_name)].append(grt_pred[channel_id].cpu().numpy())

            # --- Ground Truth extraction ---
            for ch in channels:
                exp_name = ch["channel_name"]
                gt_signal = extract_gt_tile(tissue_id, chrom, tile_start, exp_name)
                if gt_signal is None:
                    continue
                gt_row = {
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_signal])
                }
                gt_rows.append(gt_row)
                gt_accum[(tissue_id, exp_name)].append(gt_signal)
    
    grt_df = pd.DataFrame(grt_rows)
    borzoi_df = pd.DataFrame(borzoi_rows)
    gt_df = pd.DataFrame(gt_rows)
    #grt_df.to_csv("/project/deeprna/predictions/grt_vs_borzoi/grt_tiles.csv", index=False)
    #borzoi_df.to_csv("/project/deeprna/predictions/grt_vs_borzoi/borzoi_tiles.csv", index=False)
    #gt_df.to_csv("/project/deeprna/predictions/grt_vs_borzoi/gt_tiles.csv", index=False)
    #logging.info("Tile-level CSV files written: grt_tiles.csv, borzoi_tiles.csv, gt_tiles.csv")

    # --- Compute correlations per tissue and experiment ---
    def safe_corr(a, b):
        if a.size == 0 or b.size == 0 or np.std(a)==0 or np.std(b)==0:
            return float('nan'), float('nan')
        return pearsonr(a, b)[0], spearmanr(a, b)[0]

    corr_results = []
    for key in grt_accum.keys():
        tissue_id, exp_name = key
        # Concatenate along axis=0 (i.e. tile concatenation)
        grt_concat = np.concatenate(grt_accum[key])
        borzoi_concat = np.concatenate(borzoi_accum[key])
        gt_concat = np.concatenate(gt_accum[key])
        
        grt_pearson, grt_spearman = safe_corr(grt_concat, gt_concat)
        borzoi_pearson, borzoi_spearman = safe_corr(borzoi_concat, gt_concat)
        corr_results.append({
            "tissue": tissue_id,
            "experiment": exp_name,
            "model": "GRT",
            "pearson": round(grt_pearson, 3),
            "spearman": round(grt_spearman, 3)
        })
        corr_results.append({
            "tissue": tissue_id,
            "experiment": exp_name,
            "model": "Borzoi",
            "pearson": round(borzoi_pearson, 3),
            "spearman": round(borzoi_spearman, 3)
        })
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(f"{OUTDIR}/correlation_results.csv", index=False)
    logging.info(f"Correlation results written to {OUTDIR}/correlation_results.csv")
    
    # --- Writing to bigwig ---
    def export_bedgraph_to_bigwig(df, model_type, outdir, chrom_sizes):
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
            subprocess.run(["bedGraphToBigWig", bg_path, CHROM_SIZES_FILE, bw_path], check=True)
            print(f"Exported {model_type} BigWig for tissue {tissue}, experiment {experiment} to {bw_path}")
            #os.remove(bg_path)

    # Export BigWigs for predictions and ground truth.
    export_bedgraph_to_bigwig(grt_df, "grt", OUTDIR, CHROM_SIZES_FILE)
    export_bedgraph_to_bigwig(borzoi_df, "borzoi", OUTDIR, CHROM_SIZES_FILE)
    export_bedgraph_to_bigwig(gt_df, "encode", OUTDIR, CHROM_SIZES_FILE)
    
if __name__ == "__main__":
    main()
