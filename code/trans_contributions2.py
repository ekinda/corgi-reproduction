#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import pickle as pk
import os
import logging

from captum.attr import DeepLift

# Runtime patches and utilities
from tangermeme.deep_lift_shap import dinucleotide_shuffle  # kept for cis, not used here
import sys
sys.path.insert(0, os.path.abspath('/project/deeprna/scripts/GRT_v3'))
from config import config_molgen as config
from utils import load_grt_model
from benchmark_utils import load_genome, one_hot_encode, parse_bed_file_with_coords, tile_regions

# --- SafeActivation to avoid dtype mismatches ---
class SafeActivation(nn.Module):
    def __init__(self, activation_module):
        super().__init__()
        self.activation = activation_module
    def forward(self, x):
        x_float = x.to(torch.float32)
        out = self.activation(x_float)
        return out.to(x.dtype)

def patch_activations(model):
    for name, module in model.named_children():
        if isinstance(module, (nn.GELU, nn.Softplus, nn.ReLU)):
            setattr(model, name, SafeActivation(module))
        else:
            patch_activations(module)

# --- Wrapper that fixes sequence and takes TF inputs ---
class GRTModelWrapper(torch.nn.Module):
    def __init__(self, model, sequence, exon_mask=None, channels_to_sum=[16, 17]):
        super().__init__()
        self.model = model
        self.sequence = sequence
        self.channels_to_sum = channels_to_sum
        self.exon_mask = exon_mask

    def forward(self, tf_input):
        # model takes sequence and tf expression
        outputs = self.model(self.sequence, tf_input)
        if self.exon_mask is not None:
            rna = outputs[:, self.channels_to_sum, :].float()
            mask = self.exon_mask.view(1, 1, -1)
            masked = rna * mask
            summed = masked.sum(dim=(1, 2), keepdim=True)
        else:
            sel = outputs[:, self.channels_to_sum, :].float()
            summed = sel.sum(dim=(1, 2), keepdim=True)
        return summed

def calculate_trans_contributions(dl, tf_expr, tf_mean):
    """
    Compute DeepLift contributions for TF-expression input using mean expression as baseline.
    Returns a tensor of shape (1, n_tfs).
    """
    with torch.autocast('cuda', dtype=torch.bfloat16):
        contrib = dl.attribute(tf_expr, baselines=tf_mean, target=0)
    return contrib.detach().cpu()

if __name__ == '__main__':
    # Paths and params
    MODEL_PATH = "/project/deeprna/models/grt_v3_pretraining/adaptive_mn/grt_epoch_4_2025-03-25_20:34.pt"
    GENOME_PATH = "/project/deeprna_data/borzoi/hg38.ml.fa"
    BED_PATH = "/project/deeprna/benchmark/fold3_notf_merged.bed"
    TF_EXPR_PATH = "/project/deeprna_data/pretraining_data_final2/tf_expression.npy"
    OUTPUT_DIR = "/project/deeprna_data/benchmark/contributions"
    N_SHUFFLES = 20
    PRED_SEQ_LENGTH = 524288
    PRED_STRIDE = 393216
    tissues = [213, 277, 323]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load reference data
    print("Loading data...")
    genome = load_genome(GENOME_PATH)
    tf_expression = torch.from_numpy(np.load(TF_EXPR_PATH)).float()
    tf_mean = tf_expression.mean(dim=0, keepdim=True).to(device)

    # Load and patch model
    print("Loading model...")
    model = load_grt_model(MODEL_PATH, config, device, shapley=True)
    patch_activations(model)
    model.eval()

    # Regions
    bed_regions = parse_bed_file_with_coords(BED_PATH)
    tiles = tile_regions(bed_regions, PRED_SEQ_LENGTH, PRED_STRIDE, drop_last=True)

    for tissue in tissues:
        print(f"Processing tissue: {tissue}")
        tf_exp = tf_expression[tissue].unsqueeze(0).to(device)
        tissue_scores = {}

        for chrom, start, end in tiles:
            try:
                seq = genome[chrom][start:end].seq
            except Exception as e:
                logging.warning(f"Skipping {chrom}:{start}-{end}: {e}")
                continue
            # one-hot encode
            enc = one_hot_encode(seq)
            seq_tensor = torch.from_numpy(enc).unsqueeze(0).permute(0, 2, 1).to(device)

            # wrapper, deepLift
            wrapped = GRTModelWrapper(model, seq_tensor).to(device)
            dl = DeepLift(wrapped)

            # compute trans contributions
            scores = calculate_trans_contributions(dl, tf_exp, tf_mean)
            tissue_scores[(chrom, start, end)] = scores.squeeze(0).numpy()

        out_file = os.path.join(OUTPUT_DIR, f"tissue_{tissue}_trans_scores.pkl")
        with open(out_file, 'wb') as f:
            pk.dump(tissue_scores, f)
        print(f"Saved trans contributions for tissue {tissue} at {out_file}")