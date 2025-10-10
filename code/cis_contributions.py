#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import pickle as pk
import os
import logging

from captum.attr import DeepLift
from tangermeme.deep_lift_shap import dinucleotide_shuffle

import sys
sys.path.insert(0, os.path.abspath('/project/deeprna/scripts/GRT_v3'))
from config import config_molgen as config
from utils import load_grt_model
from benchmark_utils import load_genome, one_hot_encode, parse_bed_file_with_coords, tile_regions


################################################################################
# Part 1: Model Definitions and Runtime Patches
################################################################################

# --- The "SafeActivation" Wrapper ---
# This class creates a "float32 island" to solve the dtype mismatch with hooks.
class SafeActivation(nn.Module):
    def __init__(self, activation_module):
        super().__init__()
        self.activation = activation_module

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        activation_val = self.activation(x_float)
        return activation_val.to(input_dtype)

# --- The "Monkey-Patching" Function ---
# This function dynamically applies the SafeActivation wrapper at runtime.
def patch_activations(model):
    """
    Recursively finds all nn.GELU and nn.Softplus modules in a model
    and replaces them with a SafeActivation wrapper. Modifies the model in-place.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.GELU, nn.Softplus, nn.ReLU)):
            print(f"Patching: {name} of type {type(module).__name__}")
            setattr(model, name, SafeActivation(module))
        else:
            patch_activations(module)

# --- Wrapper for Attribution ---
# This class sums the model's track outputs for a single scalar value.
class GRTModelWrapper(torch.nn.Module):
    def __init__(self, model, tf_exp, exon_mask=None, channels_to_sum=[16, 17]):
        super(GRTModelWrapper, self).__init__()
        self.model = model
        self.tf_exp = tf_exp
        self.channels_to_sum = channels_to_sum
        self.exon_mask = exon_mask

    def forward(self, x):
        outputs = self.model(x, self.tf_exp)  # Shape: (1, 18, 6144)
        if self.exon_mask is not None:
            # Sum exonic bins
            rna_tracks = outputs[:, self.channels_to_sum, :].float() # (1, 2, 6144)
            # Fix: Reshape the mask to match rna_tracks for proper broadcasting
            mask = self.exon_mask.view(1, 1, -1)  # Shape: (1, 1, 6144)
            masked_rna = rna_tracks * mask  # Now properly broadcasts to (1, 2, 6144)
            channel_sum = torch.sum(masked_rna, dim=1)  # (1, 6144)
            bin_sum = torch.sum(channel_sum, dim=1, keepdim=True)  # (1, 1)
        else:
            # Sum all bins
            selected_channels = outputs[:, self.channels_to_sum, :].float()
            channel_sum = torch.sum(selected_channels, dim=1)
            bin_sum = torch.sum(channel_sum, dim=1, keepdim=True)
                
        return bin_sum

def calculate_contributions(dl, sequence, n_shuffles=20):
    """
    Calculate contributions using DeepLIFT with shuffled sequences.
    """
    all_contributions = []
    for _ in range(n_shuffles):
        # Generate reference sequences (baselines)
        try:
            shuffled_seqs = dinucleotide_shuffle(sequence.cpu(), n=1)
        except:
            return torch.zeros_like(sequence).mean(dim=0)
        
        reference = shuffled_seqs.squeeze(0).to(sequence.device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            raw_attributions = dl.attribute(sequence, baselines=reference, target=0)

        # Calculate final contribution scores (dL * dX * X)
        contribution_scores = raw_attributions * sequence
        cont = contribution_scores.detach().cpu()
        all_contributions.append(cont)
        torch.cuda.empty_cache()
    
    return torch.cat(all_contributions, dim=0).mean(dim=0)  # Average across all shuffles
   
if __name__ == '__main__':
    # Configuration
    MODEL_PATH = "/project/deeprna/models/grt_v3_pretraining/adaptive_mn/grt_epoch_4_2025-03-25_20:34.pt"
    GENOME_PATH = "/project/deeprna_data/borzoi/hg38.ml.fa"
    BED_PATH = "/project/deeprna/benchmark/fold3_notf_merged.bed"
    TF_EXPR_PATH = "/project/deeprna_data/pretraining_data_final2/tf_expression.npy"
    MOTIF_PATH = "/project/deeprna/data/motifs/transfac2021_vertebrate_recommended.meme"
    OUTPUT_DIR = "/project/deeprna/benchmark/contributions"
    N_SHUFFLES = 20
    PRED_SEQ_LENGTH = 524288
    PRED_STRIDE = 393216

    #tissues = [59, 124, 192, 213, 277, 323]
    tissues = [277, 323]

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    genome = load_genome(GENOME_PATH)
    tf_expression = torch.from_numpy(np.load(TF_EXPR_PATH)).float()

    # Model loading, patching
    print("Loading model...")
    model = load_grt_model(MODEL_PATH, config, device, shapley=True)
    patch_activations(model)
    model.eval()
    
    # Load BED regions.
    bed_regions = parse_bed_file_with_coords(BED_PATH)
    logging.info(f"Parsed {len(bed_regions)} regions from BED file.")

    # Tile regions for predictions.
    pred_tiles = tile_regions(bed_regions, PRED_SEQ_LENGTH, PRED_STRIDE, drop_last=True)
    logging.info(f"Generated {len(pred_tiles)} prediction tiles.")

    for tissue in tissues:
        tissue_scores = {}
        print(f"Processing tissue ID: {tissue}")
        tf_exp  = tf_expression[tissue, :].unsqueeze(0).to(device)
        model_wrapped = GRTModelWrapper(model, tf_exp).to(device)

        for (chrom, tile_start, tile_end) in pred_tiles:
            # Get sequence for the full 524kb tile.
            try:
                seq = genome[chrom][tile_start:tile_end].seq
            except Exception as e:
                logging.warning(f"Error fetching sequence for {chrom}:{tile_start}-{tile_end}: {e}")
                continue
            # One-hot encode and convert to tensor.
            seq_encoded = one_hot_encode(seq)
            seq_tensor = torch.from_numpy(seq_encoded).unsqueeze(0).permute(0, 2, 1).to(device)
            dl = DeepLift(model_wrapped)
            scores = calculate_contributions(dl, seq_tensor, N_SHUFFLES)
            tissue_scores[(chrom, tile_start, tile_end)] = scores.cpu().numpy()

        # Save scores for the tissue
        tissue_file = os.path.join(OUTPUT_DIR, f"tissue_{tissue}_scores.pkl")
        with open(tissue_file, 'wb') as f:
            pk.dump(tissue_scores, f)

        print(f"Scores saved for tissue {tissue} in {tissue_file}")