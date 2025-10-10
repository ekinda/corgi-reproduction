import os
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pyfaidx import Fasta

import torch
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
from trainers import GRTBaseTrainer
from data_classes import GRTDatasetGnomad, GRTDistributedSampler
from utils import poisson_nll_masked, poisson_multinomial_masked, poisson_nll_masked, poisson_multinomial_masked_v2
from models import GRT_v3_Pretraining2

# These are constants from the original configuration.
transform_params = {
    'dnase':          {'clip': 64, 'soft_clip': 32,  'scale': 2.0,  'sum_stat': 'mean'},
    'atac':           {'clip': 64, 'soft_clip': 32,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k4me1':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k4me2':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k4me3':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k9ac':         {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k9me3':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k27ac':        {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k27me3':       {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k36me3':       {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'h3k79me2':       {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'ctcf':           {'clip': 64, 'soft_clip': 48,  'scale': 1.0,  'sum_stat': 'mean'},
    'cage_plus':      {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'cage_minus':     {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'rampage_plus':   {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'rampage_minus':  {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum'},
    'rna_total_plus':  {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_total_minus': {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_polya_plus':  {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_polya_minus': {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'rna_10x':         {'clip': 512, 'soft_clip': 384, 'scale': 1.0,  'sum_stat': 'sum_sqrt'},
    'wgbs':          {'clip': 128, 'soft_clip': 64,  'scale': 1.0,  'sum_stat': 'mean'}
}
special_atac_params = {'clip': 64, 'soft_clip': 32, 'scale': 3.0, 'sum_stat': 'mean'}

with open(REPO_ROOT / 'data' / 'experiments_final.txt', 'r') as f:
    experiments_list = f.read().strip().split()

# Precompute tensors for undoing squashed scale.
transform_softclip = torch.tensor([transform_params[x]['soft_clip'] for x in experiments_list]).unsqueeze(0).unsqueeze(-1)
transform_scale = torch.tensor([transform_params[x]['scale'] for x in experiments_list]).unsqueeze(0).unsqueeze(-1)
transform_scale_special = transform_scale.clone()
transform_scale_special[:, 1, :] = special_atac_params['scale']

def process_coverage(values: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply scaling, soft clip, and hard clip to coverage data.
    """
    coverage = values.astype(np.float16, copy=False)
    scale = params.get('scale', 1.0)
    soft_clip_val = params.get('soft_clip', None)
    clip_val = params.get('clip', None)

    coverage *= scale

    if soft_clip_val is not None:
        tc = float(soft_clip_val)
        coverage = np.minimum(coverage, tc + np.sqrt(np.maximum(0, coverage - tc)))
    if clip_val is not None:
        coverage = np.clip(coverage, 0, clip_val)
    return coverage

def fast_bin(raw_coverage: np.ndarray, sum_stat: str, n_bins: int, bin_size: int) -> np.ndarray:
    """
    Bins the raw coverage vector (length 524288) into 8192 bins (each of length 64)
    using the specified summary statistic.
    """
    reshaped = raw_coverage.reshape(n_bins, bin_size)
    if sum_stat == 'mean':
        return reshaped.mean(axis=1)
    elif sum_stat == 'sum':
        return reshaped.sum(axis=1)
    elif sum_stat == 'sum_sqrt':
        return np.sqrt(reshaped.sum(axis=1))
    else:
        raise ValueError(f"Unsupported sum_stat: {sum_stat}")
        
def undo_squashed_scale(coverage_pred, tissue_id, device):
    """
    Undo the squashed scale transformation on model predictions.
    Expects coverage_pred of shape (batch, channels, bins).
    """
    coverage = coverage_pred.clone().to(device)
    clip = transform_softclip.to(device)
    if 464 <= tissue_id <= 480:
        scale = transform_scale_special.to(device)
    else:
        scale = transform_scale.to(device)
    mask = coverage > clip
    adjusted = (coverage - clip) ** 2 + clip
    coverage = torch.where(mask, adjusted, coverage)
    coverage = coverage / scale
    return coverage

def parse_bed_file_with_coords(bed_path):
    """
    Parse a BED file and return a list of tuples (chr, start, end).
    Assumes no header and at least three columns.
    """
    df = pd.read_csv(bed_path, sep='\t', header=None, names=["chr", "start", "end", "fold"])
    coords = []
    for _, row in df.iterrows():
        coords.append((row["chr"], int(row["start"]), int(row["end"])))
    return coords

def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence.
    """
    mapping = {'A':0, 'C':1, 'G':2, 'T':3,
               'a':0, 'c':1, 'g':2, 't':3}
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            onehot[i, mapping[base]] = 1.0
    return onehot

def load_genome(fasta_path):
    """
    Load the genome using pyfaidx.
    """
    return Fasta(fasta_path)

def tile_region(chrom, start, end, sequence_length, stride, drop_last=True):
    """
    Given a region defined by (chrom, start, end),
    create tiles of length sequence_length with the given stride.
    Returns a list of tuples: (chrom, tile_start, tile_end)
    """
    tiles = []
    region_length = end - start
    if region_length < sequence_length:
        return tiles
    current_start = start
    while current_start + sequence_length <= end:
        tiles.append((chrom, current_start, current_start + sequence_length))
        current_start += stride
    # Optionally, if drop_last is False, add a shifted tile.
    return tiles

def tile_regions(regions, sequence_length, stride, drop_last=True):
    """
    For a list of regions (each as (chr, start, end)), return all tiles.
    """
    tiled = []
    for region in regions:
        chrom, start, end = region
        tiled.extend(tile_region(chrom, start, end, sequence_length, stride, drop_last))
    return tiled

def crop_center(tensor, target_bins):
    """
    Crop the tensor (last dimension) to target_bins by taking the central portion.
    Assumes tensor shape: (batch, channels, L)
    """
    L = tensor.shape[-1]
    start = (L - target_bins) // 2
    return tensor[..., start:start+target_bins]

def dna_rc(sequence):
    # sequence: tensor of shape (1, len, 4)
    sequence = torch.flip(sequence, dims=[1])  # Reverse along length
    sequence = sequence[..., [3, 2, 1, 0]]       # Complement: A<->T, C<->G
    return sequence

def predictions_rc(pred):
    # pred: tensor of shape (1, 22, bins-usually 6144)
    rc_flipped_strands = list(range(22))
    rc_flipped_strands[12], rc_flipped_strands[13] = rc_flipped_strands[13], rc_flipped_strands[12]
    rc_flipped_strands[14], rc_flipped_strands[15] = rc_flipped_strands[15], rc_flipped_strands[14]
    rc_flipped_strands[16], rc_flipped_strands[17] = rc_flipped_strands[17], rc_flipped_strands[16]
    rc_flipped_strands[18], rc_flipped_strands[19] = rc_flipped_strands[19], rc_flipped_strands[18]

    pred = torch.flip(pred, dims=[2])
    pred = pred[:, rc_flipped_strands, :]
    return pred
    
def shift_dna(sequence, shift):
    # Shifts one hot encoded DNA sequence to the right or left, + means padding on the left.
    # Input sequence is 3-dimensional tensor of shape (1, len, 4)
    device = 'cpu' if sequence.get_device() == -1 else 'cuda'
    if shift == 0:
        return sequence
    elif shift > 0:
        padding = torch.zeros(1, shift, 4).to(device)
        return torch.cat([padding, sequence[:, :-shift, :]], dim=1)
    elif shift < 0:
        padding = torch.zeros(1, -shift, 4).to(device)
        return torch.cat([sequence[:, -shift:, :], padding], dim=1)

class GRTEvaluate(GRTBaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.rank = 0
        self.world_size = 1
        self.seed = config["seed"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._set_seed(self.config["seed"])
        self._prepare_data()
        self.new_build_model()

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)

    def new_build_model(self):
        model = GRT_v3_Pretraining2(self.config)
        model = model.to(self.device)
        self.model = model

    def evaluate(self, eval_seq_ids, eval_tissues):
        cfg = self.config
        with open(cfg['experiments_path'], 'r') as f:
            track_names = f.read().strip().split()
        num_channels = cfg["output_channels"]
        loss_epsilon = cfg['loss_epsilon']
        poisson_loss_weight = cfg['poisson_loss_weighting']

        eval_results = {}
        for tissue in eval_tissues:
            eval_dataset = GRTDatasetGnomad(
                dna_sequences=self.dna_path,
                sequence_ids=eval_seq_ids,
                tissue_dir=cfg["data_dir"],
                tissue_ids=[tissue],
                experiment_mask=self.experiment_mask,
                tf_expression=self.tf_exp,
                output_channels=cfg["output_channels"],
                augment_dna=False,
                augment_tf_std=0,
                gnomad_pickle=None,
                tf_exp_clip=None
            )
            eval_sampler = GRTDistributedSampler(
                sequence_ids=eval_seq_ids,
                tissue_ids=[tissue],
                num_processes=1,
                rank=self.rank,
                seed=self.seed
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=cfg["batch_size"],
                sampler=eval_sampler,
                num_workers=2,
                pin_memory=False
            )

            i = 0
            overall_loss_sum = 0.0
            #total_channel_loss = torch.zeros(num_channels, device=self.device)
            all_preds = []
            all_labels = []
            first_batch_exp_mask = None

            self.model.eval()
            with torch.no_grad():
                for dna_seq, tf_exp, label, exp_mask in eval_loader:
                    dna_seq = dna_seq.to(self.device, non_blocking=True)
                    tf_exp = tf_exp.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)
                    exp_mask = exp_mask.to(self.device, non_blocking=True)
                    
                    cropped_label = self.crop_tensor(label, cfg["output_central_bins"]).float()

                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model(dna_seq, tf_exp)

                    outputs = outputs.float()

                    # Compute channel losses.
                    if cfg['loss_style'] == 'poisson_mn':
                        loss = poisson_multinomial_masked(outputs, cropped_label, exp_mask, poisson_loss_weight, loss_epsilon, channel_loss_weights)
                    elif cfg['loss_style'] == 'poisson_nll':
                        channel_losses = poisson_nll_masked(outputs, cropped_label, exp_mask, loss_epsilon)
                        print(channel_losses.shape)
                        channel_losses = (channel_losses * channel_loss_weights.unsqueeze(0)) * exp_mask.squeeze(-1)
                        print(channel_losses.shape)
                        loss = channel_losses.sum() / exp_mask.sum()
                    elif cfg['loss_style'] == 'adaptive_loss':
                        channel_losses = poisson_nll_masked(outputs, cropped_label, exp_mask, loss_epsilon)
                        weights = (1 / (2 * self.model.loss_channel_weights ** 2))  # shape: (22,)
                        channel_losses = (channel_losses * weights.unsqueeze(0) + torch.log(self.model.loss_channel_weights).unsqueeze(0)) * exp_mask.squeeze(-1)
                        loss = channel_losses.sum() / exp_mask.sum()
                    elif cfg['loss_style'] == 'adaptive_mn':
                        channel_losses = poisson_multinomial_masked_v2(outputs, cropped_label, exp_mask, poisson_loss_weight, loss_epsilon)
                        weights = (1 / (2 * self.model.loss_channel_weights ** 2))  # shape: (22,)
                        channel_losses = (channel_losses * weights.unsqueeze(0) + torch.log(self.model.loss_channel_weights).unsqueeze(0)) * exp_mask.squeeze(-1)
                        loss = channel_losses.sum() / exp_mask.sum()
                    
                    # Update overall loss accumulators.
                    overall_loss_sum += loss
                    
                    # Update per-channel loss accumulators.
                    #total_channel_loss += channel_losses.sum(dim=0)

                    # Collect predictions and labels.
                    preds_batch = outputs[:, :, :cfg["output_central_bins"]].float()
                    labels_batch = cropped_label[:, :, :cfg["output_central_bins"]].float()
                    all_preds.append(preds_batch.detach().cpu())
                    all_labels.append(labels_batch.detach().cpu())

                    if first_batch_exp_mask is None:
                        first_batch_exp_mask = exp_mask[0].detach().cpu()  # shape: (22, 1)

                    i += 1

            overall_loss = overall_loss_sum / i
            # per_channel_loss = total_channel_loss.detach().cpu().numpy()

            # Concatenate predictions and labels across batches.
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Reshape to (num_seq * bins, num_channels) for vectorized correlation calculation.
            all_preds_flat = all_preds.permute(0, 2, 1).reshape(-1, num_channels)
            all_labels_flat = all_labels.permute(0, 2, 1).reshape(-1, num_channels)

            preds_np = all_preds_flat.numpy()
            labels_np = all_labels_flat.numpy()

            # ---------------------------
            # Vectorized Pearson correlation.
            preds_mean = preds_np.mean(axis=0)
            labels_mean = labels_np.mean(axis=0)
            numerator = ((preds_np - preds_mean) * (labels_np - labels_mean)).sum(axis=0)
            denom = np.sqrt(((preds_np - preds_mean)**2).sum(axis=0) *
                            ((labels_np - labels_mean)**2).sum(axis=0))
            pearson = np.where(denom == 0, np.nan, numerator / denom)

            # ---------------------------
            # Vectorized Spearman correlation.
            preds_ranks = np.argsort(np.argsort(preds_np, axis=0), axis=0) + 1
            labels_ranks = np.argsort(np.argsort(labels_np, axis=0), axis=0) + 1
            preds_ranks_mean = preds_ranks.mean(axis=0)
            labels_ranks_mean = labels_ranks.mean(axis=0)
            sp_num = ((preds_ranks - preds_ranks_mean) * (labels_ranks - labels_ranks_mean)).sum(axis=0)
            sp_denom = np.sqrt(((preds_ranks - preds_ranks_mean)**2).sum(axis=0) *
                               ((labels_ranks - labels_ranks_mean)**2).sum(axis=0))
            spearman = np.where(sp_denom == 0, np.nan, sp_num / sp_denom)

            # Create per-channel correlation dictionary.
            per_channel_corr = {}
            for ch in range(num_channels):
                if first_batch_exp_mask is not None and first_batch_exp_mask[ch, 0].item() == 0:
                    m_p, m_s = float('nan'), float('nan')
                else:
                    m_p, m_s = float(pearson[ch]), float(spearman[ch])
                #ch_loss = float(per_channel_loss[ch])
                per_channel_corr[ch] = {"mean_pearson": m_p, "mean_spearman": m_s}

            eval_results[tissue] = {
                "overall_loss": overall_loss,
                "correlations": per_channel_corr
            }

            logging.info(f"Evaluation for Tissue {tissue}:")
            logging.info(f"  Overall Loss: {overall_loss:.4f}")
            for ch in range(num_channels):
                tr_name = track_names[ch] if ch < len(track_names) else f"Track_{ch}"
                m_p = per_channel_corr[ch]["mean_pearson"]
                m_s = per_channel_corr[ch]["mean_spearman"]
                #ch_loss = per_channel_corr[ch]["mean_loss"]
                logging.info(f"  Channel {ch} ({tr_name}): Pearson: {m_p:.4f}, Spearman: {m_s:.4f}")

        return eval_results