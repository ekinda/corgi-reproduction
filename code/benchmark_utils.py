"""Shared utilities for the benchmark scripts that compare Corgi predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pyfaidx import Fasta
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

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

with open(REPO_ROOT / "data" / "experiments_final.txt", "r", encoding="utf-8") as handle:
    experiments_list = handle.read().strip().split()

# Precompute tensors for undoing squashed scale.
_softclips = [transform_params[exp]["soft_clip"] for exp in experiments_list]
_scales = [transform_params[exp]["scale"] for exp in experiments_list]

transform_softclip = torch.tensor(_softclips, dtype=torch.float32).view(1, -1, 1)
transform_scale = torch.tensor(_scales, dtype=torch.float32).view(1, -1, 1)
transform_scale_special = transform_scale.clone()
if transform_scale_special.shape[1] > 1:
    transform_scale_special[:, 1, :] = special_atac_params["scale"]

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
    scale = transform_scale_special.to(device) if 464 <= tissue_id <= 480 else transform_scale.to(device)
    mask = coverage > clip
    adjusted = (coverage - clip) ** 2 + clip
    coverage = torch.where(mask, adjusted, coverage)
    coverage = coverage / scale
    return coverage

def parse_bed_file_with_coords(bed_path: Path) -> List[Tuple[str, int, int]]:
    """Parse a BED file and return coordinate tuples."""
    df = pd.read_csv(bed_path, sep="\t", header=None, names=["chr", "start", "end", "fold"])
    coords: list[tuple[str, int, int]] = []
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

def load_genome(fasta_path: Path) -> Fasta:
    """Load the reference genome using pyfaidx."""
    return Fasta(str(fasta_path))

def tile_region(chrom: str, start: int, end: int, sequence_length: int, stride: int, drop_last: bool = True) -> List[Tuple[str, int, int]]:
    """Generate fixed-length tiles inside a single region."""
    tiles: list[tuple[str, int, int]] = []
    region_length = end - start
    if region_length < sequence_length:
        return tiles
    current_start = start
    while current_start + sequence_length <= end:
        tiles.append((chrom, current_start, current_start + sequence_length))
        current_start += stride
    if not drop_last and current_start < end:
        tiles.append((chrom, end - sequence_length, end))
    return tiles

def tile_regions(regions: Sequence[Tuple[str, int, int]], sequence_length: int, stride: int, drop_last: bool = True) -> List[Tuple[str, int, int]]:
    """Expand multiple regions into tiles."""
    tiled: list[tuple[str, int, int]] = []
    for chrom, start, end in regions:
        tiled.extend(tile_region(chrom, start, end, sequence_length, stride, drop_last))
    return tiled

def crop_center(tensor: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Crop the last dimension of a tensor to the central window."""
    length = tensor.shape[-1]
    start = max((length - target_bins) // 2, 0)
    end = start + target_bins
    return tensor[..., start:end]

def dna_rc(sequence: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a one-hot encoded DNA tensor of shape (1, L, 4)."""
    sequence = torch.flip(sequence, dims=[1])
    sequence = sequence[..., [3, 2, 1, 0]]
    return sequence

def predictions_rc(pred: torch.Tensor) -> torch.Tensor:
    """Reverse-complement model predictions while swapping strand-specific channels."""
    strand_map = list(range(pred.shape[1]))
    swaps = [(12, 13), (14, 15), (16, 17), (18, 19)]
    for src, dst in swaps:
        if src < len(strand_map) and dst < len(strand_map):
            strand_map[src], strand_map[dst] = strand_map[dst], strand_map[src]

    pred = torch.flip(pred, dims=[2])
    pred = pred[:, strand_map, :]
    return pred
    
def shift_dna(sequence: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift a one-hot encoded sequence left or right by padding with zeros."""
    device = sequence.device
    if shift == 0:
        return sequence
    pad = torch.zeros(1, abs(shift), 4, device=device)
    if shift > 0:
        return torch.cat([pad, sequence[:, :-shift, :]], dim=1)
    return torch.cat([sequence[:, -shift:, :], pad], dim=1)