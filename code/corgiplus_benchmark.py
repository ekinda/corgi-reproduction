#!/usr/bin/env python
import sys
import os
from pathlib import Path
import argparse
import logging
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
from pyfaidx import Fasta
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path('/project/deeprna_data/corgi-reproduction')
sys.path.insert(0, str(REPO_ROOT / "code"))
DATA_DIR = Path('/project/deeprna_data/corgi-reproduction/data')

from benchmark_utils import (
    transform_params,
    special_atac_params,
    transform_softclip,
    transform_scale,
    transform_scale_special,
    undo_squashed_scale,
    parse_bed_file_with_coords,
    one_hot_encode,
    load_genome,
    tile_regions,
    crop_center,
    dna_rc,
    predictions_rc,
    shift_dna,
    process_coverage,
    fast_bin,
    experiments_list,
)
from corgi.config import config_corgi
from corgi.config_corgiplus import config_corgiplus
from corgi.model import Corgi, CorgiPlus

PRED_SEQ_LENGTH = 524288
PRED_STRIDE = 393216
CROP = 65536
GT_SEQ_LENGTH = 393216
BINSIZE = 64
TARGET_BINS = 6144
AUX_BINS = PRED_SEQ_LENGTH // BINSIZE

with open('/project/deeprna/data/experiments_final.txt', 'r', encoding='utf-8') as f:
    experiments_list_local = f.read().strip().split()

exp_name_to_channel_id = {exp: i for i, exp in enumerate(experiments_list_local)}

regular_tracks = ['dnase','atac','h3k4me1','h3k4me2','h3k4me3','h3k9ac','h3k9me3','h3k27ac','h3k27me3','h3k36me3','h3k79me2','ctcf','rna_10x','wgbs']
stranded_tracks = ['cage', 'rampage', 'rna_total', 'rna_polya']


def export_bedgraph_and_bigwig(df: pd.DataFrame, model_type: str, outdir: Path, chrom_sizes: Path, tracks_to_export='all'):
    for (tissue, experiment), group in df.groupby(["tissue", "experiment"]):
        if tracks_to_export is not 'all':
            if experiment.lower() not in tracks_to_export:
                continue
        bedgraph_lines: List[str] = []
        for _, row in group.iterrows():
            chrom = row["chr"]
            tile_start = int(row["start"])
            region_start = tile_start + CROP
            bin_scores = row["values"].split(",")
            for i, score in enumerate(bin_scores):
                bin_start = region_start + i * BINSIZE
                bin_end = bin_start + BINSIZE
                bedgraph_lines.append(f"{chrom}\t{bin_start}\t{bin_end}\t{score}")
        bg_path = outdir / f"tissue{tissue}_{experiment}_{model_type}.bedgraph"
        bw_path = outdir / f"tissue{tissue}_{experiment}_{model_type}.bw"
        with open(bg_path, "w") as f:
            f.write("\n".join(bedgraph_lines))
        subprocess.run(["bedGraphToBigWig", str(bg_path), str(chrom_sizes), str(bw_path)], check=True)
        logging.info(f"Exported {model_type} BigWig for tissue {tissue}, experiment {experiment} to {bw_path}")


def extract_gt_tile(gt_data_dir: Path, tissue_id: int, chrom: str, tile_start: int, track_name: str) -> np.ndarray:
    params = transform_params.get(track_name, {})
    if track_name.lower() == 'atac' and 464 <= tissue_id <= 480:
        params = special_atac_params
    sum_stat = params.get('sum_stat', 'mean')

    gt_file = os.path.join(gt_data_dir, str(tissue_id), f"{track_name}.h5")
    if not os.path.exists(gt_file):
        return None
    gt_start = tile_start + CROP
    gt_end = gt_start + GT_SEQ_LENGTH
    with h5py.File(gt_file, 'r') as hf:
        if chrom not in hf:
            return None
        data = hf[chrom][gt_start:gt_end]
        if len(data) < GT_SEQ_LENGTH:
            return None
        num_bins = GT_SEQ_LENGTH // BINSIZE
        data = data[:num_bins * BINSIZE]
        data = process_coverage(data, params)
        binned = fast_bin(data, sum_stat, n_bins=TARGET_BINS, bin_size=BINSIZE)
        return binned


def extract_aux_tile(gt_data_dir: Path, tissue_id: int, chrom: str, tile_start: int, track_name: str) -> Optional[np.ndarray]:
    """Load the full-length auxiliary track (8192 bins) used by CorgiPlus."""
    params = transform_params.get(track_name, {})
    if track_name.lower() == 'atac' and 464 <= tissue_id <= 480:
        params = special_atac_params
    sum_stat = params.get('sum_stat', 'mean')

    gt_file = os.path.join(gt_data_dir, str(tissue_id), f"{track_name}.h5")
    if not os.path.exists(gt_file):
        return None
    window_start = tile_start
    window_end = tile_start + PRED_SEQ_LENGTH
    with h5py.File(gt_file, 'r') as hf:
        if chrom not in hf:
            return None
        data = hf[chrom][window_start:window_end]
        if len(data) < PRED_SEQ_LENGTH:
            return None
        data = data[:PRED_SEQ_LENGTH]
        data = process_coverage(data, params)
        binned = fast_bin(data, sum_stat, n_bins=AUX_BINS, bin_size=BINSIZE)
        return binned


def load_model_entry(name: str, mode: str, ckpt_path: Path, device: torch.device, dnase_global: torch.Tensor, trans_reg: torch.Tensor):
    if mode in ('corgi', 'corgi_finetune_gnomad', 'corgi_finetune_nognomad'):
        cfg = dict(config_corgi)
        model = Corgi(cfg).to(device)
    elif mode == 'corgiplus_rna':
        cfg = dict(config_corgiplus)
        cfg['corgiplus_aux_input_dim'] = 2
        model = CorgiPlus(cfg).to(device)
    elif mode == 'corgiplus_dnase':
        cfg = dict(config_corgiplus)
        cfg['input_trans_regulators'] = dnase_global.shape[1]
        cfg['corgiplus_aux_input_dim'] = 1
        model = CorgiPlus(cfg).to(device)
    else:
        raise ValueError(f"Unsupported mode {mode}")

    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('model_state_dict', state)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return {
        'name': name,
        'mode': mode,
        'model': model,
    }


def predict_tile(entry: Dict, seq_tensor: torch.Tensor, tissue_id: int, trans_reg: torch.Tensor, dnase_global: torch.Tensor, device: torch.device, aux_inputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    mode = entry['mode']
    model = entry['model']
    preds = []
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16):
            for shift in [-2, 0, 2]:
                seq = shift_dna(seq_tensor, shift)
                if mode in ('corgi', 'corgi_finetune_gnomad', 'corgi_finetune_nognomad'):
                    cond = trans_reg[tissue_id].unsqueeze(0).to(device)
                    pred = model(seq, cond)
                    aux = None
                elif mode == 'corgiplus_rna':
                    if aux_inputs is None or 'rna_total' not in aux_inputs:
                        raise ValueError("Missing RNA auxiliary inputs for corgiplus_rna")
                    cond = trans_reg[tissue_id].unsqueeze(0).to(device)
                    aux = aux_inputs['rna_total']
                    pred = model(seq, aux.permute(0, 2, 1), cond)
                elif mode == 'corgiplus_dnase':
                    if aux_inputs is None or 'dnase' not in aux_inputs:
                        raise ValueError("Missing DNase auxiliary inputs for corgiplus_dnase")
                    cond = dnase_global[tissue_id].unsqueeze(0)
                    aux = aux_inputs['dnase']
                    pred = model(seq, aux.permute(0, 2, 1), cond)
                else:
                    raise ValueError(mode)
                preds.append(pred)

                seq_rc = dna_rc(seq)
                if mode in ('corgi', 'corgi_finetune_gnomad', 'corgi_finetune_nognomad'):
                    pred_rc = model(seq_rc, cond)
                elif mode == 'corgiplus_rna':
                    aux_rc = torch.flip(aux, dims=[2])[:, [1, 0], :]
                    pred_rc = model(seq_rc, aux_rc.permute(0, 2, 1), cond)
                elif mode == 'corgiplus_dnase':
                    aux_rc = torch.flip(aux, dims=[2])
                    pred_rc = model(seq_rc, aux_rc.permute(0, 2, 1), cond)
                else:
                    pred_rc = model(seq_rc, aux.permute(0, 2, 1), cond)
                preds.append(predictions_rc(pred_rc))

    preds = [x.float() for x in preds]
    outputs = torch.cat(preds).mean(dim=0)
    outputs = crop_center(outputs, TARGET_BINS)
    return outputs.squeeze(0)


def safe_corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if a.size == 0 or b.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan'), float('nan')
    return pearsonr(a, b)[0], spearmanr(a, b)[0]


def parse_tissue_ids(path_or_list: str) -> List[int]:
    p = Path(path_or_list)
    if p.exists():
        with open(p) as f:
            return [int(x) for x in f.read().strip().split() if x.strip()]
    return [int(x) for x in path_or_list.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Corgi/CorgiPlus checkpoints with TTA and export BigWigs + correlations.")
    parser.add_argument('--bed', type=Path, required=True, help='BED file with regions to tile')
    parser.add_argument('--tissues', type=str, required=True, help='Path or comma-separated tissue ids')
    parser.add_argument('--genome', type=Path, default=DATA_DIR / 'hg38.ml.fa')
    parser.add_argument('--gt-dir', type=Path, default=DATA_DIR / 'ground_truth')
    parser.add_argument('--tf-expression', type=Path, default=DATA_DIR / 'tf_expression.npy')
    parser.add_argument('--chrom-sizes', type=Path, default=DATA_DIR / 'hg38.chrom.sizes')
    parser.add_argument('--outdir', type=Path, default=REPO_ROOT / 'processed_data' / 'corgiplus_benchmark')
    parser.add_argument('--models', nargs=3, action='append', metavar=('NAME', 'MODE', 'CKPT'), required=True, help='Repeat: NAME MODE CKPT')
    parser.add_argument('--export-bigwig', action='store_true', help='Export BigWig files')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    tissues = parse_tissue_ids(args.tissues)
    logging.info(f"Tissues: {tissues}")

    bed_regions = parse_bed_file_with_coords(args.bed)
    logging.info(f"Parsed {len(bed_regions)} regions from BED file.")

    pred_tiles = tile_regions(bed_regions, PRED_SEQ_LENGTH, PRED_STRIDE, drop_last=True)
    logging.info(f"Generated {len(pred_tiles)} prediction tiles.")

    genome = load_genome(args.genome)
    logging.info("Genome loaded.")

    tf_expression = torch.from_numpy(np.load(args.tf_expression)).float().to(device)
    dnase_global = None
    if any(m[1] == 'corgiplus_dnase' for m in args.models):
        dnase_global = torch.from_numpy(np.load(config_corgiplus['dnase_global_path'])).float().to(device)

    model_entries = []
    for name, mode, ckpt in args.models:
        entry = load_model_entry(name, mode, Path(ckpt), device, dnase_global, tf_expression)
        model_entries.append(entry)
        logging.info(f"Loaded model {name} mode={mode} from {ckpt}")

    requires_rna_aux = any(entry['mode'] == 'corgiplus_rna' for entry in model_entries)
    requires_dnase_aux = any(entry['mode'] == 'corgiplus_dnase' for entry in model_entries)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    pred_rows: Dict[str, List[Dict]] = defaultdict(list)
    gt_rows: List[Dict] = []
    pred_accum: Dict[str, Dict[Tuple[int, str], List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    gt_accum: Dict[Tuple[int, str], List[np.ndarray]] = defaultdict(list)

    for tissue_id in tissues:
        logging.info(f"Processing tissue {tissue_id}")
        tf_exp_tissue = tf_expression[tissue_id].unsqueeze(0)
        for chrom, tile_start, tile_end in pred_tiles:
            try:
                seq = genome[chrom][tile_start:tile_end].seq
            except Exception as e:
                logging.warning(f"Error fetching sequence for {chrom}:{tile_start}-{tile_end}: {e}")
                continue
            seq_encoded = one_hot_encode(seq)
            seq_tensor = torch.from_numpy(seq_encoded).unsqueeze(0).to(device)

            aux_inputs: Dict[str, torch.Tensor] = {}
            if requires_rna_aux:
                aux_rna_plus = extract_aux_tile(args.gt_dir, tissue_id, chrom, tile_start, 'rna_total_plus')
                aux_rna_minus = extract_aux_tile(args.gt_dir, tissue_id, chrom, tile_start, 'rna_total_minus')
                if aux_rna_plus is None or aux_rna_minus is None:
                    logging.warning(f"Missing RNA aux for tissue {tissue_id} at {chrom}:{tile_start}-{tile_end}; skipping RNA-conditioned models for this tile.")
                else:
                    aux_rna = np.stack([aux_rna_plus, aux_rna_minus], axis=0)
                    aux_inputs['rna_total'] = torch.from_numpy(aux_rna).unsqueeze(0).to(device=device, dtype=seq_tensor.dtype)

            if requires_dnase_aux:
                aux_dnase = extract_aux_tile(args.gt_dir, tissue_id, chrom, tile_start, 'dnase')
                if aux_dnase is None:
                    logging.warning(f"Missing DNase aux for tissue {tissue_id} at {chrom}:{tile_start}-{tile_end}; skipping DNase-conditioned models for this tile.")
                else:
                    aux_inputs['dnase'] = torch.from_numpy(aux_dnase).unsqueeze(0).unsqueeze(0).to(device=device, dtype=seq_tensor.dtype)

            pred_cache = {}
            for entry in model_entries:
                try:
                    pred_cache[entry['name']] = predict_tile(entry, seq_tensor, tissue_id, tf_expression, dnase_global, device, aux_inputs)
                except ValueError as exc:
                    logging.warning(f"{entry['name']} skipped for {chrom}:{tile_start}-{tile_end}: {exc}")
                    continue

            # Regular tracks
            for exp_name in regular_tracks:
                gt_array = extract_gt_tile(args.gt_dir, tissue_id, chrom, tile_start, exp_name)
                if gt_array is None:
                    continue
                channel_id = exp_name_to_channel_id[exp_name]
                for entry in model_entries:
                    if entry['name'] not in pred_cache:
                        continue
                    pred_array = pred_cache[entry['name']][channel_id].detach().cpu().numpy()
                    pred_rows[entry['name']].append({
                        "tissue": tissue_id,
                        "experiment": exp_name,
                        "chr": chrom,
                        "start": tile_start,
                        "end": tile_end,
                        "values": ",".join([f"{v:.3f}" for v in pred_array])
                    })
                    pred_accum[entry['name']][(tissue_id, exp_name)].append(pred_array)
                gt_rows.append({
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_array])
                })
                gt_accum[(tissue_id, exp_name)].append(gt_array)

            # Stranded tracks
            for exp_name in stranded_tracks:
                plus_name = exp_name + '_plus'
                minus_name = exp_name + '_minus'
                gt_plus = extract_gt_tile(args.gt_dir, tissue_id, chrom, tile_start, plus_name)
                gt_minus = extract_gt_tile(args.gt_dir, tissue_id, chrom, tile_start, minus_name)
                if gt_plus is None or gt_minus is None:
                    continue
                channel_id_plus = exp_name_to_channel_id[plus_name]
                channel_id_minus = exp_name_to_channel_id[minus_name]
                for entry in model_entries:
                    if entry['name'] not in pred_cache:
                        continue
                    pred_plus = pred_cache[entry['name']][channel_id_plus].detach().cpu().numpy()
                    pred_minus = pred_cache[entry['name']][channel_id_minus].detach().cpu().numpy()
                    pred_rows[entry['name']].append({
                        "tissue": tissue_id,
                        "experiment": plus_name,
                        "chr": chrom,
                        "start": tile_start,
                        "end": tile_end,
                        "values": ",".join([f"{v:.3f}" for v in pred_plus])
                    })
                    pred_rows[entry['name']].append({
                        "tissue": tissue_id,
                        "experiment": minus_name,
                        "chr": chrom,
                        "start": tile_start,
                        "end": tile_end,
                        "values": ",".join([f"{v:.3f}" for v in pred_minus])
                    })
                    pred_accum[entry['name']][(tissue_id, plus_name)].append(pred_plus)
                    pred_accum[entry['name']][(tissue_id, minus_name)].append(pred_minus)

                    pred_total = pred_plus + pred_minus
                    pred_rows[entry['name']].append({
                        "tissue": tissue_id,
                        "experiment": exp_name,
                        "chr": chrom,
                        "start": tile_start,
                        "end": tile_end,
                        "values": ",".join([f"{v:.3f}" for v in pred_total])
                    })
                    pred_accum[entry['name']][(tissue_id, exp_name)].append(pred_total)

                gt_rows.append({
                    "tissue": tissue_id,
                    "experiment": plus_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_plus])
                })
                gt_rows.append({
                    "tissue": tissue_id,
                    "experiment": minus_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_minus])
                })
                gt_total = gt_plus + gt_minus
                gt_rows.append({
                    "tissue": tissue_id,
                    "experiment": exp_name,
                    "chr": chrom,
                    "start": tile_start,
                    "end": tile_end,
                    "values": ",".join([f"{v:.3f}" for v in gt_total])
                })
                gt_accum[(tissue_id, plus_name)].append(gt_plus)
                gt_accum[(tissue_id, minus_name)].append(gt_minus)
                gt_accum[(tissue_id, exp_name)].append(gt_total)

    # Correlations
    corr_results = []
    for entry in model_entries:
        name = entry['name']
        for key, preds in pred_accum[name].items():
            tissue_id, exp_name = key
            gt_list = gt_accum.get(key, [])
            if not gt_list:
                continue
            pred_concat = np.concatenate(preds)
            gt_concat = np.concatenate(gt_list)
            p, s = safe_corr(pred_concat, gt_concat)
            corr_results.append({
                "model": name,
                "mode": entry['mode'],
                "tissue": tissue_id,
                "experiment": exp_name,
                "pearson": round(p, 3) if not np.isnan(p) else np.nan,
                "spearman": round(s, 3) if not np.isnan(s) else np.nan,
            })

    corr_df = pd.DataFrame(corr_results)
    corr_path = outdir / "correlation_results.csv"
    corr_df.to_csv(corr_path, index=False)
    logging.info(f"Correlation results written to {corr_path}")

    # Save predictions and GT as CSV for reference
    for entry in model_entries:
        df = pd.DataFrame(pred_rows[entry['name']])
        pred_path = outdir / f"predictions_{entry['name']}.csv"
        df.to_csv(pred_path, index=False)
        logging.info(f"Predictions for {entry['name']} written to {pred_path}")

    gt_df = pd.DataFrame(gt_rows)
    gt_path = outdir / "ground_truth.csv"
    gt_df.to_csv(gt_path, index=False)
    logging.info(f"Ground truth written to {gt_path}")

    if args.export_bigwig:
        for entry in model_entries:
            df = pd.DataFrame(pred_rows[entry['name']])
            export_bedgraph_and_bigwig(df, entry['name'], outdir, args.chrom_sizes)
        export_bedgraph_and_bigwig(gt_df, "encode", outdir, args.chrom_sizes)


if __name__ == "__main__":
    main()
