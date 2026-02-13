#!/usr/bin/env python
"""
Fast delta-mode benchmark using the original CorgiPlus dataloader.

- Uses the training dataloader logic (CorgiPlusDataset + CorgiSampler) to stream batches.
- Selects every 3rd sequence (optional subsample) and computes delta correlations per track.
- Optional bedGraph/BigWig export with correct coordinates.
"""

import argparse
import logging
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from corgi.config import config_corgi
from corgi.config_corgiplus import config_corgiplus
from corgi.model import Corgi, CorgiPlus
from corgi.trainer_corgiplus import CorgiPlusDataset
from corgi.data_classes import CorgiSampler
from corgi.utils import load_experiment_mask


# Channel swap mapping used for reverse-complement handling
RC_FLIPPED_STRANDS = [i for i in range(22)]
RC_FLIPPED_STRANDS[12], RC_FLIPPED_STRANDS[13] = RC_FLIPPED_STRANDS[13], RC_FLIPPED_STRANDS[12]
RC_FLIPPED_STRANDS[14], RC_FLIPPED_STRANDS[15] = RC_FLIPPED_STRANDS[15], RC_FLIPPED_STRANDS[14]
RC_FLIPPED_STRANDS[16], RC_FLIPPED_STRANDS[17] = RC_FLIPPED_STRANDS[17], RC_FLIPPED_STRANDS[16]
RC_FLIPPED_STRANDS[18], RC_FLIPPED_STRANDS[19] = RC_FLIPPED_STRANDS[19], RC_FLIPPED_STRANDS[18]


BINSIZE = 64
FULL_SEQ_LEN = 524_288
TARGET_BINS = config_corgiplus["output_central_bins"]
CROP = (FULL_SEQ_LEN - TARGET_BINS * BINSIZE) // 2


def parse_tissue_ids(path_or_list: str) -> List[int]:
	path = Path(path_or_list)
	if path.exists():
		with open(path) as handle:
			return [int(x) for x in handle.read().strip().split() if x.strip()]
	return [int(x) for x in path_or_list.split(",") if x.strip()]


def crop_center(tensor: torch.Tensor, target_bins: int) -> torch.Tensor:
	start = max((tensor.shape[-1] - target_bins) // 2, 0)
	end = start + target_bins
	return tensor[..., start:end]


def load_experiments(path: Path) -> List[str]:
	with open(path, encoding="utf-8") as handle:
		return handle.read().strip().split()


def load_models(models_dir: Path, device: torch.device, dnase_global: Optional[torch.Tensor]) -> List[Dict]:
	entries: List[Dict] = []
	if not models_dir.exists():
		raise FileNotFoundError(f"Models dir not found: {models_dir}")

	def add_entry(name: str, mode: str, ckpt: Path):
		if mode in ("corgi", "corgi_delta"):
			cfg = dict(config_corgi)
			model = Corgi(cfg).to(device)
			outputs_delta = mode.endswith("delta")
			requires_aux = None
			needs_dnase_cond = False
		elif mode == "corgiplus_dnase_delta":
			if dnase_global is None:
				raise ValueError("DNase global embeddings are required for corgiplus_dnase_delta")
			cfg = dict(config_corgiplus)
			cfg["input_trans_regulators"] = dnase_global.shape[1]
			cfg["corgiplus_aux_input_dim"] = 1
			model = CorgiPlus(cfg).to(device)
			outputs_delta = True
			requires_aux = "dnase"
			needs_dnase_cond = True
		elif mode == "corgiplus_rna_delta":
			cfg = dict(config_corgiplus)
			cfg["corgiplus_aux_input_dim"] = 2
			model = CorgiPlus(cfg).to(device)
			outputs_delta = True
			requires_aux = "rna"
			needs_dnase_cond = False
		else:
			raise ValueError(f"Unsupported mode {mode}")

		state = torch.load(ckpt, map_location=device)
		state_dict = state.get("model_state_dict", state)
		model.load_state_dict(state_dict, strict=False)
		model.eval()
		entries.append(
			{
				"name": name,
				"mode": mode,
				"model": model,
				"outputs_delta": outputs_delta,
				"requires_aux": requires_aux,
				"needs_dnase_cond": needs_dnase_cond,
			}
		)

	for ckpt in sorted(models_dir.glob("*.pt")):
		stem = ckpt.stem.lower()
		if "cp_rna" in stem:
			add_entry("corgiplus_rna_delta", "corgiplus_rna_delta", ckpt)
		elif "cp_dnase" in stem:
			add_entry("corgiplus_dnase_delta", "corgiplus_dnase_delta", ckpt)
		elif "corgi_delta" in stem or 'corgidelta' in stem:
			add_entry("corgi_delta", "corgi_delta", ckpt)
		elif "corgi" in stem:
			add_entry("corgi", "corgi", ckpt)

	if not entries:
		raise RuntimeError(f"No checkpoints found in {models_dir}")
	return entries


def safe_corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
	if a.size == 0 or b.size == 0 or a.shape != b.shape:
		return float("nan"), float("nan")
	if np.std(a) == 0 or np.std(b) == 0:
		return float("nan"), float("nan")
	return pearsonr(a, b)[0], spearmanr(a, b)[0]


def export_bedgraph(lines: Iterable[Tuple[str, int, int, float]], path: Path, chrom_sizes: Path):
	lines = list(lines)
	if not lines:
		return
	lines.sort(key=lambda x: (x[0], x[1]))
	with open(path, "w") as handle:
		for chrom, start, end, val in lines:
			handle.write(f"{chrom}\t{start}\t{end}\t{val}\n")
	subprocess.run(["bedGraphToBigWig", str(path), str(chrom_sizes), str(path.with_suffix(".bw"))], check=True)


def rc_tracks(tensor: torch.Tensor) -> torch.Tensor:
	return torch.flip(tensor, dims=[-1])[:, RC_FLIPPED_STRANDS, :]


def rc_dna_batch(dna_seq: torch.Tensor) -> torch.Tensor:
	dna_rc = torch.flip(dna_seq, dims=[1])
	return dna_rc[:, :, [3, 2, 1, 0]]


def shift_dna_batch(dna_seq: torch.Tensor, shift: int) -> torch.Tensor:
	if shift == 0:
		return dna_seq
	if shift > 0:
		pad = torch.zeros((dna_seq.shape[0], shift, dna_seq.shape[2]), device=dna_seq.device, dtype=dna_seq.dtype)
		return torch.cat([dna_seq[:, shift:], pad], dim=1)
	shift_abs = abs(shift)
	pad = torch.zeros((dna_seq.shape[0], shift_abs, dna_seq.shape[2]), device=dna_seq.device, dtype=dna_seq.dtype)
	return torch.cat([pad, dna_seq[:, : dna_seq.shape[1] - shift_abs]], dim=1)


def main():
	parser = argparse.ArgumentParser(description="Delta-mode benchmark using CorgiPlus dataloader")
	parser.add_argument("--models-dir", type=Path, default=Path("/project/deeprna_data/models/revision/imputation"))
	parser.add_argument("--tissues", type=str, default=config_corgiplus["test_tissues_path"])
	parser.add_argument("--coords-bed", type=Path, default=Path(config_corgiplus["bed_path"]))
	parser.add_argument("--dna-path", type=Path, default=Path(config_corgiplus["dna_path"]))
	parser.add_argument("--tissue-dir", type=Path, default=Path(config_corgiplus["tissue_dir"]))
	parser.add_argument("--mask-path", type=Path, default=Path(config_corgiplus["mask_path"]))
	parser.add_argument("--tf-expression", type=Path, default=Path(config_corgiplus["trans_regulator_expression_path"]))
	parser.add_argument("--dnase-global", type=Path, default=Path(config_corgiplus["dnase_global_path"]))
	parser.add_argument("--mean-signal", type=Path, default=Path("/project/deeprna/data/revision/training_baseline_signal.npy"))
	parser.add_argument("--experiments", type=Path, default=Path(config_corgiplus["experiments_path"]))
	parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of non-overlapping sequences")
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--outdir", type=Path, default=Path("/project/deeprna_data/corgiplus_benchmark/imputation"))
	parser.add_argument("--export-bigwig", action="store_true")
	parser.add_argument("--chrom-sizes", type=Path, default=Path("/project/deeprna_data/corgi-reproduction/data/hg38.chrom.sizes"))
	parser.add_argument("--tta", action="store_true", help="Enable TTA with shifts (-2,0,2) and RC")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info(f"Using device: {device}")

	rng = np.random.default_rng(args.seed)
	tissues = parse_tissue_ids(args.tissues)
	logging.info(f"Tissues: {tissues}")

	coords_df = pd.read_csv(args.coords_bed, sep="\t", header=None, names=["chr", "start", "end", "fold"])
	total_seqs = len(coords_df)
	base_indices = np.arange(0, total_seqs, 3)
	if args.fraction < 1.0:
		keep = max(1, int(len(base_indices) * args.fraction))
		base_indices = np.sort(rng.choice(base_indices, size=keep, replace=False))
	logging.info(f"Evaluating {len(base_indices)} sequences (every 3rd) out of {total_seqs}")

	experiments = load_experiments(args.experiments)
	exp_name_to_channel = {name: idx for idx, name in enumerate(experiments)}
	rna_aux = (exp_name_to_channel.get("rna_total_plus"), exp_name_to_channel.get("rna_total_minus"))
	dnase_channel = exp_name_to_channel.get("dnase")

	experiment_mask = load_experiment_mask(args.mask_path)
	trans_reg_expression = torch.from_numpy(np.load(args.tf_expression)).float()
	dnase_global = torch.from_numpy(np.load(args.dnase_global)).float().to(device)
	models = load_models(args.models_dir, device, dnase_global.to(device))

	# Accumulators for correlations
	pred_delta_accum: Dict[str, Dict[Tuple[int, str], List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
	pred_full_accum: Dict[str, Dict[Tuple[int, str], List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
	gt_delta_accum: Dict[Tuple[int, str], List[np.ndarray]] = defaultdict(list)
	gt_full_accum: Dict[Tuple[int, str], List[np.ndarray]] = defaultdict(list)
	mean_full_accum: Dict[Tuple[int, str], List[np.ndarray]] = defaultdict(list)

	mean_lines: Dict[str, List[Tuple[str, int, int, float]]] = defaultdict(list)
	coords_chr = coords_df["chr"].tolist()
	coords_start = coords_df["start"].tolist()
	mean_done: set[int] = set()
	mean_written = False
	args.outdir.mkdir(parents=True, exist_ok=True)

	for tissue_id_single in tissues:
		tissue_list = [tissue_id_single]
		dataset = CorgiPlusDataset(
			dna_sequences=str(args.dna_path),
			sequence_ids=base_indices.tolist(),
			tissue_dir=str(args.tissue_dir),
			tissue_ids=tissue_list,
			experiment_mask=experiment_mask,
			trans_reg_expression=trans_reg_expression,
			output_channels=config_corgiplus["output_channels"],
			augment_dna=False,
			augment_gnomad=False,
			augment_trans_reg_std=0.0,
			gnomad_pickle=None,
			trans_reg_clip=None,
			return_mean_baseline=True,
			mean_baseline_file=str(args.mean_signal),
		)

		sampler = CorgiSampler(sequence_ids=base_indices.tolist(), tissue_ids=tissue_list, shuffled=False)
		loader = DataLoader(
			dataset,
			batch_size=args.batch_size,
			sampler=sampler,
			num_workers=args.workers,
			pin_memory=True,
			drop_last=False,
		)

		gt_lines: Dict[Tuple[int, str], List[Tuple[str, int, int, float]]] = defaultdict(list)
		gt_delta_lines: Dict[Tuple[int, str], List[Tuple[str, int, int, float]]] = defaultdict(list)
		pred_delta_lines: Dict[Tuple[str, int, str], List[Tuple[str, int, int, float]]] = defaultdict(list)
		pred_full_lines: Dict[Tuple[str, int, str], List[Tuple[str, int, int, float]]] = defaultdict(list)

		for batch_idx, batch in enumerate(loader, 1):
			dna_seq, trans_reg, label, exp_mask, mean_baseline, tissue_id, seq_id = batch
			dna_seq = dna_seq.to(device, non_blocking=True)
			trans_reg = trans_reg.to(device, non_blocking=True)
			label_full = label.to(device=device, dtype=torch.float32, non_blocking=True)
			exp_mask = exp_mask.to(device, non_blocking=True)
			mean_baseline = mean_baseline.to(device, non_blocking=True)
			tissue_id = tissue_id.to(device)
			seq_id = seq_id.to(device)

			mean_baseline = mean_baseline.to(dtype=torch.float32)
			mean_crop = crop_center(mean_baseline, TARGET_BINS)
			label_crop = crop_center(label_full, TARGET_BINS)
			gt_delta = label_crop - mean_crop
			available = exp_mask.squeeze(-1) > 0

			if args.export_bigwig:
				region_start = (
					torch.tensor([coords_start[i] for i in seq_id.tolist()], device=device, dtype=torch.long) + CROP
				)
				bin_offsets = torch.arange(TARGET_BINS, device=device, dtype=torch.long) * BINSIZE
				bin_starts = region_start[:, None] + bin_offsets[None, :]
				bin_ends = bin_starts + BINSIZE

			mask_cpu = available.cpu().numpy()
			tissues_cpu = tissue_id.cpu().numpy()
			seq_cpu = seq_id.cpu().numpy()
			gt_delta_np_all = gt_delta.detach().cpu().numpy()
			gt_abs_np_all = label_crop.detach().cpu().numpy()

			for i in range(label_crop.shape[0]):
				t_id = int(tissues_cpu[i])
				active_channels = np.nonzero(mask_cpu[i])[0]
				for c in active_channels:
					exp_name = experiments[c]
					gt_delta_accum[(t_id, exp_name)].append(gt_delta_np_all[i, c])
					gt_full_accum[(t_id, exp_name)].append(gt_abs_np_all[i, c])
					mean_full_accum[(t_id, exp_name)].append(mean_crop[i, c].detach().cpu().numpy())

			if args.export_bigwig:
				for i in range(label_crop.shape[0]):
					t_id = int(tissues_cpu[i])
					seqi = int(seq_cpu[i])
					chrom = coords_chr[seqi]
					starts_np = bin_starts[i].cpu().numpy()
					ends_np = bin_ends[i].cpu().numpy()
					active_channels = np.nonzero(mask_cpu[i])[0]
					if seqi not in mean_done:
						for c in active_channels:
							exp_name = experiments[c]
							mean_vals = mean_crop[i, c].detach().cpu().numpy().astype(float)
							mean_lines[exp_name].extend(
								zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), mean_vals.tolist())
							)
						mean_done.add(seqi)
					for c in active_channels:
						exp_name = experiments[c]
						vals_gt = label_crop[i, c].detach().cpu().numpy().astype(float)
						vals_gt_delta = gt_delta[i, c].detach().cpu().numpy().astype(float)
						gt_lines[(t_id, exp_name)].extend(
							zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), vals_gt.tolist())
						)
						gt_delta_lines[(t_id, exp_name)].extend(
							zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), vals_gt_delta.tolist())
						)

			with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
				for entry in models:
					cond = dnase_global[tissue_id] if entry["needs_dnase_cond"] else trans_reg

					if entry["requires_aux"] == "rna" and rna_aux[0] is not None and rna_aux[1] is not None:
						aux_pos = label_full[:, [rna_aux[0], rna_aux[1]], :].permute(0, 2, 1)
						aux_rc = rc_tracks(label_full)[:, [rna_aux[0], rna_aux[1]], :].permute(0, 2, 1) if args.tta else None
					elif entry["requires_aux"] == "dnase" and dnase_channel is not None:
						aux_pos = label_full[:, dnase_channel : dnase_channel + 1, :].permute(0, 2, 1)
						aux_rc = rc_tracks(label_full)[:, dnase_channel : dnase_channel + 1, :].permute(0, 2, 1) if args.tta else None
					elif entry["requires_aux"] == "dnase" and dnase_channel is None:
						raise ValueError("DNase auxiliary channel not found in experiments list")
					else:
						aux_pos = None
						aux_rc = None

					outputs_delta = entry["outputs_delta"]

					def forward_raw(dna_input: torch.Tensor, aux_input: Optional[torch.Tensor]):
						pred = entry["model"](dna_input, aux_input, cond) if entry["requires_aux"] else entry["model"](dna_input, cond)
						return crop_center(pred.float(), TARGET_BINS)

					def add_baseline_and_clamp(pred: torch.Tensor, mean_ref: torch.Tensor) -> torch.Tensor:
						if outputs_delta:
							pred = pred + mean_ref
						return pred.clamp(min=0.0)

					if args.tta:
						agg = None
						count = 0
						for shift in (-2, 0, 2):
							dna_shifted = shift_dna_batch(dna_seq, shift)
							pred_pos = forward_raw(dna_shifted, aux_pos)
							pred_pos = add_baseline_and_clamp(pred_pos, mean_crop)
							agg = pred_pos if agg is None else agg + pred_pos
							count += 1

							dna_seq_rc = rc_dna_batch(dna_shifted)
							mean_crop_rc = rc_tracks(mean_crop)
							pred_rc = forward_raw(dna_seq_rc, aux_rc)
							pred_rc = add_baseline_and_clamp(pred_rc, mean_crop_rc)
							pred_rc = rc_tracks(pred_rc)
							agg = agg + pred_rc
							count += 1

						pred_full = agg / count
					else:
						pred_full = add_baseline_and_clamp(forward_raw(dna_seq, aux_pos), mean_crop)

					pred_delta = pred_full - mean_crop

					pred_delta_np = pred_delta.detach().cpu().numpy()
					pred_abs_np = pred_full.detach().cpu().numpy()
					gt_delta_np = gt_delta.detach().cpu().numpy()
					gt_abs_np = label_crop.detach().cpu().numpy()

					for i in range(pred_delta_np.shape[0]):
						t_id = int(tissues_cpu[i])
						active_channels = np.nonzero(mask_cpu[i])[0]
						if args.export_bigwig:
							chrom = coords_chr[int(seq_cpu[i])]
							starts_np = bin_starts[i].cpu().numpy()
							ends_np = bin_ends[i].cpu().numpy()
						for c in active_channels:
							exp_name = experiments[c]
							pred_delta_accum[entry["name"]][(t_id, exp_name)].append(pred_delta_np[i, c])
							pred_full_accum[entry["name"]][(t_id, exp_name)].append(pred_abs_np[i, c])

							if args.export_bigwig:
								vals_pred_delta = pred_delta_np[i, c].astype(float)
								vals_pred_full = pred_abs_np[i, c].astype(float)
								pred_delta_lines[(entry["name"], t_id, exp_name)].extend(
									zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), vals_pred_delta.tolist())
								)
								pred_full_lines[(entry["name"], t_id, exp_name)].extend(
									zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), vals_pred_full.tolist())
								)

			if batch_idx % 100 == 0:
				logging.info(f"Processed {batch_idx} batches / {len(loader)}")

		if args.export_bigwig:
			if mean_lines and not mean_written:
				for exp, lines in mean_lines.items():
					export_bedgraph(lines, args.outdir / f"mean_{exp}.bedgraph", args.chrom_sizes)
				mean_lines.clear()
				mean_written = True
			for (t_id, exp), lines in gt_lines.items():
				export_bedgraph(lines, args.outdir / f"gt_t{t_id}_{exp}.bedgraph", args.chrom_sizes)
			for (t_id, exp), lines in gt_delta_lines.items():
				export_bedgraph(lines, args.outdir / f"gt_delta_t{t_id}_{exp}.bedgraph", args.chrom_sizes)
			for (model_name, t_id, exp), lines in pred_delta_lines.items():
				export_bedgraph(lines, args.outdir / f"pred_delta_{model_name}_t{t_id}_{exp}.bedgraph", args.chrom_sizes)
			#for (model_name, t_id, exp), lines in pred_full_lines.items():
			#	export_bedgraph(lines, args.outdir / f"pred_full_{model_name}_t{t_id}_{exp}.bedgraph", args.chrom_sizes)
			gt_lines.clear()
			gt_delta_lines.clear()
			pred_delta_lines.clear()
			pred_full_lines.clear()

	corr_delta_rows: List[Dict] = []
	corr_full_rows: List[Dict] = []

	# Delta correlations per model
	for entry in models:
		name = entry["name"]
		for key, preds in pred_delta_accum[name].items():
			gt_list = gt_delta_accum.get(key, [])
			if not gt_list:
				continue
			pred_concat = np.concatenate(preds)
			gt_concat = np.concatenate(gt_list)
			p, s = safe_corr(pred_concat, gt_concat)
			corr_delta_rows.append(
				{
					"model": name,
					"mode": entry["mode"],
					"tissue": key[0],
					"experiment": key[1],
					"pearson": round(p, 3) if not math.isnan(p) else np.nan,
					"spearman": round(s, 3) if not math.isnan(s) else np.nan,
				}
			)

	# Full-signal correlations per model
	for entry in models:
		name = entry["name"]
		for key, preds in pred_full_accum[name].items():
			gt_list = gt_full_accum.get(key, [])
			if not gt_list:
				continue
			pred_concat = np.concatenate(preds)
			gt_concat = np.concatenate(gt_list)
			p, s = safe_corr(pred_concat, gt_concat)
			corr_full_rows.append(
				{
					"model": name,
					"mode": entry["mode"],
					"tissue": key[0],
					"experiment": key[1],
					"pearson": round(p, 3) if not math.isnan(p) else np.nan,
					"spearman": round(s, 3) if not math.isnan(s) else np.nan,
				}
			)

	# Baseline mean vs GT (full signal)
	for key, gt_list in gt_full_accum.items():
		mean_list = mean_full_accum.get(key, [])
		if not mean_list or len(mean_list) != len(gt_list):
			continue
		mean_concat = np.concatenate(mean_list)
		gt_concat = np.concatenate(gt_list)
		p, s = safe_corr(mean_concat, gt_concat)
		corr_full_rows.append(
			{
				"model": "mean_baseline",
				"mode": "baseline",
				"tissue": key[0],
				"experiment": key[1],
				"pearson": round(p, 3) if not math.isnan(p) else np.nan,
				"spearman": round(s, 3) if not math.isnan(s) else np.nan,
			}
		)

	args.outdir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(corr_delta_rows).to_csv(args.outdir / "correlation_delta.csv", index=False)
	pd.DataFrame(corr_full_rows).to_csv(args.outdir / "correlation_full.csv", index=False)
	logging.info("Wrote correlation_delta.csv and correlation_full.csv")


if __name__ == "__main__":
	main()
