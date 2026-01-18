import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from corgi.config import config_corgi
from corgi.config_corgiplus import config_corgiplus
from corgi.model import Corgi, CorgiPlus, CorgiPlusNofilm
from corgi.trainer_corgiplus import CorgiPlusDataset
from corgi.data_classes import CorgiSampler
from corgi.utils import load_experiment_mask

from benchmark_utils import crop_center, dna_rc, predictions_rc, shift_dna

SEQ_LENGTH = 524_288
STRIDE = 393_216
CROP = 65536
BINSIZE = 64
TARGET_BINS = 6144
TTA_SHIFTS = (-2, 0, 2)
RC_FLIP_PAIRS = ((12, 13), (14, 15), (16, 17), (18, 19))
DEFAULT_EXPORT_TRACKS = [
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
]


def _shift_dna_batch(sequence: torch.Tensor, shift: int) -> torch.Tensor:
	"""Batch-aware shift of one-hot DNA; pads with zeros."""
	if shift == 0:
		return sequence
	batch = sequence.shape[0]
	pad = torch.zeros(batch, abs(shift), 4, device=sequence.device, dtype=sequence.dtype)
	if shift > 0:
		return torch.cat([pad, sequence[:, :-shift, :]], dim=1)
	return torch.cat([sequence[:, -shift:, :], pad], dim=1)

def load_model(checkpoint_path: str, model_type: str, device: torch.device, dnase_global: Optional[torch.Tensor] = None):
	if model_type == 'corgi':
		cfg = config_corgi.copy()
		cfg['final_softplus'] = True
		model = Corgi(cfg)
	elif model_type == 'big_corgi':
		cfg = config_corgi.copy()
		cfg['final_softplus'] = False
		model = Corgi(cfg)
	elif model_type in ('corgiplus', 'corgiplus_impute', 'corgiplus_rna', 'corgiplus_dnase', 'corgiplusplus_rna', 'corgiplus_rna_nofilm'):
		cfg = config_corgiplus.copy()
		if model_type == 'corgiplus_rna':
			cfg['corgiplus_aux_input_dim'] = 2 + cfg['output_channels']
		elif model_type == 'corgiplus_dnase':
			if dnase_global is None:
				raise ValueError("DNase global features are required for corgiplus_dnase models")
			cfg['input_trans_regulators'] = dnase_global.shape[1]
			cfg['corgiplus_aux_input_dim'] = 1 + cfg['output_channels']
		elif model_type == 'corgiplusplus_rna':
			cfg['corgiplus_aux_input_dim'] = cfg['output_channels'] + 4
		elif model_type == 'corgiplus_rna_nofilm':
			cfg['corgiplus_aux_input_dim'] = 2 + cfg['output_channels']
		else:  # corgiplus_impute or generic corgiplus
			cfg['corgiplus_aux_input_dim'] = cfg['output_channels']

		if model_type == 'corgiplus_rna_nofilm':
			model = CorgiPlusNofilm(cfg)
		else:
			model = CorgiPlus(cfg)
	else:
		raise ValueError(f"Unknown model type: {model_type}")

	model = model.to(device)
	state = torch.load(checkpoint_path, map_location=device)
	state_dict = state.get('model_state_dict', state)
	model.load_state_dict(state_dict)
	model.eval()
	return model


def infer_model_type(name: str) -> str:
	name = name.lower()
	if 'corgiplusplus' in name:
		return 'corgiplusplus_rna'
	if 'rna_nofilm' in name:
		return 'corgiplus_rna_nofilm'
	if 'corgiplus_impute' in name:
		return 'corgiplus_impute'
	if 'rna' in name:
		return 'corgiplus_rna'
	if 'dnase' in name:
		return 'corgiplus_dnase'
	if 'big_corgi' in name:
		return 'big_corgi'
	else:
		return 'corgi'

def _flip_strand_pairs(tensor: torch.Tensor, offset: int = 0) -> torch.Tensor:
	"""Swap predefined strand pairs in-place on a cloned tensor."""
	out = tensor.clone()
	for src, dst in RC_FLIP_PAIRS:
		src_i, dst_i = src + offset, dst + offset
		if src_i < out.shape[1] and dst_i < out.shape[1]:
			out[:, [src_i, dst_i], :] = out[:, [dst_i, src_i], :]
	return out


def rc_auxiliary(aux: torch.Tensor, model_type: str) -> torch.Tensor:
	"""Reverse-complement auxiliary inputs according to model type."""
	if aux is None:
		return aux

	def flip_last(x: torch.Tensor) -> torch.Tensor:
		return torch.flip(x, dims=[-1])

	if model_type == 'corgiplus_dnase':
		dnase = flip_last(aux[:, 0:1, :])
		mb = flip_last(aux[:, 1:, :])
		mb = _flip_strand_pairs(mb, offset=0)
		return torch.cat([dnase, mb], dim=1)

	if model_type in ('corgiplus_rna', 'corgiplus_rna_nofilm'):
		rna = flip_last(aux[:, 0:2, :])
		rna = rna[:, [1, 0], :]
		mb = flip_last(aux[:, 2:, :])
		mb = _flip_strand_pairs(mb, offset=0)
		return torch.cat([rna, mb], dim=1)

	if model_type == 'corgiplusplus_rna':
		rna = flip_last(aux[:, 0:2, :])
		rna = rna[:, [1, 0], :]
		mb_end = 2 + config_corgiplus["output_channels"]
		mb = flip_last(aux[:, 2:mb_end, :])
		mb = _flip_strand_pairs(mb, offset=0)
		delta = flip_last(aux[:, mb_end:, :])
		delta = delta[:, [1, 0], :]
		return torch.cat([rna, mb, delta], dim=1)

	# Generic corgiplus / corgiplus_impute: aux is mean baseline only.
	mb = flip_last(aux)
	mb = _flip_strand_pairs(mb, offset=0)
	return mb


def pred_tta(model: torch.nn.Module, dna_seq: torch.Tensor, trans_reg: Optional[torch.Tensor], model_type: str, aux: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
	"""Run model with TTA (shifts and reverse complement) and average predictions."""
	device_type = dna_seq.device.type
	aux_rc = rc_auxiliary(aux, model_type) if aux is not None else None

	def forward(seq_in: torch.Tensor, aux_in: Optional[torch.Tensor]) -> torch.Tensor:
		if model_type == 'corgi':
			return model(seq_in, trans_reg)
		if model_type == 'big_corgi':
			pred = model(seq_in, trans_reg)
			return torch.clamp(pred, min=0.0)
		if model_type == 'corgiplus_dnase':
			if cond is None:
				raise ValueError("DNase conditioning tensor is required for corgiplus_dnase")
			return model(seq_in, aux_in.permute(0, 2, 1), cond)
		if model_type == 'corgiplus_rna':
			return model(seq_in, aux_in.permute(0, 2, 1), trans_reg)
		if model_type == 'corgiplus_rna_nofilm':
			return model(seq_in, aux_in.permute(0, 2, 1), None)
		if model_type == 'corgiplusplus_rna':
			return model(seq_in, aux_in.permute(0, 2, 1), trans_reg)
		return model(seq_in, aux_in.permute(0, 2, 1), trans_reg)

	preds: List[torch.Tensor] = []
	with torch.no_grad():
		with torch.autocast(device_type, dtype=torch.bfloat16):
			for shift in TTA_SHIFTS:
				seq_shift = _shift_dna_batch(dna_seq, shift)
				pred = forward(seq_shift, aux)
				preds.append(pred)

				seq_rc = dna_rc(seq_shift)
				pred_rc = forward(seq_rc, aux_rc)
				pred_rc = predictions_rc(pred_rc)
				preds.append(pred_rc)

	preds = [p.float() for p in preds]
	return torch.stack(preds, dim=0).mean(dim=0)

def export_bedgraph(lines: Iterable[Tuple[str, int, int, float]], path: Path, chrom_sizes: Path, delete_bedgraph: bool = False):
	lines = list(lines)
	if not lines:
		return
	lines.sort(key=lambda x: (x[0], x[1]))
	with open(path, "w") as handle:
		for chrom, start, end, val in lines:
			handle.write(f"{chrom}\t{start}\t{end}\t{val}\n")
	subprocess.run(["bedGraphToBigWig", str(path), str(chrom_sizes), str(path.with_suffix(".bw"))], check=True)
	if delete_bedgraph:
		path.unlink(missing_ok=True)

def parse_tissue_ids(path_or_list: str) -> List[int]:
	path = Path(path_or_list)
	if path.exists():
		with open(path) as handle:
			return [int(x) for x in handle.read().strip().split() if x.strip()]
	return [int(x) for x in path_or_list.split(",") if x.strip()]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-ckpt", type=str, help="Path to the model checkpoint.")
	parser.add_argument(
		"--model-type",
		type=str,
		choices=["corgi", "corgiplus", "corgiplus_impute", "corgiplus_rna", "corgiplus_dnase", "corgiplusplus_rna", "corgiplus_rna_nofilm"],
		help="Type of the model.",
	)
	parser.add_argument("--auto", action="store_true", help="Auto-detect all checkpoints in the input directory.")
	parser.add_argument("--input-dir", type=str, help="Directory containing checkpoints when using --auto.")
	parser.add_argument("--outdir", type=str, required=True, help="Output directory for bedgraph and bigwig files.")
	parser.add_argument("--tissues", type=str, default=config_corgiplus["test_tissues_path"])
	parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of non-overlapping sequences to test")
	parser.add_argument("--coords-bed", type=Path, default=Path(config_corgiplus["bed_path"]))
	parser.add_argument("--mask-path", type=Path, default=Path(config_corgiplus["mask_path"]))
	parser.add_argument("--tf-expression", type=Path, default=Path(config_corgiplus["trans_regulator_expression_path"]))
	parser.add_argument("--dnase-global", type=Path, default=Path(config_corgiplus["dnase_global_path"]))
	parser.add_argument("--mean-signal", type=Path, default=Path("/project/deeprna/data/revision/training_baseline_signal_qn.npy"))
	parser.add_argument("--dna-path", type=Path, default=Path(config_corgiplus["dna_path"]))
	parser.add_argument("--tissue-dir", type=Path, default=Path("/project/deeprna_data/revision_data_qn_parallel"))
	parser.add_argument("--export-bigwig", action="store_true", help="Export bedGraph/BigWig tracks for predictions and baselines.")
	parser.add_argument("--delete-bedgraph", action="store_true", help="Delete bedGraph files after converting to BigWig.")
	parser.add_argument(
		"--export-tracks",
		type=str,
		default=",".join(DEFAULT_EXPORT_TRACKS),
		help="Comma-separated list of track names to export to BigWig; use 'all' to export every available track.",
	)
	parser.add_argument("--chrom-sizes", type=Path, default=Path("/project/deeprna_data/corgi-reproduction/data/hg38.chrom.sizes"))
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
	device = torch.device("cuda")
	logging.info(f"Using device: {device}")

	if args.auto:
		if not args.input_dir:
			raise ValueError("--input-dir is required when using --auto")
		model_specs = []
		for ckpt in sorted(Path(args.input_dir).glob("*.pt")):
			model_type = infer_model_type(ckpt.stem)
			model_specs.append((ckpt, model_type))
		if not model_specs:
			raise RuntimeError(f"No checkpoints found in {args.input_dir}")
	else:
		if not args.model_ckpt or not args.model_type:
			raise ValueError("--model-ckpt and --model-type are required unless --auto is set")
		model_specs = [(Path(args.model_ckpt), args.model_type)]

	need_dnase = any(mt == 'corgiplus_dnase' for _, mt in model_specs)
	dnase_global = None
	if need_dnase:
		dnase_global = torch.from_numpy(np.load(args.dnase_global)).float().to(device)

	rng = np.random.default_rng(1)
	tissues = parse_tissue_ids(args.tissues)
	logging.info(f"Tissues: {tissues}")

	coords_df = pd.read_csv(args.coords_bed, sep="\t", header=None, names=["chr", "start", "end", "fold"])
	total_seqs = len(coords_df)
	base_indices = np.arange(0, total_seqs, 3)
	if args.fraction < 1.0:
		keep = max(1, int(len(base_indices) * args.fraction))
		base_indices = np.sort(rng.choice(base_indices, size=keep, replace=False))
	logging.info(f"Evaluating {len(base_indices)} sequences (every 3rd) out of {total_seqs}")
	coords_chr = coords_df["chr"].tolist()
	coords_start = coords_df["start"].tolist()

	with open('/project/deeprna/data/experiments_final.txt', 'r', encoding='utf-8') as f:
		experiments= f.read().strip().split()
	exp_name_to_channel_id = {exp: i for i, exp in enumerate(experiments)}
	export_tracks = None if args.export_tracks.lower() == 'all' else {t.strip().lower() for t in args.export_tracks.split(',') if t.strip()}

	experiment_mask = load_experiment_mask(args.mask_path)
	trans_reg_expression = torch.from_numpy(np.load(args.tf_expression)).float()

	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	mean_lines: Dict[str, List[Tuple[str, int, int, float]]] = {}
	mean_done: set[int] = set()
	mean_written = False
	
	rows = []
	
	regular_tracks = ['dnase','atac','h3k4me1','h3k4me2','h3k4me3','h3k9ac','h3k9me3','h3k27ac','h3k27me3','h3k36me3','h3k79me2','ctcf','rna_10x','wgbs']
	stranded_tracks = ['cage', 'rampage', 'rna_total', 'rna_polya']

	for ckpt_path, model_type in model_specs:
		model_name = Path(ckpt_path).stem
		logging.info(f"Loading model {model_name} ({model_type})")
		model = load_model(str(ckpt_path), model_type, device, dnase_global)

		all_spearman_deltas = []
		all_pearson_deltas = []
		all_pearson_delta_de = []
		all_spearman_raw = []
		all_pearson_raw = []
		all_spearman_baseline = []
		all_pearson_baseline = []
		for tissue_id in tissues:
			logging.info(f"Processing tissue {tissue_id}")
			dataset = CorgiPlusDataset(
				dna_sequences=str(args.dna_path),
				sequence_ids=base_indices.tolist(),
				tissue_dir=str(args.tissue_dir),
				tissue_ids=[tissue_id],
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
			sampler = CorgiSampler(sequence_ids=base_indices.tolist(), tissue_ids=[tissue_id], shuffled=False)
			loader = DataLoader(
				dataset,
				batch_size=2,
				sampler=sampler,
				num_workers=2,
				pin_memory=True,
				drop_last=False,
			)

			gt_lines: Dict[Tuple[int, str], List[Tuple[str, int, int, float]]] = {}
			gt_delta_lines: Dict[Tuple[int, str], List[Tuple[str, int, int, float]]] = {}
			pred_delta_lines: Dict[Tuple[str, int, str], List[Tuple[str, int, int, float]]] = {}

			pred_bins = []
			true_bins = []
			mean_baseline_bins = []
			for batch in loader:
				dna_seq, trans_reg, label, exp_mask, mean_baseline, tissue_id_tensor, seq_id = batch
				dna_seq = dna_seq.to(device)
				trans_reg = trans_reg.to(device)
				mean_baseline = mean_baseline.to(device)
				label = label.to(device)

				tissue_id_device = tissue_id_tensor.to(device)
				label_cpu = crop_center(label, TARGET_BINS).to(dtype=torch.float32, device='cpu')
				mean_baseline_cpu = crop_center(mean_baseline, TARGET_BINS).to(dtype=torch.float32, device='cpu')
				mask_cpu = exp_mask.squeeze(-1).numpy()

				if args.export_bigwig:
					seq_np = seq_id.numpy()
					region_start_np = np.array([coords_start[i] for i in seq_np], dtype=np.int64) + CROP
					bin_offsets = np.arange(TARGET_BINS, dtype=np.int64) * BINSIZE
					bin_starts_np = region_start_np[:, None] + bin_offsets[None, :]
					bin_ends_np = bin_starts_np + BINSIZE

				with torch.no_grad():
					aux = None
					cond = None
					trans_reg_for_pred: Optional[torch.Tensor] = trans_reg
					if model_type == 'corgi':
						pred = pred_tta(model, dna_seq, trans_reg_for_pred, model_type, aux, cond)
					elif model_type == 'big_corgi':
						pred = pred_tta(model, dna_seq, trans_reg_for_pred, model_type, aux, cond)
					elif model_type == 'corgiplus_dnase':
						aux = torch.cat([label[:, 0:1, :], mean_baseline], dim=1)
						cond = dnase_global[tissue_id_device]
						pred = pred_tta(model, dna_seq, trans_reg_for_pred, model_type, aux, cond)
					elif model_type == 'corgiplus_rna':
						aux = torch.cat([label[:, 16:18, :], mean_baseline], dim=1)
						pred = pred_tta(model, dna_seq, trans_reg_for_pred, model_type, aux, cond)
					elif model_type == 'corgiplus_rna_nofilm':
						aux = torch.cat([label[:, 16:18, :], mean_baseline], dim=1)
						trans_reg_for_pred = None
						pred = pred_tta(model, dna_seq, trans_reg_for_pred, model_type, aux, cond)
					elif model_type == 'corgiplusplus_rna':
						rna_tracks = label[:, 16:18, :]
						delta_rna = rna_tracks - mean_baseline[:, 16:18, :]
						aux = torch.cat([rna_tracks, mean_baseline, delta_rna], dim=1)
						pred = pred_tta(model, dna_seq, trans_reg_for_pred, model_type, aux, cond)
					else:
						aux = mean_baseline
						pred = pred_tta(model, dna_seq, trans_reg_for_pred, model_type, aux, cond)

				pred_cpu = pred.to(device='cpu', dtype=torch.float32)
				pred_delta_cpu = pred_cpu - mean_baseline_cpu
				label_delta_cpu = label_cpu - mean_baseline_cpu

				if args.export_bigwig:
					seq_np = seq_id.numpy()
					for i in range(label_cpu.shape[0]):
						t_id = int(tissue_id_tensor[i])
						active_channels = np.nonzero(mask_cpu[i])[0]
						if active_channels.size == 0:
							continue
						chrom = coords_chr[int(seq_np[i])]
						starts_np = bin_starts_np[i]
						ends_np = bin_ends_np[i]
						if not mean_written and seq_np[i] not in mean_done:
							for c in active_channels:
								exp_name = experiments[c]
								exp_key = exp_name.lower()
								if export_tracks is not None and exp_key not in export_tracks:
									continue
								mean_vals = mean_baseline_cpu[i, c].numpy().astype(float)
								mean_lines.setdefault(exp_name, []).extend(
									zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), mean_vals.tolist())
								)
							mean_done.add(int(seq_np[i]))
						for c in active_channels:
							exp_name = experiments[c]
							exp_key = exp_name.lower()
							if export_tracks is not None and exp_key not in export_tracks:
								continue
							gt_vals = label_cpu[i, c].numpy().astype(float)
							gt_delta_vals = label_delta_cpu[i, c].numpy().astype(float)
							pred_delta_vals = pred_delta_cpu[i, c].numpy().astype(float)
							gt_lines.setdefault((t_id, exp_name), []).extend(
								zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), gt_vals.tolist())
							)
							gt_delta_lines.setdefault((t_id, exp_name), []).extend(
								zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), gt_delta_vals.tolist())
							)
							pred_delta_lines.setdefault((model_name, t_id, exp_name), []).extend(
								zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), pred_delta_vals.tolist())
							)

				pred_bins.append(pred_cpu)
				true_bins.append(label_cpu)
				mean_baseline_bins.append(mean_baseline_cpu)

			# Shapes: (num_sequences, num_channels, num_bins)
			pred_bins = torch.cat(pred_bins, dim=0)
			true_bins = torch.cat(true_bins, dim=0)
			mean_baseline_bins = torch.cat(mean_baseline_bins, dim=0)

			# Concatenate all sequences along the bin axis
			pred_bins = pred_bins.permute(1, 0, 2).reshape(pred_bins.shape[1], -1)
			true_bins = true_bins.permute(1, 0, 2).reshape(true_bins.shape[1], -1)
			mean_baseline_bins = mean_baseline_bins.permute(1, 0, 2).reshape(mean_baseline_bins.shape[1], -1)

			# Calculate deltas
			pred_delta = pred_bins - mean_baseline_bins
			label_delta = true_bins - mean_baseline_bins

			# Channel-wise metrics
			spearman_deltas = []
			pearson_deltas = []
			pearson_delta_de = []
			spearman_raw = []
			pearson_raw = []
			spearman_baseline = []
			pearson_baseline = []
			for i in range(label_delta.shape[0]):
				# If channel ground truth not available, skip
				if exp_mask[0, i] == 0:
					spearman_deltas.append(np.nan)
					pearson_deltas.append(np.nan)
					pearson_delta_de.append(np.nan)
					spearman_raw.append(np.nan)
					pearson_raw.append(np.nan)
					spearman_baseline.append(np.nan)
					pearson_baseline.append(np.nan)
					continue

				# Spearman delta
				spearman_corr, _ = spearmanr(pred_delta[i, :].flatten().numpy(), label_delta[i, :].flatten().numpy())
				spearman_deltas.append(spearman_corr)

				# Pearson delta
				pearson_corr, _ = pearsonr(pred_delta[i, :].flatten().numpy(), label_delta[i, :].flatten().numpy())
				pearson_deltas.append(pearson_corr)

				# Pearson delta over differentially expressed bins (abs label delta > 0.1)
				mask = np.abs(label_delta[i, :].flatten().numpy()) > 0.1
				if mask.sum() > 1:  # Need at least 2 points to compute Pearson correlation
					pearson_corr_de, _ = pearsonr(pred_delta[i, :].flatten().numpy()[mask], label_delta[i, :].flatten().numpy()[mask])
				else:
					pearson_corr_de = np.nan  # Not enough data to compute correlation
				pearson_delta_de.append(pearson_corr_de)

				# Raw cors
				spearman_corr_non_delta, _ = spearmanr(pred_bins[i, :].flatten().numpy(), true_bins[i, :].flatten().numpy())
				pearson_corr_non_delta, _ = pearsonr(pred_bins[i, :].flatten().numpy(), true_bins[i, :].flatten().numpy())
				spearman_raw.append(spearman_corr_non_delta)
				pearson_raw.append(pearson_corr_non_delta)
				spearman_corr_non_delta_baseline, _ = spearmanr(mean_baseline_bins[i, :].flatten().numpy(), true_bins[i, :].flatten().numpy())
				pearson_corr_non_delta_baseline, _ = pearsonr(mean_baseline_bins[i, :].flatten().numpy(), true_bins[i, :].flatten().numpy())
				spearman_baseline.append(spearman_corr_non_delta_baseline)
				pearson_baseline.append(pearson_corr_non_delta_baseline)

			spearman_deltas = np.array(spearman_deltas)
			pearson_deltas = np.array(pearson_deltas)
			pearson_delta_de = np.array(pearson_delta_de)
			spearman_raw = np.array(spearman_raw)
			pearson_raw = np.array(pearson_raw)
			spearman_baseline = np.array(spearman_baseline)
			pearson_baseline = np.array(pearson_baseline)

			all_spearman_deltas.append(spearman_deltas)
			all_pearson_deltas.append(pearson_deltas)
			all_pearson_delta_de.append(pearson_delta_de)
			all_spearman_raw.append(spearman_raw)
			all_pearson_raw.append(pearson_raw)
			all_spearman_baseline.append(spearman_baseline)
			all_pearson_baseline.append(pearson_baseline)

			if args.export_bigwig:
				#for (t_id, exp_name), lines in gt_lines.items():
				#	export_bedgraph(lines, outdir / f"gt_t{t_id}_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
				for (t_id, exp_name), lines in gt_delta_lines.items():
					export_bedgraph(lines, outdir / f"gt_delta_t{t_id}_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
				for (m_name, t_id, exp_name), lines in pred_delta_lines.items():
					export_bedgraph(lines, outdir / f"pred_delta_{m_name}_t{t_id}_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
				gt_lines.clear()
				gt_delta_lines.clear()
				pred_delta_lines.clear()

		# Aggregate results across tissues
		all_spearman_deltas = np.array(all_spearman_deltas)
		all_pearson_deltas = np.array(all_pearson_deltas)
		all_pearson_delta_de = np.array(all_pearson_delta_de)
		all_spearman_raw = np.array(all_spearman_raw)
		all_pearson_raw = np.array(all_pearson_raw)
		all_spearman_baseline = np.array(all_spearman_baseline)
		all_pearson_baseline = np.array(all_pearson_baseline)

		for t_idx, tissue_id in enumerate(tissues):
			logging.info(f"Recording results for tissue {tissue_id}")
			for channel_idx, channel_name in enumerate(experiments):
				available = bool(experiment_mask[tissue_id][channel_idx])
				if not available:
					continue

				rows.append({
					'tissue': tissue_id,
					'model': model_name,
					'channel': channel_name,
					'pearson_cor': float(all_pearson_raw[t_idx, channel_idx]) if not np.isnan(all_pearson_raw[t_idx, channel_idx]) else np.nan,
					'spearman_cor': float(all_spearman_raw[t_idx, channel_idx]) if not np.isnan(all_spearman_raw[t_idx, channel_idx]) else np.nan,
					'pearson_delta': float(all_pearson_deltas[t_idx, channel_idx]) if not np.isnan(all_pearson_deltas[t_idx, channel_idx]) else np.nan,
					'pearson_delta_de': float(all_pearson_delta_de[t_idx, channel_idx]) if not np.isnan(all_pearson_delta_de[t_idx, channel_idx]) else np.nan,
					'spearman_delta': float(all_spearman_deltas[t_idx, channel_idx]) if not np.isnan(all_spearman_deltas[t_idx, channel_idx]) else np.nan,
				})

		if args.export_bigwig and not mean_written and mean_lines:
			for exp_name, lines in mean_lines.items():
				export_bedgraph(lines, outdir / f"mean_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
			mean_written = True
			mean_lines.clear()

	for t_idx, tissue_id in enumerate(tissues):
			logging.info(f"Recording results for tissue {tissue_id}")
			for channel_idx, channel_name in enumerate(experiments):
				available = bool(experiment_mask[tissue_id][channel_idx])
				if not available:
					continue
				
				rows.append({
					'tissue': tissue_id,
					'model': 'baseline',
					'channel': channel_name,
					'pearson_cor': float(all_pearson_baseline[t_idx, channel_idx]) if not np.isnan(all_pearson_baseline[t_idx, channel_idx]) else np.nan,
					'spearman_cor': float(all_spearman_baseline[t_idx, channel_idx]) if not np.isnan(all_spearman_baseline[t_idx, channel_idx]) else np.nan,
					'pearson_delta': np.nan,
					'pearson_delta_de': np.nan,
					'spearman_delta': np.nan,
				})

	if rows:
		df = pd.DataFrame(rows)
		out_csv = outdir / "metrics.csv"
		df.to_csv(out_csv, index=False, float_format="%.4f")
		logging.info(f"Saved results to {out_csv}")

if __name__ == "__main__":
	main()
