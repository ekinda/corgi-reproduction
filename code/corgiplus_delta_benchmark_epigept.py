import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

# Make epigept package importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPIGEPT_ROOT = PROJECT_ROOT.parent / "epigept_directory"
if str(EPIGEPT_ROOT) not in sys.path:
	sys.path.append(str(EPIGEPT_ROOT))

from corgi.config_corgiplus import config_corgiplus
from corgi.data_classes import CorgiSampler
from corgi.model import CorgiPlus
from corgi.trainer_corgiplus import CorgiPlusDataset
from corgi.utils import load_experiment_mask
from epigept.model import EpiGePT as epigept_module
from epigept.model import config as epigept_config

from benchmark_utils import crop_center

SEQ_LENGTH = 524_288
STRIDE = 393_216
CROP = 65536
BINSIZE = 64
TARGET_BINS = 6144

DEFAULT_EXPORT_TRACKS = [
	"h3k4me1",
	"h3k4me2",
	"h3k4me3",
	"h3k9ac",
	"h3k9me3",
	"h3k27ac",
	"h3k27me3",
	"h3k36me3",
	"h3k79me2",
]

# Reshaping helpers for 128 bp resolution
CORGIPLUS_TRIM_BINS = 72  # 72 * 64 = 4608 bp trimmed on each side
PROCESSED_BINS = 3000
PROCESSED_BIN_SIZE = 128
EPIGEPT_CHUNKS = 3
EPIGEPT_CHUNK_LEN = 128_000
EPIGEPT_TF_TRIM = 548  # 548 * 128 = 70,144 bp; matches DNA crop
EPIGEPT_BIN_LEN = 1000
EPIGEPT_CROP_START = CROP + CORGIPLUS_TRIM_BINS * BINSIZE  # 70,144 bp offset into the 524,288 bp window

def load_model(checkpoint_path: str, device: torch.device) -> CorgiPlus:
	"""Load a corgiplusplus_rna model only."""
	cfg = config_corgiplus.copy()
	cfg['corgiplus_aux_input_dim'] = cfg['output_channels'] + 4
	model = CorgiPlus(cfg).to(device)
	state = torch.load(checkpoint_path, map_location=device)
	state_dict = state.get('model_state_dict', state)
	model.load_state_dict(state_dict)
	model.eval()
	return model


def load_epigept_model(checkpoint_path: Path, device: torch.device, batch_size: int) -> epigept_module.EpiGePT:
	model = epigept_module.EpiGePT(
		epigept_config.WORD_NUM,
		epigept_config.SEQUENCE_DIM,
		epigept_config.TF_DIM,
		batch_size,
	)
	state = torch.load(checkpoint_path, map_location=device)
	state_dict = state.get('state_dict', state)
	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	if missing:
		logging.warning(f"Missing epigept parameters: {missing}")
	if unexpected:
		logging.warning(f"Unexpected epigept parameters: {unexpected}")
	model = model.to(device)
	model.eval()
	return model


def downsample_to_128bp(track: torch.Tensor) -> torch.Tensor:
	"""Crop 72 bins each side (64 bp bins) then mean-pool by 2 -> 3000 bins of 128 bp."""
	if track.shape[-1] < (CORGIPLUS_TRIM_BINS * 2):
		raise ValueError(f"Track length {track.shape[-1]} is too short for trimming")
	cropped = track[..., CORGIPLUS_TRIM_BINS:-CORGIPLUS_TRIM_BINS]
	if cropped.shape[-1] != PROCESSED_BINS * 2:
		raise ValueError(f"Expected cropped length {PROCESSED_BINS * 2}, got {cropped.shape[-1]}")
	# cropped last dim = 6000, reshape to (3000, 2) and mean over the last axis
	reshaped = cropped.reshape(*cropped.shape[:-1], PROCESSED_BINS, 2)
	return reshaped.mean(dim=-1)


def prepare_epigept_inputs(
	dna_seq: torch.Tensor,
	tf_binding_batch: np.ndarray,
	tf_expression: torch.Tensor,
	tissue_ids: torch.Tensor,
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Create epigept DNA and TF feature tensors for 3x128 kb chunks.

	Args:
		dna_seq: (B, 524288, 4) float tensor on device
		tf_binding_batch: numpy array (B, 4096, tf_dim)
		tf_expression: tensor (num_tissues, tf_dim) on device
		tissue_ids: tensor (B,) of tissue indices
	Return:
		dna_chunks: (B*3, 4, 128000)
		tf_feats: (B*3, 1000, tf_dim)
	"""
	# DNA: center crop 70,144 bp on each side -> 384,000 bp, then 3 x 128,000 chunks
	dna_center = dna_seq[:, EPIGEPT_CROP_START:SEQ_LENGTH - EPIGEPT_CROP_START, :]  # (B, 384000, 4)
	dna_chunks = dna_center.contiguous().reshape(dna_center.shape[0] * EPIGEPT_CHUNKS, EPIGEPT_CHUNK_LEN, 4).permute(0, 2, 1)

	# TF binding: crop 548 bins each side (128 bp bins) -> 3000 bins, then 3 x 1000 chunks
	tf_binding_tensor = torch.from_numpy(tf_binding_batch).to(device=device, dtype=torch.float32)
	tf_cropped = tf_binding_tensor[:, EPIGEPT_TF_TRIM:-EPIGEPT_TF_TRIM, :]
	tf_chunks = tf_cropped.contiguous().reshape(tf_cropped.shape[0] * EPIGEPT_CHUNKS, EPIGEPT_BIN_LEN, tf_cropped.shape[-1])

	# Multiply by expression per tissue, repeat per chunk
	tf_expr = tf_expression[tissue_ids].to(device)
	tf_expr_repeated = tf_expr.repeat_interleave(EPIGEPT_CHUNKS, dim=0).unsqueeze(1)
	tf_feats = tf_chunks * tf_expr_repeated
	return dna_chunks, tf_feats


def compute_channel_metrics(
	pred_bins: torch.Tensor,
	true_bins: torch.Tensor,
	mean_bins: torch.Tensor,
	exp_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
	"""Compute channel-wise correlations for flattened bins (num_seq x bins)."""
	pred_delta = pred_bins - mean_bins
	true_delta = true_bins - mean_bins

	pred_flat = pred_bins.permute(1, 0, 2).reshape(pred_bins.shape[1], -1).numpy()
	true_flat = true_bins.permute(1, 0, 2).reshape(true_bins.shape[1], -1).numpy()
	mean_flat = mean_bins.permute(1, 0, 2).reshape(mean_bins.shape[1], -1).numpy()
	pred_delta_flat = pred_delta.permute(1, 0, 2).reshape(pred_delta.shape[1], -1).numpy()
	true_delta_flat = true_delta.permute(1, 0, 2).reshape(true_delta.shape[1], -1).numpy()

	metrics = {
		'spearman_delta': [],
		'pearson_delta': [],
		'pearson_delta_de': [],
		'spearman_raw': [],
		'pearson_raw': [],
		'spearman_baseline': [],
		'pearson_baseline': [],
	}

	for i in range(true_flat.shape[0]):
		if not exp_mask[i]:
			for key in metrics:
				metrics[key].append(np.nan)
			continue

		sp_delta, _ = spearmanr(pred_delta_flat[i], true_delta_flat[i])
		pe_delta, _ = pearsonr(pred_delta_flat[i], true_delta_flat[i])
		de_mask = np.abs(true_delta_flat[i]) > 0.1
		if de_mask.sum() > 1:
			pe_delta_de, _ = pearsonr(pred_delta_flat[i][de_mask], true_delta_flat[i][de_mask])
		else:
			pe_delta_de = np.nan
		sp_raw, _ = spearmanr(pred_flat[i], true_flat[i])
		pe_raw, _ = pearsonr(pred_flat[i], true_flat[i])
		sp_base, _ = spearmanr(mean_flat[i], true_flat[i])
		pe_base, _ = pearsonr(mean_flat[i], true_flat[i])

		metrics['spearman_delta'].append(sp_delta)
		metrics['pearson_delta'].append(pe_delta)
		metrics['pearson_delta_de'].append(pe_delta_de)
		metrics['spearman_raw'].append(sp_raw)
		metrics['pearson_raw'].append(pe_raw)
		metrics['spearman_baseline'].append(sp_base)
		metrics['pearson_baseline'].append(pe_base)

	return {k: np.array(v) for k, v in metrics.items()}


def compute_pairwise_pearson(a_bins: torch.Tensor, b_bins: torch.Tensor, exp_mask: np.ndarray) -> np.ndarray:
	"""Pearson correlations between two prediction tensors (B, C, L)."""
	a_flat = a_bins.permute(1, 0, 2).reshape(a_bins.shape[1], -1).numpy()
	b_flat = b_bins.permute(1, 0, 2).reshape(b_bins.shape[1], -1).numpy()
	pearsons = []
	for i in range(a_flat.shape[0]):
		if not exp_mask[i]:
			pearsons.append(np.nan)
			continue
		val, _ = pearsonr(a_flat[i], b_flat[i])
		pearsons.append(val)
	return np.array(pearsons)


def compute_pairwise_pearson_delta(
	a_bins: torch.Tensor,
	b_bins: torch.Tensor,
	mean_bins: torch.Tensor,
	exp_mask: np.ndarray,
) -> np.ndarray:
	"""Pearson correlations between delta predictions (B, C, L)."""
	a_delta = a_bins - mean_bins
	b_delta = b_bins - mean_bins
	return compute_pairwise_pearson(a_delta, b_delta, exp_mask)


def correlation_sample_sizes(true_bins: torch.Tensor, mean_bins: torch.Tensor, exp_mask: np.ndarray) -> Tuple[int, np.ndarray]:
	"""Return total sample size (raw/delta) and DE sample sizes per channel."""
	batch, _, bins = true_bins.shape
	raw_size = batch * bins
	true_delta = true_bins - mean_bins
	true_delta_flat = true_delta.permute(1, 0, 2).reshape(true_delta.shape[1], -1).numpy()
	de_sizes = np.array([
		(np.abs(true_delta_flat[i]) > 0.1).sum() if exp_mask[i] else 0
		for i in range(true_delta_flat.shape[0])
	], dtype=np.int64)
	return raw_size, de_sizes

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
	parser.add_argument("--corgi-ckpt", type=Path, required=True, help="Path to corgiplusplus_rna checkpoint (.pt)")
	parser.add_argument("--epigept-ckpt", type=Path, required=True, help="Path to EpiGePT lightning checkpoint (.ckpt)")
	parser.add_argument("--outdir", type=str, required=True, help="Output directory for bedgraph and bigwig files.")
	parser.add_argument("--tissues", type=str, default=config_corgiplus["test_tissues_path"])
	parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of non-overlapping sequences to test")
	parser.add_argument("--coords-bed", type=Path, default=Path(config_corgiplus["bed_path"]))
	parser.add_argument("--mask-path", type=Path, default=Path(config_corgiplus["mask_path"]))
	parser.add_argument("--tf-expression", type=Path, default=Path(config_corgiplus["trans_regulator_expression_path"]))
	parser.add_argument("--mean-signal", type=Path, default=Path("/project/deeprna/data/revision/training_baseline_signal_qn.npy"))
	parser.add_argument("--dna-path", type=Path, default=Path(config_corgiplus["dna_path"]))
	parser.add_argument("--tissue-dir", type=Path, default=Path("/project/deeprna_data/revision_data_qn_parallel"))
	parser.add_argument("--export-bigwig", action="store_true", help="Export bedGraph/BigWig tracks for predictions and baselines.")
	parser.add_argument("--delete-bedgraph", action="store_true", help="Delete bedGraph files after conversion to BigWig.")
	parser.add_argument("--export-tracks", type=str, default=",".join(DEFAULT_EXPORT_TRACKS), help="Comma-separated track names to export; use 'all' to export every available track.")
	parser.add_argument("--chrom-sizes", type=Path, default=Path("/project/deeprna_data/corgi-reproduction/data/hg38.chrom.sizes"))
	parser.add_argument("--epigept-tf-binding", type=Path, default=Path("/project/deeprna_data/epigept/motifdata/tfbs_matrix.npy"), help="Path to TF binding affinity npy (n_seq x 4096 x 711).")
	parser.add_argument("--epigept-tf-expression", type=Path, default=Path("/project/deeprna_data/epigept/motifdata/tf_expression_epigept.npy"), help="Path to TF expression npy (n_tissue x 711).")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
	device = torch.device("cuda")
	logging.info(f"Using device: {device}")

	export_tracks = None if args.export_tracks.lower() == 'all' else {t.strip().lower() for t in args.export_tracks.split(',') if t.strip()}

	logging.info(f"Loading EpiGePT from {args.epigept_ckpt}")
	epigept_model = load_epigept_model(args.epigept_ckpt, device, batch_size=2)
	logging.info(f"Loading TF binding memmap from {args.epigept_tf_binding}")
	epigept_tf_binding = np.load(args.epigept_tf_binding, mmap_mode='r')
	if epigept_tf_binding.shape[1] < (EPIGEPT_TF_TRIM * 2):
		raise ValueError("TF binding array has insufficient length for 128 bp trimming")
	if epigept_tf_binding.shape[-1] != epigept_config.TF_DIM:
		raise ValueError(f"TF binding last dimension must be {epigept_config.TF_DIM}, got {epigept_tf_binding.shape[-1]}")
	logging.info(f"Loading TF expression from {args.epigept_tf_expression}")
	epigept_tf_expression = torch.from_numpy(np.load(args.epigept_tf_expression)).float().to(device)
	if epigept_tf_expression.shape[1] != epigept_config.TF_DIM:
		raise ValueError(f"TF expression second dimension must be {epigept_config.TF_DIM}, got {epigept_tf_expression.shape[1]}")

	model_ckpt = Path(args.corgi_ckpt)

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

	experiment_mask = load_experiment_mask(args.mask_path)
	trans_reg_expression = torch.from_numpy(np.load(args.tf_expression)).float()

	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	mean_lines: Dict[str, List[Tuple[str, int, int, float]]] = {}
	mean_done: set[int] = set()
	mean_written = False

	rows = []

	baseline_written: set[int] = set()
	epigept_written: set[int] = set()

	model_name = model_ckpt.stem
	logging.info(f"Loading model {model_name} (corgiplusplus_rna only)")
	model = load_model(str(model_ckpt), device)

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
		epigept_bins = []

		exp_mask_vec = np.asarray(experiment_mask[tissue_id]).squeeze().astype(bool)

		for batch in loader:
			dna_seq, trans_reg, label, exp_mask, mean_baseline, tissue_id_tensor, seq_id = batch
			dna_seq = dna_seq.to(device)
			trans_reg = trans_reg.to(device)
			mean_baseline = mean_baseline.to(device)
			label = label.to(device)

			label_cpu = crop_center(label, TARGET_BINS).to(dtype=torch.float32, device='cpu')
			mean_baseline_cpu = crop_center(mean_baseline, TARGET_BINS).to(dtype=torch.float32, device='cpu')

			with torch.no_grad():
				with torch.autocast('cuda', dtype=torch.bfloat16):
					rna_tracks = label[:, 16:18, :]
					delta_rna = rna_tracks - mean_baseline[:, 16:18, :]
					aux = torch.cat([rna_tracks, mean_baseline, delta_rna], dim=1)
					pred = model(dna_seq, aux.permute(0, 2, 1), trans_reg)

			pred_cpu = pred.to(device='cpu', dtype=torch.float32)
			pred_proc = downsample_to_128bp(pred_cpu)
			label_proc = downsample_to_128bp(label_cpu)
			mean_proc = downsample_to_128bp(mean_baseline_cpu)

			# Epigept forward
			seq_np = seq_id.numpy()
			tf_binding_batch = epigept_tf_binding[seq_np]
			dna_chunks, tf_feats = prepare_epigept_inputs(dna_seq, tf_binding_batch, epigept_tf_expression, tissue_id_tensor, device)
			with torch.no_grad():
				with torch.autocast('cuda', dtype=torch.bfloat16):
					epi_pred = epigept_model(dna_chunks, tf_feats)
			# epi_pred: (B*3, 1000, C)
			epi_pred = epi_pred.to(device='cpu', dtype=torch.float32)
			epi_pred = epi_pred.view(label_proc.shape[0], EPIGEPT_CHUNKS, EPIGEPT_BIN_LEN, epigept_config.NUM_SIGNALS)
			epi_pred = epi_pred.permute(0, 3, 1, 2).reshape(label_proc.shape[0], epigept_config.NUM_SIGNALS, PROCESSED_BINS)
			if epi_pred.shape[1] != label_proc.shape[1]:
				raise ValueError(f"Epigept outputs {epi_pred.shape[1]} channels but labels have {label_proc.shape[1]}")
			epigept_bins.append(epi_pred)

			if args.export_bigwig:
				seq_np = seq_id.numpy()
				region_start_np = np.array([coords_start[i] for i in seq_np], dtype=np.int64) + EPIGEPT_CROP_START
				bin_offsets = np.arange(PROCESSED_BINS, dtype=np.int64) * PROCESSED_BIN_SIZE
				bin_starts_np = region_start_np[:, None] + bin_offsets[None, :]
				bin_ends_np = bin_starts_np + PROCESSED_BIN_SIZE
				mask_cpu = exp_mask.squeeze(-1).numpy()
				for i in range(label_proc.shape[0]):
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
							mean_vals = mean_proc[i, c].numpy().astype(float)
							mean_lines.setdefault(exp_name, []).extend(
								zip((chrom,) * PROCESSED_BINS, starts_np.tolist(), ends_np.tolist(), mean_vals.tolist())
							)
						mean_done.add(int(seq_np[i]))
					for c in active_channels:
						exp_name = experiments[c]
						exp_key = exp_name.lower()
						if export_tracks is not None and exp_key not in export_tracks:
							continue
						gt_vals = label_proc[i, c].numpy().astype(float)
						gt_delta_vals = (label_proc[i, c] - mean_proc[i, c]).numpy().astype(float)
						pred_delta_vals = (pred_proc[i, c] - mean_proc[i, c]).numpy().astype(float)
						gt_lines.setdefault((t_id, exp_name), []).extend(
							zip((chrom,) * PROCESSED_BINS, starts_np.tolist(), ends_np.tolist(), gt_vals.tolist())
						)
						gt_delta_lines.setdefault((t_id, exp_name), []).extend(
							zip((chrom,) * PROCESSED_BINS, starts_np.tolist(), ends_np.tolist(), gt_delta_vals.tolist())
						)
						pred_delta_lines.setdefault((model_name, t_id, exp_name), []).extend(
							zip((chrom,) * PROCESSED_BINS, starts_np.tolist(), ends_np.tolist(), pred_delta_vals.tolist())
						)
						epi_delta_vals = (epigept_bins[-1][i, c] - mean_proc[i, c]).numpy().astype(float)
						pred_delta_lines.setdefault(("epigept", t_id, exp_name), []).extend(
							zip((chrom,) * PROCESSED_BINS, starts_np.tolist(), ends_np.tolist(), epi_delta_vals.tolist())
						)

			pred_bins.append(pred_proc)
			true_bins.append(label_proc)
			mean_baseline_bins.append(mean_proc)

		# Shapes: (num_sequences, num_channels, 3000)
		pred_bins = torch.cat(pred_bins, dim=0)
		true_bins = torch.cat(true_bins, dim=0)
		mean_baseline_bins = torch.cat(mean_baseline_bins, dim=0)
		epigept_bins_tensor = torch.cat(epigept_bins, dim=0) if epigept_bins is not None else None

		metrics_corgi = compute_channel_metrics(pred_bins, true_bins, mean_baseline_bins, exp_mask_vec)
		metrics_epigept = compute_channel_metrics(epigept_bins_tensor, true_bins, mean_baseline_bins, exp_mask_vec)
		pairwise_raw = compute_pairwise_pearson(pred_bins, epigept_bins_tensor, exp_mask_vec)
		pairwise_delta = compute_pairwise_pearson_delta(pred_bins, epigept_bins_tensor, mean_baseline_bins, exp_mask_vec)
		raw_size, de_sizes = correlation_sample_sizes(true_bins, mean_baseline_bins, exp_mask_vec)
		available_de_sizes = de_sizes[exp_mask_vec]
		if available_de_sizes.size:
			logging.info(
				f"Tissue {tissue_id}: correlation sample size per channel = {raw_size} "
				f"({pred_bins.shape[0]} sequences x {pred_bins.shape[2]} bins); "
				f"DE sample sizes (|delta|>0.1) min/median/max: "
				f"{available_de_sizes.min()} / {int(np.median(available_de_sizes))} / {available_de_sizes.max()}"
			)
		else:
			logging.info(
				f"Tissue {tissue_id}: correlation sample size per channel = {raw_size} "
				f"({pred_bins.shape[0]} sequences x {pred_bins.shape[2]} bins); no available channels after mask"
			)

		if args.export_bigwig:
			for (t_id, exp_name), lines in gt_lines.items():
				export_bedgraph(lines, outdir / f"gt_t{t_id}_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
			for (t_id, exp_name), lines in gt_delta_lines.items():
				export_bedgraph(lines, outdir / f"gt_delta_t{t_id}_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
			for (m_name, t_id, exp_name), lines in pred_delta_lines.items():
				export_bedgraph(lines, outdir / f"pred_delta_{m_name}_t{t_id}_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
			gt_lines.clear()
			gt_delta_lines.clear()
			pred_delta_lines.clear()

		for channel_idx, channel_name in enumerate(experiments):
			available = bool(exp_mask_vec[channel_idx])
			if not available:
				continue
			rows.append({
				'tissue': tissue_id,
				'model': model_name,
				'channel': channel_name,
				'pearson_cor': float(metrics_corgi['pearson_raw'][channel_idx]) if not np.isnan(metrics_corgi['pearson_raw'][channel_idx]) else np.nan,
				'spearman_cor': float(metrics_corgi['spearman_raw'][channel_idx]) if not np.isnan(metrics_corgi['spearman_raw'][channel_idx]) else np.nan,
				'pearson_delta': float(metrics_corgi['pearson_delta'][channel_idx]) if not np.isnan(metrics_corgi['pearson_delta'][channel_idx]) else np.nan,
				'pearson_delta_de': float(metrics_corgi['pearson_delta_de'][channel_idx]) if not np.isnan(metrics_corgi['pearson_delta_de'][channel_idx]) else np.nan,
				'spearman_delta': float(metrics_corgi['spearman_delta'][channel_idx]) if not np.isnan(metrics_corgi['spearman_delta'][channel_idx]) else np.nan,
				'pearson_epigept_corgi': float(pairwise_raw[channel_idx]) if not np.isnan(pairwise_raw[channel_idx]) else np.nan,
				'pearson_epigept_corgi_delta': float(pairwise_delta[channel_idx]) if not np.isnan(pairwise_delta[channel_idx]) else np.nan,
			})

		if tissue_id not in epigept_written:
			for channel_idx, channel_name in enumerate(experiments):
				available = bool(exp_mask_vec[channel_idx])
				if not available:
					continue
				rows.append({
					'tissue': tissue_id,
					'model': 'epigept',
					'channel': channel_name,
					'pearson_cor': float(metrics_epigept['pearson_raw'][channel_idx]) if not np.isnan(metrics_epigept['pearson_raw'][channel_idx]) else np.nan,
					'spearman_cor': float(metrics_epigept['spearman_raw'][channel_idx]) if not np.isnan(metrics_epigept['spearman_raw'][channel_idx]) else np.nan,
					'pearson_delta': float(metrics_epigept['pearson_delta'][channel_idx]) if not np.isnan(metrics_epigept['pearson_delta'][channel_idx]) else np.nan,
					'pearson_delta_de': float(metrics_epigept['pearson_delta_de'][channel_idx]) if not np.isnan(metrics_epigept['pearson_delta_de'][channel_idx]) else np.nan,
					'spearman_delta': float(metrics_epigept['spearman_delta'][channel_idx]) if not np.isnan(metrics_epigept['spearman_delta'][channel_idx]) else np.nan,
					'pearson_epigept_corgi': np.nan,
					'pearson_epigept_corgi_delta': np.nan,
				})
			epigept_written.add(tissue_id)

		if tissue_id not in baseline_written:
			for channel_idx, channel_name in enumerate(experiments):
				available = bool(exp_mask_vec[channel_idx])
				if not available:
					continue
				rows.append({
					'tissue': tissue_id,
					'model': 'baseline',
					'channel': channel_name,
					'pearson_cor': float(metrics_corgi['pearson_baseline'][channel_idx]) if not np.isnan(metrics_corgi['pearson_baseline'][channel_idx]) else np.nan,
					'spearman_cor': float(metrics_corgi['spearman_baseline'][channel_idx]) if not np.isnan(metrics_corgi['spearman_baseline'][channel_idx]) else np.nan,
					'pearson_delta': np.nan,
					'pearson_delta_de': np.nan,
					'spearman_delta': np.nan,
					'pearson_epigept_corgi': np.nan,
					'pearson_epigept_corgi_delta': np.nan,
				})
			baseline_written.add(tissue_id)

		if args.export_bigwig and not mean_written and mean_lines:
			for exp_name, lines in mean_lines.items():
				export_bedgraph(lines, outdir / f"mean_{exp_name}.bedgraph", args.chrom_sizes, args.delete_bedgraph)
			mean_written = True
			mean_lines.clear()

	if rows:
		df = pd.DataFrame(rows)
		out_csv = outdir / "metrics.csv"
		df.to_csv(out_csv, index=False, float_format="%.4f")
		logging.info(f"Saved results to {out_csv}")

if __name__ == "__main__":
	main()
