import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from corgi.config_corgiplus import config_corgiplus
from corgi.trainer_corgiplus import CorgiPlusDataset
from corgi.data_classes import CorgiSampler
from corgi.utils import load_experiment_mask

from benchmark_utils import crop_center

SEQ_LENGTH = 524_288
STRIDE = 393_216
CROP = 65_536
BINSIZE = 64
TARGET_BINS = 6_144


def export_bedgraph(lines: Iterable[Tuple[str, int, int, float]], path: Path, chrom_sizes: Path) -> None:
    lines = list(lines)
    if not lines:
        return
    lines.sort(key=lambda x: (x[0], x[1]))
    with open(path, "w") as handle:
        for chrom, start, end, val in lines:
            handle.write(f"{chrom}\t{start}\t{end}\t{val}\n")
    Path(path).with_suffix(".bw").unlink(missing_ok=True)
    import subprocess

    subprocess.run([
        "bedGraphToBigWig",
        str(path),
        str(chrom_sizes),
        str(path.with_suffix(".bw")),
    ], check=True)


def choose_tissues(
    num_tissues: int,
    seed: int,
    experiment_mask: Dict[int, np.ndarray],
    original_dir: Path,
    qn_dir: Path,
) -> List[int]:
    available: List[int] = []
    for t_id in experiment_mask.keys():
        orig_file = original_dir / f"tissue_{t_id}.npy"
        qn_file = qn_dir / f"tissue_{t_id}.npy"
        if orig_file.exists() and qn_file.exists():
            available.append(t_id)
    if len(available) < num_tissues:
        raise ValueError(
            f"Only found {len(available)} tissues with data in both datasets; cannot sample {num_tissues}."
        )
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(available, size=num_tissues, replace=False).tolist())


def load_experiments(experiments_path: Path) -> List[str]:
    with open(experiments_path, "r", encoding="utf-8") as handle:
        return [x for x in handle.read().strip().split() if x]


def collect_lines(
    label_dir: Path,
    tissue_ids: List[int],
    experiment_mask: Dict[int, np.ndarray],
    trans_reg_expression: torch.Tensor,
    mean_baseline_file: Path,
    coords_chr: List[str],
    coords_start: List[int],
    chrom_sizes: Path,
    outdir: Path,
    experiments: List[str],
    tag: str,
    sequence_ids: np.ndarray,
    batch_size: int,
) -> None:
    dataset = CorgiPlusDataset(
        dna_sequences=config_corgiplus["dna_path"],
        sequence_ids=sequence_ids.tolist(),
        tissue_dir=str(label_dir),
        tissue_ids=tissue_ids,
        experiment_mask=experiment_mask,
        trans_reg_expression=trans_reg_expression,
        output_channels=config_corgiplus["output_channels"],
        augment_dna=False,
        augment_gnomad=False,
        augment_trans_reg_std=0.0,
        gnomad_pickle=None,
        trans_reg_clip=None,
        return_mean_baseline=True,
        mean_baseline_file=str(mean_baseline_file),
    )
    sampler = CorgiSampler(sequence_ids=sequence_ids.tolist(), tissue_ids=tissue_ids, shuffled=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )

    gt_lines: Dict[Tuple[int, str], List[Tuple[str, int, int, float]]] = {}
    mean_lines: Dict[str, List[Tuple[str, int, int, float]]] = {}
    mean_done: set[int] = set()

    for batch in loader:
        dna_seq, trans_reg, label, exp_mask, mean_baseline, tissue_id_tensor, seq_id = batch
        label_cpu = crop_center(label, TARGET_BINS).to(dtype=torch.float32, device="cpu")
        mean_cpu = crop_center(mean_baseline, TARGET_BINS).to(dtype=torch.float32, device="cpu")
        mask_cpu = exp_mask.squeeze(-1).numpy()

        seq_np = seq_id.numpy()
        region_start_np = np.array([coords_start[i] for i in seq_np], dtype=np.int64) + CROP
        bin_offsets = np.arange(TARGET_BINS, dtype=np.int64) * BINSIZE
        bin_starts_np = region_start_np[:, None] + bin_offsets[None, :]
        bin_ends_np = bin_starts_np + BINSIZE

        for i in range(label_cpu.shape[0]):
            t_id = int(tissue_id_tensor[i])
            active_channels = np.nonzero(mask_cpu[i])[0]
            if active_channels.size == 0:
                continue
            chrom = coords_chr[int(seq_np[i])]
            starts_np = bin_starts_np[i]
            ends_np = bin_ends_np[i]

            if int(seq_np[i]) not in mean_done:
                for c in active_channels:
                    exp_name = experiments[c]
                    mean_vals = mean_cpu[i, c].numpy().astype(float)
                    mean_lines.setdefault(exp_name, []).extend(
                        zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), mean_vals.tolist())
                    )
                mean_done.add(int(seq_np[i]))

            for c in active_channels:
                exp_name = experiments[c]
                gt_vals = label_cpu[i, c].numpy().astype(float)
                gt_lines.setdefault((t_id, exp_name), []).extend(
                    zip((chrom,) * TARGET_BINS, starts_np.tolist(), ends_np.tolist(), gt_vals.tolist())
                )

    for (t_id, exp_name), lines in gt_lines.items():
        export_bedgraph(lines, outdir / f"{tag}_t{t_id}_{exp_name}.bedgraph", chrom_sizes)
    for exp_name, lines in mean_lines.items():
        export_bedgraph(lines, outdir / f"mean_{tag}_{exp_name}.bedgraph", chrom_sizes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ground truth and mean baseline BigWigs for original and QN data.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for bedGraph/BigWig files")
    parser.add_argument("--num-tissues", type=int, default=3, help="Number of tissues to sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for tissue sampling")
    parser.add_argument("--fraction", type=float, default=1.0, help="Optional fraction of selected sequences to keep")
    parser.add_argument("--coords-bed", type=Path, default=Path(config_corgiplus["bed_path"]), help="BED file with coordinates")
    parser.add_argument("--mask-path", type=Path, default=Path(config_corgiplus["mask_path"]), help="Experiment mask .npy path")
    parser.add_argument(
        "--experiments-path",
        type=Path,
        default=Path(config_corgiplus["experiments_path"]),
        help="Path to experiments_final.txt",
    )
    parser.add_argument(
        "--orig-dir",
        type=Path,
        default=Path("/project/deeprna_data/pretraining_data_final2"),
        help="Directory containing original tissue_*.npy files",
    )
    parser.add_argument(
        "--qn-dir",
        type=Path,
        default=Path("/project/deeprna_data/revision_data_qn_parallel"),
        help="Directory containing quantile-normalized tissue_*.npy files",
    )
    parser.add_argument(
        "--orig-mean",
        type=Path,
        default=Path("/project/deeprna/data/revision/training_baseline_signal.npy"),
        help="Mean baseline .npy for original data",
    )
    parser.add_argument(
        "--qn-mean",
        type=Path,
        default=Path("/project/deeprna/data/revision/training_baseline_signal_qn.npy"),
        help="Mean baseline .npy for quantile-normalized data",
    )
    parser.add_argument(
        "--chrom-sizes",
        type=Path,
        default=Path("/project/deeprna_data/corgi-reproduction/data/hg38.chrom.sizes"),
        help="Chrom sizes file for bedGraphToBigWig",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Dataloader batch size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading experiment mask and metadata")
    experiment_mask = load_experiment_mask(str(args.mask_path))
    experiments = load_experiments(args.experiments_path)

    tissue_ids = choose_tissues(args.num_tissues, args.seed, experiment_mask, args.orig_dir, args.qn_dir)
    logging.info(f"Selected tissues: {tissue_ids}")

    coords_df = pd.read_csv(args.coords_bed, sep="\t", header=None, names=["chr", "start", "end", "fold"])
    total_seqs = len(coords_df)
    base_indices = np.arange(0, total_seqs, 3)  # match corgiplus_delta_benchmark: every 3rd sequence
    if args.fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = max(1, int(len(base_indices) * args.fraction))
        base_indices = np.sort(rng.choice(base_indices, size=keep, replace=False))
    logging.info(f"Using {len(base_indices)} sequences (every 3rd, fraction={args.fraction}) out of {total_seqs}")

    coords_chr = coords_df["chr"].tolist()
    coords_start = coords_df["start"].tolist()

    trans_reg_expression = torch.from_numpy(np.load(config_corgiplus["trans_regulator_expression_path"])).float()

    logging.info("Exporting original data tracks")
    collect_lines(
        label_dir=args.orig_dir,
        tissue_ids=tissue_ids,
        experiment_mask=experiment_mask,
        trans_reg_expression=trans_reg_expression,
        mean_baseline_file=args.orig_mean,
        coords_chr=coords_chr,
        coords_start=coords_start,
        chrom_sizes=args.chrom_sizes,
        outdir=outdir,
        experiments=experiments,
        tag="orig",
        sequence_ids=base_indices,
        batch_size=args.batch_size,
    )

    logging.info("Exporting quantile-normalized data tracks")
    collect_lines(
        label_dir=args.qn_dir,
        tissue_ids=tissue_ids,
        experiment_mask=experiment_mask,
        trans_reg_expression=trans_reg_expression,
        mean_baseline_file=args.qn_mean,
        coords_chr=coords_chr,
        coords_start=coords_start,
        chrom_sizes=args.chrom_sizes,
        outdir=outdir,
        experiments=experiments,
        tag="qn",
        sequence_ids=base_indices,
        batch_size=args.batch_size,
    )

    logging.info("Done")


if __name__ == "__main__":
    main()
