import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_INPUT_DIR = DATA_DIR / "pretraining_data_final2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "processed_data" / "figure4" / "avocado_trainingfolds"
DEFAULT_BED = DEFAULT_INPUT_DIR / "hg38_sequence_folds_tfexcluded34.bed"
DEFAULT_SUBSET_BED = DATA_DIR / "figure4" / "borzoi_trainingfolds_subset_2.bed"
DEFAULT_TRACKS = DATA_DIR / "experiments_final.txt"
DEFAULT_MASK = DEFAULT_INPUT_DIR / "experiment_mask.npy"

def prepare_avocado(
    input_dir: str,
    output_dir: str,
    bed_path: str,
    subset_bed_path: str,
    tracks_txt: str,
    mask_path: str,
    start_tissue: int,
    end_tissue: int,
):
    # create output directory
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    bed_path = Path(bed_path)
    subset_bed_path = Path(subset_bed_path)
    tracks_txt = Path(tracks_txt)
    mask_path = Path(mask_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # load track names and mask
    with open(tracks_txt) as f:
        tracks = [line.strip() for line in f]
    mask = np.load(mask_path)  # shape (600, 22)

    # load original and subset BED files
    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "fold_id"]
    )
    subset_bed = pd.read_csv(
        subset_bed_path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "fold_id"]
    )

    # find row indices in the original bed that match rows in subset
    bed_index = pd.merge(subset_bed, bed.reset_index(), on=["chr", "start", "end", "fold_id"], how="left")["index"].tolist()
    sel_idx = sorted(bed_index)

    # write selected BED
    sel_bed = bed.loc[sel_idx]
    sel_bed.to_csv(
        output_dir / "selected_regions.bed",
        sep="\t", index=False, header=False
    )

    # define plus/minus pairs for combination
    combine_pairs = [
        ("cage_plus", "cage_minus", "cage"),
        ("rampage_plus", "rampage_minus", "rampage"),
        ("rna_total_plus", "rna_total_minus", "rna_total"),
        ("rna_polya_plus", "rna_polya_minus", "rna_polya"),
    ]

    for ti in range(start_tissue, end_tissue + 1):
        if ti >= mask.shape[0]:
            raise ValueError(
                f"Tissue index {ti} exceeds mask rows ({mask.shape[0]}). Check --end-tissue."
            )
        tissue_name = f"tissue_{ti}"
        tissue_file = input_dir / f"{tissue_name}.npy"
        if not tissue_file.exists():
            raise FileNotFoundError(f"Missing tensor for {tissue_name} at {tissue_file}.")
        data = np.load(tissue_file)
        # global track indices available for this tissue
        global_avail = np.where(mask[ti] == 1)[0]
        # map global index to local data axis
        local_map = {}
        for i, g in enumerate(global_avail):
            if g >= len(tracks):
                raise IndexError(
                    f"Mask index {g} for {tissue_name} exceeds track list length {len(tracks)}."
                )
            local_map[tracks[g]] = i

        # slice selected regions
        selected = data[sel_idx, :, :]
        processed = set()
        # combine strand-specific tracks
        for plus, minus, base in combine_pairs:
            if plus in local_map and minus in local_map:
                pi, mi = local_map[plus], local_map[minus]
                arr = selected[:, :, pi] + selected[:, :, mi]
                flat = arr.ravel()
                np.save(output_dir / f"{tissue_name}_{base}.npy", flat)
                processed.update({plus, minus})

        # save remaining tracks
        for track, idx in local_map.items():
            if track in processed:
                continue
            arr = selected[:, :, idx]
            flat = arr.ravel()
            np.save(output_dir / f"{tissue_name}_{track}.npy", flat)

        print(f"Processed tissue {tissue_name} with {len(processed)} combined tracks and {len(local_map) - len(processed)} single tracks.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare Avocado training-fold tensors using repository-relative paths.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing tissue_*.npy tensors (default: data/avocado/pretraining_data_final2)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for processed Avocado tensors (default: processed_data/figure4/avocado_trainingfolds)")
    parser.add_argument("--bed-path", default=str(DEFAULT_BED), help="BED file listing the full training regions (default: data/avocado/pretraining_data_final2/hg38_sequence_folds_tfexcluded34.bed)")
    parser.add_argument("--subset-bed-path", default=str(DEFAULT_SUBSET_BED), help="Subset BED defining the regions used for reproduction (default: data/figure4/borzoi_trainingfolds_subset.bed)")
    parser.add_argument("--tracks", default=str(DEFAULT_TRACKS), help="Text file with Avocado experiment names (default: data/experiments_final.txt)")
    parser.add_argument("--mask", default=str(DEFAULT_MASK), help="NumPy mask array with available assays per tissue (default: data/avocado/pretraining_data_final2/experiment_mask.npy)")
    parser.add_argument("--start-tissue", type=int, default=0, help="First tissue index to process")
    parser.add_argument("--end-tissue", type=int, default=599, help="Last tissue index to process")
    args = parser.parse_args()

    required = {
        "input directory": Path(args.input_dir),
        "full BED": Path(args.bed_path),
        "subset BED": Path(args.subset_bed_path),
        "tracks file": Path(args.tracks),
        "experiment mask": Path(args.mask),
    }
    missing = [f"{label} -> {path}" for label, path in required.items() if not path.exists()]
    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(
            "Missing required Avocado inputs. Ensure the benchmark assets are copied into the repository.\n" + message
        )

    if args.start_tissue > args.end_tissue:
        raise ValueError("--start-tissue must be <= --end-tissue.")

    prepare_avocado(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        bed_path=args.bed_path,
        subset_bed_path=args.subset_bed_path,
        tracks_txt=args.tracks,
        mask_path=args.mask,
        start_tissue=args.start_tissue,
        end_tissue=args.end_tissue,
    )
