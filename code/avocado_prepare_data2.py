import os
import numpy as np
import pandas as pd

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
    os.makedirs(output_dir, exist_ok=True)

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
        os.path.join(output_dir, "selected_regions.bed"),
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
        tissue_name = f"tissue_{ti}"
        data = np.load(os.path.join(input_dir, f"{tissue_name}.npy"))
        # global track indices available for this tissue
        global_avail = np.where(mask[ti] == 1)[0]
        # map global index to local data axis
        local_map = {tracks[g]: i for i, g in enumerate(global_avail)}

        # slice selected regions
        selected = data[sel_idx, :, :]

        processed = set()
        # combine strand-specific tracks
        for plus, minus, base in combine_pairs:
            if plus in local_map and minus in local_map:
                pi, mi = local_map[plus], local_map[minus]
                arr = selected[:, :, pi] + selected[:, :, mi]
                flat = arr.ravel()
                np.save(
                    os.path.join(output_dir, f"{tissue_name}_{base}.npy"),
                    flat
                )
                processed.update({plus, minus})

        # save remaining tracks
        for track, idx in local_map.items():
            if track in processed:
                continue
            arr = selected[:, :, idx]
            flat = arr.ravel()
            np.save(
                os.path.join(output_dir, f"{tissue_name}_{track}.npy"),
                flat
            )
        
        print(f"Processed tissue {tissue_name} with {len(processed)} combined tracks and {len(local_map) - len(processed)} single tracks.")

if __name__ == '__main__':
    prepare_avocado(
        input_dir='/project/deeprna_data/pretraining_data_final2',
        output_dir='/project/deeprna_data/avocado_data_trainingfolds',
        bed_path='/project/deeprna_data/pretraining_data_final2/hg38_sequence_folds_tfexcluded34.bed',
        subset_bed_path='/project/deeprna/benchmark/borzoi_trainingfolds_subset.bed',
        tracks_txt='/project/deeprna_data/pretraining_data_final2/experiments_final.txt',
        mask_path='/project/deeprna_data/pretraining_data_final2/experiment_mask.npy',
        start_tissue=0,
        end_tissue=392
    )
