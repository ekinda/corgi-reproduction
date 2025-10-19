import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import pickle
from benchmark_utils import tile_regions, parse_bed_file_with_coords
np.random.seed(0)

epoch = 2700
test_tissues = [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
                 214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
                 323, 324, 422, 442, 443, 473, 474, 515, 517]

assays = ['dnase', 'atac', 'h3k4me1', 'h3k4me2', 'h3k4me3', 'h3k9ac', 'h3k9me3', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k79me2',
          'ctcf', 'cage', 'rampage', 'rna_total', 'rna_polya', 'rna_10x', 'wgbs']
celltypes = list(range(0,392))

root_path = '/project/deeprna_data/corgi-reproduction/'
subset_bed_path = f'{root_path}/processed_data/figure4/avocado_trainingfolds/selected_regions.bed'

subset_bed = pd.read_csv(
    subset_bed_path,
    sep="\t",
    header=None,
    names=["chr", "start", "end", "fold_id"]
)

with open(f'{root_path}/processed_data/figure4/training_baselines.pk', 'rb') as f:
    training_baseline = pickle.load(f)

with open(f'{root_path}/processed_data/figure4/avocado_test/avocado_trainingfolds_epoch_{epoch}_predictions.pkl', 'rb') as f:
    predictions = pickle.load(f, encoding='latin1')

# Load BED regions.
bed_regions = parse_bed_file_with_coords(subset_bed_path)
print(f"Parsed {len(bed_regions)} regions from BED file.")

# Tile regions for predictions.
pred_tiles = tile_regions(bed_regions, 64, 64, drop_last=True)
pred_tiles = [(a,str(b), str(c)) for (a,b,c) in pred_tiles]
print(f"Generated {len(pred_tiles)} prediction tiles.")

with open(f'{root_path}/processed_data/figure4/avocado_tiles.bed', 'w') as f:
    f.write('\n'.join(['\t'.join(x) for x in pred_tiles]))

results = {'tissue':[], 'assay':[], 'model':[], 'pearson':[], 'spearman':[]}

for assay in predictions:
    for tissue in predictions[assay]:
        true = np.load(f'{root_path}/processed_data/figure4/avocado_trainingfolds/tissue_{tissue}_{assay}.npy')
        corgi = pd.read_csv(f'{root_path}/data/figure4/avocado_vs_corgi/tissue{tissue}_{assay}_corgi.bed', sep='\t')
        avail = corgi.loc[corgi['mean'].isna() == False].index

        y = {}
        y['corgi'] = corgi.loc[avail, 'mean'].values.astype(float)
        y['avocado'] = predictions[assay][tissue][avail].astype(float)
        y['baseline'] = training_baseline[assay][avail].astype(float)
        y['true'] = true[avail].astype(float)

        for model in ['corgi', 'avocado', 'baseline']:
            p = round(pearsonr(y['true'], y[model])[0], 3)
            s = round(spearmanr(y['true'], y[model])[0], 3)
            
            results['tissue'].append(tissue)
            results['assay'].append(assay)
            results['model'].append(model)
            results['pearson'].append(p)
            results['spearman'].append(s)

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv(f'{root_path}/processed_data/figure4/avocado_vs_corgi_results.csv')