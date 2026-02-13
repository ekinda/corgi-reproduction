# Merges bigwig file of plus and minus strand for each tissue and experiment.
# Outputs in bedGraph format (due to wiggletools bw output problems
# Requires wiggletools and bedgraphtobigwig

import os
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import re
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent

def compute_gene_level_expression(pred_bw, true_bw, exon_bed, outdir):
    os.makedirs(outdir, exist_ok=True)
    try:
        pred_out = os.path.join(outdir, os.path.basename(pred_bw) + ".genelevel.bg")
        true_out = os.path.join(outdir, os.path.basename(true_bw) + ".genelevel.bg")

        subprocess.run(
            ["bwtool", "summary", "-with-sum", "-keep-bed", "-header", exon_bed, pred_bw, pred_out],
            check=True
        )
        subprocess.run(
            ["bwtool", "summary", "-with-sum", "-keep-bed", "-header", exon_bed, true_bw, true_out],
            check=True
        )
    except Exception as e:
        print(f"Error running bwtool: {e}")
        return

    pred = pd.read_csv(pred_out, sep='\t', header='infer')
    true = pd.read_csv(true_out, sep='\t', header='infer')
    pred_genes = pred[['name', 'sum']].groupby('name').sum()
    true_genes = true[['name', 'sum']].groupby('name').sum()

    # Export gene expression
    pred_genes_export = np.log1p(pred_genes)
    pred_genes_export.to_csv(f'{outdir}/{os.path.basename(pred_bw)}.genes.csv', float_format='%.3f')
    true_genes_export = np.log1p(true_genes)
    true_genes_export.to_csv(f'{outdir}/{os.path.basename(true_bw)}.genes.csv', float_format='%.3f')
    
    # Compute correlation
    p = round(pearsonr(pred_genes, true_genes)[0][0], 3)
    logp = round(pearsonr(np.log1p(pred_genes), np.log1p(true_genes))[0][0], 3)
    s = round(spearmanr(pred_genes, true_genes)[0], 3)
    return (p, logp, s)
        

def main():
    bw_dir = REPO_ROOT / "processed_data_extended" / "figure2" / "cross_cell_tta"
    exon_bed = REPO_ROOT / "data" / "genebody_coding_trainingfolds_subset.bed"
    outdir = REPO_ROOT / "processed_data_extended" / "figure2" / "gene_level_cross_celltype_genebody"

    all_models = ['grt', 'encode']
    prediction_models = ['grt']
    truth_model = 'encode'
    accepted_experiments = ['rna_polya', 'rna_total']

    pattern = re.compile(r'tissue(\d+)_([a-zA-Z0-9_]+)_(' + '|'.join(all_models) + r')\.bw')
    
    file_dict = defaultdict(dict)
    for fname in os.listdir(bw_dir):
        match = pattern.match(fname)
        if match:
            tissue_id, experiment_type, model = match.groups()
            if experiment_type in accepted_experiments:
                file_dict[(tissue_id, experiment_type)][model] = os.path.join(bw_dir, fname)


    print(f"Found {len(file_dict)} tissue-experiment pairs.")
        
    correlations = []
    all_expr_dfs = {model: [] for model in prediction_models + [truth_model]}
    
    for (tissue_id, experiment_type), model_paths in file_dict.items():
        if truth_model not in model_paths:
            continue  # need ground truth
        
        for pred_model in prediction_models:
            if pred_model not in model_paths:
                continue
            
            pred_bw = model_paths[pred_model]
            true_bw = model_paths[truth_model]
    
            print(f"Processing: tissue {tissue_id}, experiment {experiment_type}, model {pred_model}")
    
            result = compute_gene_level_expression(
                pred_bw, true_bw, str(exon_bed), str(outdir)
            )
            if result is None:
                continue
            p, logp, s = result
            print(p, logp, s)
    
            correlations.append({
                'tissue': tissue_id,
                'experiment': experiment_type,
                'model': pred_model,
                'pearson': p,
                'log_pearson': logp,
                'spearman': s
            })
    
            # Read logged gene expressions for later
            pred_expr = pd.read_csv(f"{outdir}/{os.path.basename(pred_bw)}.genes.csv", index_col=0)
            true_expr = pd.read_csv(f"{outdir}/{os.path.basename(true_bw)}.genes.csv", index_col=0)
    
            pred_expr.columns = [f"{tissue_id}_{experiment_type}_{pred_model}"]
            true_expr.columns = [f"{tissue_id}_{experiment_type}_{truth_model}"]
    
            all_expr_dfs[pred_model].append(pred_expr)
            all_expr_dfs[truth_model].append(true_expr)

    cor_df = pd.DataFrame(correlations)
    cor_df.to_csv(os.path.join(outdir, "all_correlations.csv"), index=False)
    
    # === Save expressions and compute mean-subtracted correlations ===
    for pred_model in prediction_models:
        if not all_expr_dfs[pred_model]:
            continue

        pred_df = pd.concat(all_expr_dfs[pred_model], axis=1)
        true_df = pd.concat(all_expr_dfs[truth_model], axis=1)

        pred_df.to_csv(os.path.join(outdir, f"all_expressions_{pred_model}.csv"))
        true_df.to_csv(os.path.join(outdir, f"all_expressions_{truth_model}.csv"))

        # Align and mean-subtract
        pred_df, true_df = pred_df.align(true_df, join="inner", axis=0)
        pred_centered = pred_df.subtract(pred_df.mean(axis=1), axis=0)
        true_centered = true_df.subtract(true_df.mean(axis=1), axis=0)

        corr_series = pred_centered.corrwith(true_centered, axis=0)
        corr_df = corr_series.reset_index()
        corr_df.columns = ['comparison', 'mean_subtracted_correlation']
        corr_df.to_csv(os.path.join(outdir, f"mean_subtracted_correlation_{pred_model}.csv"), index=False)

    print("All done!")

if __name__ == "__main__":
    main()