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

def _strip_strand_suffix(bw_path):
    base = os.path.basename(bw_path)
    return re.sub(r'_(plus|minus)\.bw$', '', base)

def compute_gene_level_expression(pred_bw_plus, pred_bw_minus, true_bw_plus, true_bw_minus, exon_bed, outdir):
    os.makedirs(outdir, exist_ok=True)
    try:
        pred_base = _strip_strand_suffix(pred_bw_plus)
        true_base = _strip_strand_suffix(true_bw_plus)

        pred_plus_out = os.path.join(outdir, pred_base + ".plus.genelevel.bg")
        pred_minus_out = os.path.join(outdir, pred_base + ".minus.genelevel.bg")
        true_plus_out = os.path.join(outdir, true_base + ".plus.genelevel.bg")
        true_minus_out = os.path.join(outdir, true_base + ".minus.genelevel.bg")

        subprocess.run(
            ["bwtool", "summary", "-with-sum", "-keep-bed", "-header", exon_bed, pred_bw_plus, pred_plus_out],
            check=True
        )
        subprocess.run(
            ["bwtool", "summary", "-with-sum", "-keep-bed", "-header", exon_bed, pred_bw_minus, pred_minus_out],
            check=True
        )
        subprocess.run(
            ["bwtool", "summary", "-with-sum", "-keep-bed", "-header", exon_bed, true_bw_plus, true_plus_out],
            check=True
        )
        subprocess.run(
            ["bwtool", "summary", "-with-sum", "-keep-bed", "-header", exon_bed, true_bw_minus, true_minus_out],
            check=True
        )
    except Exception as e:
        print(f"Error running bwtool: {e}")
        return

    pred_plus = pd.read_csv(pred_plus_out, sep='\t', header='infer')
    pred_minus = pd.read_csv(pred_minus_out, sep='\t', header='infer')
    true_plus = pd.read_csv(true_plus_out, sep='\t', header='infer')
    true_minus = pd.read_csv(true_minus_out, sep='\t', header='infer')

    pred_plus_genes = pred_plus[['name', 'sum']].groupby('name').sum()
    pred_minus_genes = pred_minus[['name', 'sum']].groupby('name').sum()
    true_plus_genes = true_plus[['name', 'sum']].groupby('name').sum()
    true_minus_genes = true_minus[['name', 'sum']].groupby('name').sum()

    pred_genes = pred_plus_genes.add(pred_minus_genes, fill_value=0)
    true_genes = true_plus_genes.add(true_minus_genes, fill_value=0)

    # Export gene expression
    pred_genes_export = np.log1p(pred_genes)
    pred_genes_export.to_csv(f'{outdir}/{pred_base}.genes.csv', float_format='%.3f')
    true_genes_export = np.log1p(true_genes)
    true_genes_export.to_csv(f'{outdir}/{true_base}.genes.csv', float_format='%.3f')
    
    # Compute correlation
    pred_genes, true_genes = pred_genes.align(true_genes, join="inner", axis=0)
    pred_values = pred_genes['sum']
    true_values = true_genes['sum']

    p = round(pearsonr(pred_values, true_values)[0], 3)
    logp = round(pearsonr(np.log1p(pred_values), np.log1p(true_values))[0], 3)
    s = round(spearmanr(pred_values, true_values)[0], 3)
    return (p, logp, s)
        

def main():
    bw_dir = REPO_ROOT / "processed_data_extended" / "figure3" / "grt_vs_borzoi"
    # exon_bed = REPO_ROOT / "data" / "exons_coding_fold3.bed"
    exon_bed = '/project/deeprna_data/corgi-reproduction/data/genebody_coding_fold3.bed'
    outdir = REPO_ROOT / "processed_data_extended" / "figure2" / "gene_level_cross_region_genebody_vsborzoi"

    all_models = ['borzoi', 'corgi', 'encode']
    prediction_models = ['borzoi', 'corgi']
    truth_model = 'encode'
    accepted_experiments = ['rna_polya', 'rna_total']

    pattern = re.compile(r'tissue(\d+)_([a-zA-Z0-9_]+)_(' + '|'.join(all_models) + r')\.bw')
    
    file_dict = defaultdict(lambda: defaultdict(dict))
    for fname in os.listdir(bw_dir):
        match = pattern.match(fname)
        if match:
            tissue_id, experiment_type, model = match.groups()
            strand = None
            if experiment_type.endswith("_plus") or experiment_type.endswith("_minus"):
                experiment_base, strand = experiment_type.rsplit("_", 1)
            else:
                experiment_base = experiment_type

            if experiment_base in accepted_experiments:
                file_dict[(tissue_id, experiment_base)][model][strand] = os.path.join(bw_dir, fname)


    print(f"Found {len(file_dict)} tissue-experiment pairs.")
        
    correlations = []
    all_expr_dfs = {model: [] for model in prediction_models + [truth_model]}
    
    for (tissue_id, experiment_type), model_paths in file_dict.items():
        if truth_model not in model_paths:
            continue  # need ground truth
        
        for pred_model in prediction_models:
            if pred_model not in model_paths:
                continue
            
            pred_paths = model_paths.get(pred_model, {})
            true_paths = model_paths.get(truth_model, {})

            if not {"plus", "minus"}.issubset(pred_paths.keys()):
                continue
            if not {"plus", "minus"}.issubset(true_paths.keys()):
                continue

            pred_bw_plus = pred_paths["plus"]
            pred_bw_minus = pred_paths["minus"]
            true_bw_plus = true_paths["plus"]
            true_bw_minus = true_paths["minus"]
    
            print(f"Processing: tissue {tissue_id}, experiment {experiment_type}, model {pred_model}")
    
            result = compute_gene_level_expression(
                pred_bw_plus, pred_bw_minus, true_bw_plus, true_bw_minus, str(exon_bed), str(outdir)
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
            pred_base = _strip_strand_suffix(pred_bw_plus)
            true_base = _strip_strand_suffix(true_bw_plus)

            pred_expr = pd.read_csv(f"{outdir}/{pred_base}.genes.csv", index_col=0)
            true_expr = pd.read_csv(f"{outdir}/{true_base}.genes.csv", index_col=0)
    
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