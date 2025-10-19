import pandas as pd
import numpy as np
import qnorm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

genes_df = pd.read_csv('data/Homo_sapiens.gene_info.tsv', sep='\t')
coding_genes = genes_df.loc[genes_df.type_of_gene == 'protein-coding', 'Symbol'].values

def ms_corrs(pred_exp, true_exp, genes, out=None):
    true = pd.read_csv(true_exp, index_col=0)
    pred = pd.read_csv(pred_exp, index_col=0)
    
    true = true.loc[true.index.isin(genes)]
    pred = pred.loc[pred.index.isin(genes)]
    
    true_ms = true.sub(true.mean(axis=1), axis=0)
    pred_ms = pred.sub(pred.mean(axis=1), axis=0)
    
    result = {'sample':[], 'log-TPM-pearson':[], 'mean-subtracted-pearson':[], 'mean-sub-spearman':[], 'mean-sub-scaled-spearman':[]} 

    for i in range(true.shape[1]):
        x = pred.iloc[:,i]
        y = true.iloc[:,i]
        x_ms = pred_ms.iloc[:,i]
        y_ms = true_ms.iloc[:,i]
        x_scaled = x_ms / (x + 0.001)
        y_scaled = y_ms / (y + 0.001)
        result['sample'].append(true.columns[i])
        result['log-TPM-pearson'].append(pearsonr(x,y)[0])
        result['mean-subtracted-pearson'].append(pearsonr(x_ms,y_ms)[0])
        result['mean-sub-spearman'].append(spearmanr(x_ms,y_ms)[0])
        result['mean-sub-scaled-spearman'].append(spearmanr(x_scaled,y_scaled)[0])

    print(f'Number of genes:{len(x)}')
    df = pd.DataFrame.from_dict(result)
    if out:
        df.to_csv(out)
    return df


cross_celltype = ms_corrs(
    'processed_data/figure2/gene_level_cross_celltype/all_expressions_grt.csv',
    'processed_data/figure2/gene_level_cross_celltype/all_expressions_encode.csv',
    coding_genes,
    out='processed_data/figure2/gene_level_cross_celltype/correlations_coding_genes.csv')

cross_sequence = ms_corrs(
    'processed_data/figure2/gene_level_cross_region/all_expressions_grt.csv',
    'processed_data/figure2/gene_level_cross_region/all_expressions_encode.csv',
    coding_genes,
    out='processed_data/figure2/gene_level_cross_region/correlations_coding_genes.csv')

cross_both = ms_corrs(
    'processed_data/figure2/gene_level_cross_both/all_expressions_grt.csv',
    'processed_data/figure2/gene_level_cross_both/all_expressions_encode.csv',
    coding_genes,
    out='processed_data/figure2/gene_level_cross_both/correlations_coding_genes.csv')
