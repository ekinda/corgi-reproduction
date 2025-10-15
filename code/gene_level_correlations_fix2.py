#!/usr/bin/env python3
"""Compute corrected gene-level correlations and emit correlations_fixed.pk files for Figure 2 outputs."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG2_DIR = REPO_ROOT / "processed_data" / "figure2"

DATASETS = {
    "cross_celltype": {
        "directory": FIG2_DIR / "gene_level_cross_celltype",
        "true_suffix": "_encode",
        "pred_suffix": "_grt",
        "true_file": "all_expressions_encode.csv",
        "pred_file": "all_expressions_grt.csv",
    },
    "cross_both": {
        "directory": FIG2_DIR / "gene_level_cross_both",
        "true_suffix": "_encode",
        "pred_suffix": "_grt",
        "true_file": "all_expressions_encode.csv",
        "pred_file": "all_expressions_grt.csv",
    },
    "cross_region": {
        "directory": FIG2_DIR / "gene_level_cross_region",
        "true_suffix": "_encode",
        "pred_suffix": "_grt",
        "true_file": "all_expressions_encode.csv",
        "pred_file": "all_expressions_grt.csv",
    },
}

cells = pd.read_csv('/project/deeprna/data/all_cell_types.csv', index_col=0)
cells.head()

genes_df = pd.read_csv('/project/deeprna/data/Homo_sapiens.gene_info.tsv', sep='\t')
coding_genes = genes_df.loc[genes_df.type_of_gene == 'protein-coding']

for dataset, files in DATASETS.items():
    encode = pd.read_csv(f"{files['directory']}/{files['true_file']}", index_col=0, header='infer')
    grt = pd.read_csv(f"{files['directory']}/{files['pred_file']}", index_col=0, header='infer')

    encode = encode.loc[encode.index.isin(coding_genes.Symbol)]
    grt = grt.loc[grt.index.isin(coding_genes.Symbol)]

    cors = []
    cors_df = {}
    for col in range(encode.shape[1]):
        sample_idx = int(encode.columns[col].split('_')[0])
        y = encode.iloc[:, col]
        x = grt.iloc[:, col]
        r, _ = pearsonr(x, y)
        n = len(x)

        #print([encode.index[i] for i in range(len(encode)) if encode.iloc[i, col] < 1 and grt.iloc[i, col] > 5])
        cors.append(r)
        cors_df[encode.columns[col]] = r
        # sns.scatterplot(x=x, y=y, s=20, alpha=0.8)
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        # plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black')
        # plt.text(
        #     0.05, 0.95,
        #     f"r = {r:.3f}\nn = {n}",
        #     transform=plt.gca().transAxes,
        #     fontsize=10,
        #     weight='bold',
        #     verticalalignment='top'
        # )
        # plt.xlabel('Predicted expression (log)')
        # plt.ylabel('Measured expression (log)')
        # plt.title(f"RNA-seq in {cells.loc[sample_idx, 'tissue_name']}")
        # plt.tight_layout()
        # plt.show()

    with open(f'{files["directory"]}/correlations_fixed.pk', 'wb') as f:
        pickle.dump(cors_df, f)
    print(cors_df)
    print('\n')