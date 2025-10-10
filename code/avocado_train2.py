import numpy; numpy.random.seed(0)
import itertools
import os, sys   
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = "device=cuda0"
import theano
from avocado import Avocado

print("Theano is using device:", theano.config.device)
print("Theano float type:", theano.config.floatX)

test_tissues = [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
                214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
                323, 324, 422, 442, 443, 473, 474, 515, 517]

valid_tissues = [17, 39, 60, 66, 70, 71, 72, 73, 82, 97, 98, 99, 136, 137, 227,
                 236, 240, 306, 307, 312, 339, 340, 341, 342, 349, 425, 458, 482]

easytest_tissues = [3, 25, 64, 87, 90, 98, 106, 115, 124, 137, 159, 189, 192, 247,
                    283, 284, 300, 311, 332, 334, 345, 448, 467, 480, 481, 532, 582]

excluded_tissues = test_tissues + valid_tissues + easytest_tissues

celltypes = list(range(0,392))
assays = ['dnase', 'atac', 'h3k4me1', 'h3k4me2', 'h3k4me3', 'h3k9ac', 'h3k9me3', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k79me2',
          'ctcf', 'cage', 'rampage', 'rna_total', 'rna_polya', 'rna_10x', 'wgbs']

data = {}
for celltype, assay in itertools.product(celltypes, assays):
    if celltype in excluded_tissues:
        if assay not in ['rna_polya', 'rna_total']:
            continue
    filename = '/project/deeprna_data/avocado_data_trainingfolds/tissue_{}_{}.npy'.format(celltype, assay)
    try:
        data[(celltype, assay)] = numpy.load(filename)
    except:
        continue

model = Avocado(celltypes, assays, n_genomic_positions=5611520, n_layers=2, n_nodes=2048, n_assay_factors=256, n_celltype_factors=32,
        n_25bp_factors=25, n_250bp_factors=40, n_5kbp_factors=45, batch_size=10000)

for epoch in range(1, 51):
    model.fit(data, n_epochs=20, epoch_size=561)
    model.save('/project/deeprna/benchmark/avocado/avocado_trainingfolds_epoch_{}'.format(epoch*100))
    print("Epoch {} completed and saved.".format(epoch*100))

