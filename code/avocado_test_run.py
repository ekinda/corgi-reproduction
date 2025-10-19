import argparse
import pickle
import glob
import numpy

numpy.random.seed(0)

import os

os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['THEANO_FLAGS']="device=gpu0,floatX=float32,dnn.enabled=False"
from avocado import Avocado

REPO_ROOT = '/project/deeprna_data/corgi-reproduction'
PROCESSED_ROOT = REPO_ROOT + "/processed_data/figure4"
DEFAULT_DATA_DIR = PROCESSED_ROOT + "/avocado_trainingfolds"
DEFAULT_MODEL_DIR = PROCESSED_ROOT + "/avocado_models"
DEFAULT_OUTPUT_DIR = PROCESSED_ROOT + "/avocado_test"

epoch = 2700
test_tissues = [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
                 214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
                 323, 324]

with open('/project/deeprna_data/corgi-reproduction/data/experiments_final.txt', 'r') as f:
    ALLOWED_ASSAYS = f.read().strip().split()


print("Processing epoch {}".format(epoch))
model = Avocado.load('{}/avocado_trainingfolds_epoch_{}'.format(DEFAULT_MODEL_DIR, epoch))

predictions = {}
for tissue in test_tissues:
    available_assays = []
    for file in glob.glob('{}/tissue_{}_*.npy'.format(DEFAULT_DATA_DIR, tissue)):
        assay = '_'.join(file.split('/')[-1].split('_')[2:]).split('.')[0]
        #print(assay)
        if assay in ['dnase', 'atac', 'h3k4me1', 'h3k4me3', 'h3k9ac', 'h3k27ac', 'ctcf']:
            available_assays.append(assay)
            
    print("Tissue {}: {}".format(tissue, available_assays))
    for assay in available_assays:
        if assay not in predictions:
            predictions[assay] = {}
        predictions[assay][tissue] = model.predict(tissue, assay)

with open('{}/avocado_trainingfolds_epoch_{}_predictions.pkl'.format(DEFAULT_OUTPUT_DIR, epoch), 'wb') as f:
    pickle.dump(predictions, f)
