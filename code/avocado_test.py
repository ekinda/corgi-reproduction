import pickle
import numpy; numpy.random.seed(0)
import os
import glob
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']="device=gpu0,floatX=float32,dnn.enabled=False"
from avocado import Avocado

test_tissues = [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
                             214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
                             323, 324, 422, 442, 443, 473, 474, 515, 517]

#valid_tissues = [17, 39, 60, 66, 70, 71, 72, 73, 82, 97, 98, 99, 136, 137, 227,
#                       236, 240, 306, 307, 312, 339, 340, 341, 342, 349, 425, 458, 482]

valid_tissues = [17, 39, 60, 66, 71, 82, 97, 136, 137, 227, 236, 240, 306, 312, 339, 342, 349]

easytest_tissues = [3, 25, 64, 87, 90, 98, 106, 115, 124, 137, 159, 189, 192, 247,
                     283, 284, 300, 311, 332, 334, 345, 448, 467, 480, 481, 532, 582]

celltypes = list(range(0,392))

with open('/project/deeprna_data/pretraining_data_final2/experiments_final.txt', 'r') as f:
    assays = f.read().strip().split()

epoch = 2900
print("Selected model: /project/deeprna/benchmark/avocado_models/avocado_trainingfolds_epoch_{}".format(epoch))
model = Avocado.load('/project/deeprna/benchmark/avocado_models/avocado_trainingfolds_epoch_{}'.format(epoch))

predictions = {}
for tissue in test_tissues:
    available_assays = []
    for file in glob.glob('/project/deeprna_data/avocado_data_trainingfolds/tissue_{}_*.npy'.format(tissue)):
        assay = '_'.join(file.split('/')[-1].split('_')[2:]).split('.')[0]
        if assay not in ['rna_total', 'rna_polya']:
            available_assays.append(assay)
            
    print("Tissue {}: {}".format(tissue, available_assays))
    for assay in available_assays:
        if assay not in predictions:
            predictions[assay] = {}
        predictions[assay][tissue] = model.predict(tissue, assay)

with open('/project/deeprna/benchmark/avocado_predictions/avocado_testtissues_trainingfolds_epoch_{}_predictions.pkl'.format(epoch), 'wb') as f:
    pickle.dump(predictions, f)