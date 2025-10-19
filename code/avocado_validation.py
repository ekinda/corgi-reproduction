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
DEFAULT_OUTPUT_DIR = PROCESSED_ROOT + "/avocado_validation"

VALID_TISSUES = [17, 39, 60, 66, 71, 82, 97, 136, 137, 227, 236, 240, 306, 312, 339, 342, 349]
with open('/project/deeprna_data/corgi-reproduction/data/experiments_final.txt', 'r') as f:
    ALLOWED_ASSAYS = f.read().strip().split()


for epoch in range(100, 5100, 100):
    print("Processing epoch {}".format(epoch))
    model = Avocado.load('{}/avocado_trainingfolds_epoch_{}'.format(DEFAULT_MODEL_DIR, epoch))

    predictions = {}
    for tissue in VALID_TISSUES:
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

# def list_available_assays(data_dir, tissue):
#     assays = []
#     for tensor_path in sorted(glob.glob("{data_dir}/tissue_{tissue}_*.npy".format(data_dir=data_dir, tissue=tissue))):
#         assay = tensor_path.split('/')[-1].split("_", 2)[2]
#         if assay in ALLOWED_ASSAYS:
#             assays.append(assay)
#     return assays


# def run_validation(data_dir, model_dir, output_dir, epochs, model_prefix):
#     for epoch in epochs:
#         model_path = "{model_dir}/{model_prefix}{epoch}".format(
#             model_dir=model_dir,
#             model_prefix=model_prefix,
#             epoch=epoch)
#         print("Processing epoch {epoch} using checkpoint {model_path}".format(epoch=epoch, model_path=model_path))
#         model = Avocado.load(str(model_path))

#         predictions = {}
#         for tissue in VALID_TISSUES:
#             available_assays = list_available_assays(data_dir, tissue)
#             print("Tissue {tissue}: {available_assays}".format(tissue=tissue, available_assays=available_assays))
#             for assay in available_assays:
#                 predictions.setdefault(assay, {})[tissue] = model.predict(tissue, assay)

#         out_path = "{output_dir}/avocado_trainingfolds_epoch_{epoch}_predictions.pkl".format(output_dir=output_dir, epoch=epoch)
#         with open(out_path, "wb") as handle:
#             pickle.dump(predictions, handle)
#         print("Saved validation predictions to {}".format(out_path))


# def parse_args():
#     parser = argparse.ArgumentParser(description="Generate Avocado validation predictions across multiple checkpoints.")
#     parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with prepared tissue_#_assay.npy tensors.")
#     parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing Avocado checkpoints.")
#     parser.add_argument("--model-prefix", default="avocado_trainingfolds_epoch_", help="Checkpoint prefix produced during training.")
#     parser.add_argument("--epochs", nargs='+', type=int, default=list(range(100, 5100, 100)), help="Epoch identifiers to evaluate.")
#     parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save validation prediction pickles.")
#     return parser.parse_args()

# args = parse_args()
# run_validation(
#     data_dir=args.data_dir,
#     model_dir=args.model_dir,
#     output_dir=args.output_dir,
#     epochs=args.epochs,
#     model_prefix=args.model_prefix,
# )