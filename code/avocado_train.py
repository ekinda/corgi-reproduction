import argparse
import itertools
import os

import numpy

numpy.random.seed(0)

os.environ['KERAS_BACKEND'] = 'theano'
# Use GPU with compatibility settings to avoid NVRTC errors
# If you get NVRTC errors, the issue is likely CUDA version compatibility
# Try: device=cuda (instead of cuda0), and add force_device=True if needed
os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32,optimizer=fast_run,gpuarray.preallocate=0.8,exception_verbosity=high"
import theano
from avocado import Avocado

REPO_ROOT = '/project/deeprna_data/corgi-reproduction'
PROCESSED_ROOT = REPO_ROOT + "/processed_data/figure4"
DEFAULT_DATA_DIR = PROCESSED_ROOT + "/avocado_trainingfolds"
DEFAULT_MODEL_DIR = PROCESSED_ROOT + "/avocado_models"

TEST_TISSUES = [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
                214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
                323, 324, 422, 442, 443, 473, 474, 515, 517]

VALID_TISSUES = [17, 39, 60, 66, 70, 71, 72, 73, 82, 97, 98, 99, 136, 137, 227,
                 236, 240, 306, 307, 312, 339, 340, 341, 342, 349, 425, 458, 482]

EASYTEST_TISSUES = [3, 25, 64, 87, 90, 98, 106, 115, 124, 137, 159, 189, 192, 247,
                    283, 284, 300, 311, 332, 334, 345, 448, 467, 480, 481, 532, 582]

CELLTYPES = list(range(0, 392))
ASSAYS = ['dnase', 'atac', 'h3k4me1', 'h3k4me2', 'h3k4me3', 'h3k9ac', 'h3k9me3', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k79me2',
          'ctcf', 'cage', 'rampage', 'rna_total', 'rna_polya', 'rna_10x', 'wgbs']


def load_training_tensors(data_dir):
    data = {}
    excluded_tissues = set(TEST_TISSUES + VALID_TISSUES + EASYTEST_TISSUES)
    for celltype, assay in itertools.product(CELLTYPES, ASSAYS):
        if celltype in excluded_tissues and assay not in ['rna_polya', 'rna_total']:
            continue
        tensor_path = data_dir + "/tissue_{}_{}.npy".format(celltype, assay)
        try:
            data[(celltype, assay)] = numpy.load(tensor_path)
        except:
            print("Warning: missing tensor at {}".format(tensor_path))
    if not data:
        print("No Avocado tensors found in {}. Run avocado_prepare_data2.py first.".format(data_dir))
        exit(1)
    return data


def train(data_dir, model_dir, checkpoint_prefix, num_checkpoints,
          checkpoint_step, checkpoint_offset, fit_epochs, epoch_size):
    print("Theano is using device:", theano.config.device)
    print("Theano float type:", theano.config.floatX)

    #os.makedirs(model_dir)

    data = load_training_tensors(data_dir)

    model = Avocado(
        CELLTYPES,
        ASSAYS,
        n_genomic_positions=5611520,
        n_layers=2,
        n_nodes=2048,
        n_assay_factors=256,
        n_celltype_factors=32,
        n_25bp_factors=25,
        n_250bp_factors=40,
        n_5kbp_factors=45,
        batch_size=10000,
    )

    for checkpoint_idx in range(1, num_checkpoints + 1):
        model.fit(data, n_epochs=fit_epochs, epoch_size=epoch_size)
        tag = checkpoint_offset + checkpoint_idx * checkpoint_step
        save_path = model_dir + "/{checkpoint_prefix}{tag}".format(
            checkpoint_prefix=checkpoint_prefix, tag=tag
        )
        model.save(str(save_path))
        print("Epoch {tag} completed and saved to {save_path}.".format(tag=tag, save_path=save_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Avocado models with repository-relative inputs.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with prepared tissue_#_assay.npy tensors.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory to save Avocado checkpoints.")
    parser.add_argument("--checkpoint-prefix", default="avocado_trainingfolds_epoch_", help="Filename prefix for saved checkpoints.")
    parser.add_argument("--num-checkpoints", type=int, default=50, help="Number of checkpoints to export.")
    parser.add_argument("--checkpoint-step", type=int, default=100, help="Increment added to checkpoint number each iteration.")
    parser.add_argument("--checkpoint-offset", type=int, default=0, help="Offset added before applying checkpoint multipliers.")
    parser.add_argument("--fit-epochs", type=int, default=20, help="Number of epochs per training iteration passed to model.fit().")
    parser.add_argument("--epoch-size", type=int, default=561, help="Number of batches per epoch passed to model.fit().")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        num_checkpoints=args.num_checkpoints,
        checkpoint_step=args.checkpoint_step,
        checkpoint_offset=args.checkpoint_offset,
        fit_epochs=args.fit_epochs,
        epoch_size=args.epoch_size,
    )

