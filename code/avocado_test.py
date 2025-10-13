import argparse
import pickle
from pathlib import Path

import numpy

numpy.random.seed(0)

import os

os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['THEANO_FLAGS']="device=gpu0,floatX=float32,dnn.enabled=False"
from avocado import Avocado

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = REPO_ROOT / "processed_data" / "figure4"
DEFAULT_DATA_DIR = PROCESSED_ROOT / "avocado_trainingfolds"
DEFAULT_MODEL_DIR = PROCESSED_ROOT / "avocado_models"
DEFAULT_OUTPUT_DIR = PROCESSED_ROOT / "avocado_predictions" / "test"

TEST_TISSUES = [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
                214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
                323, 324, 422, 442, 443, 473, 474, 515, 517]

EXCLUDED_ASSAYS = {'rna_total', 'rna_polya'}


def list_available_assays(data_dir: Path, tissue: int):
    assays = []
    for tensor_path in sorted(data_dir.glob(f"tissue_{tissue}_*.npy")):
        assay = tensor_path.stem.split("_", 2)[2]
        if assay in EXCLUDED_ASSAYS:
            continue
        assays.append(assay)
    return assays


def run_inference(data_dir: Path, model_path: Path, output_dir: Path, epoch: int):
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing Avocado tensors at {data_dir}. Run avocado_prepare_data2.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained Avocado model at {model_path}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected model: {model_path}")
    model = Avocado.load(str(model_path))

    predictions = {}
    for tissue in TEST_TISSUES:
        available_assays = list_available_assays(data_dir, tissue)
        print(f"Tissue {tissue}: {available_assays}")
        for assay in available_assays:
            predictions.setdefault(assay, {})[tissue] = model.predict(tissue, assay)

    out_path = output_dir / f"avocado_testtissues_trainingfolds_epoch_{epoch}_predictions.pkl"
    with out_path.open("wb") as handle:
        pickle.dump(predictions, handle)
    print(f"Wrote predictions to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Avocado predictions for the held-out test tissues.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with prepared tissue_#_assay.npy tensors.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing Avocado checkpoints.")
    parser.add_argument("--model-prefix", default="avocado_trainingfolds_epoch_", help="Checkpoint prefix used during training.")
    parser.add_argument("--epoch", type=int, default=2900, help="Checkpoint epoch number to load.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save pickled predictions.")
    parser.add_argument("--model-path", default=None, help="Optional explicit path to the Avocado checkpoint. Overrides --model-dir, --model-prefix, and --epoch.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(args.model_dir) / f"{args.model_prefix}{args.epoch}"
    run_inference(
        data_dir=Path(args.data_dir),
        model_path=model_path,
        output_dir=Path(args.output_dir),
        epoch=args.epoch,
    )