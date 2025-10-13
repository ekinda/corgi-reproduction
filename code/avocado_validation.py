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
DEFAULT_OUTPUT_DIR = PROCESSED_ROOT / "avocado_predictions" / "validation"

VALID_TISSUES = [17, 39, 60, 66, 71, 82, 97, 136, 137, 227, 236, 240, 306, 312, 339, 342, 349]
ALLOWED_ASSAYS = {'dnase', 'atac', 'h3k4me1', 'h3k4me3', 'h3k9ac', 'h3k27ac', 'ctcf'}


def list_available_assays(data_dir: Path, tissue: int):
    assays = []
    for tensor_path in sorted(data_dir.glob(f"tissue_{tissue}_*.npy")):
        assay = tensor_path.stem.split("_", 2)[2]
        if assay in ALLOWED_ASSAYS:
            assays.append(assay)
    return assays


def run_validation(data_dir: Path, model_dir: Path, output_dir: Path, epochs, model_prefix: str):
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing Avocado tensors at {data_dir}. Run avocado_prepare_data2.py first.")
    model_dir = model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing Avocado models at {model_dir}. Train the models before validation.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epochs:
        model_path = model_dir / f"{model_prefix}{epoch}"
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint {model_path} not found.")
        print(f"Processing epoch {epoch} using checkpoint {model_path}")
        model = Avocado.load(str(model_path))

        predictions = {}
        for tissue in VALID_TISSUES:
            available_assays = list_available_assays(data_dir, tissue)
            print(f"Tissue {tissue}: {available_assays}")
            for assay in available_assays:
                predictions.setdefault(assay, {})[tissue] = model.predict(tissue, assay)

        out_path = output_dir / f"avocado_trainingfolds_epoch_{epoch}_predictions.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(predictions, handle)
        print(f"Saved validation predictions to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Avocado validation predictions across multiple checkpoints.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with prepared tissue_#_assay.npy tensors.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing Avocado checkpoints.")
    parser.add_argument("--model-prefix", default="avocado_trainingfolds_epoch_", help="Checkpoint prefix produced during training.")
    parser.add_argument("--epochs", nargs='+', type=int, default=[2000, 2200, 2300, 2500, 2600, 2900, 3000, 3300], help="Epoch identifiers to evaluate.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save validation prediction pickles.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        model_prefix=args.model_prefix,
    )