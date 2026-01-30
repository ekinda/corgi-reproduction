"""EpiGePT inference wrapper with on-the-fly TFBS scanning.

This matches the original motif scanning pipeline used to build TFBS matrices:
- Split the 128,000 bp sequence into 1000 bins of 128 bp.
- Scan the full sequence with MOODS (Python API) using the same PFMs and threshold.
- For each (bin, TF), keep the max score (float16, rounded to 3 decimals).

Usage (Python):
    from epigept_inference_wrapper import EpigeptInferenceWrapper
    wrapper = EpigeptInferenceWrapper(
        checkpoint_path="/path/to/model.ckpt",
        pfm_dir="/project/deeprna/epigept_directory/motif_pfms",
        tf_csv="/project/deeprna/epigept_directory/tf_expression.csv",
        threshold=4,
    )
    preds = wrapper(dna_seq, tf_expression)

Inputs:
- dna_seq: str (length 128,000, A/C/G/T only) OR one-hot array (128000, 4)
- tf_expression: array-like length 711

Returns:
- torch.Tensor on CPU with shape (1000, NUM_SIGNALS=22)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Make epigept package importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPIGEPT_ROOT = PROJECT_ROOT.parent / "epigept_directory"
if str(EPIGEPT_ROOT) not in sys.path:
    sys.path.append(str(EPIGEPT_ROOT))

from epigept.model import EpiGePT as epigept_module
from epigept.model import config as epigept_config

SEQ_LENGTH = 128_000
BINSIZE = 128
TFBINS = 1000

class EpigeptInferenceWrapper:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        pfm_dir: Union[str, Path],
        tf_csv: Union[str, Path],
        threshold: float = 4,
        device: Optional[Union[str, torch.device]] = None,
        window_size: int = 7,
        pseudocount: float = 0.01,
        bg: Optional[List[float]] = None,
        lo_bg: Optional[List[float]] = None,
        log_base: Optional[float] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.pfm_dir = Path(pfm_dir)
        self.tf_csv = Path(tf_csv)
        self.threshold = threshold
        self.window_size = window_size
        self.pseudocount = pseudocount
        self.bg = bg or [0.25, 0.25, 0.25, 0.25]
        self.lo_bg = lo_bg or [0.25, 0.25, 0.25, 0.25]
        self.log_base = log_base

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.pfm_dir.is_dir():
            raise FileNotFoundError(f"PFM directory not found: {self.pfm_dir}")
        if not self.tf_csv.exists():
            raise FileNotFoundError(f"TF CSV not found: {self.tf_csv}")

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_epigept_model(self.checkpoint_path, self.device)

        self.tf_names = self._load_tf_names(self.tf_csv)
        if len(self.tf_names) != epigept_config.TF_DIM:
            raise ValueError(f"TF CSV must define {epigept_config.TF_DIM} TFs; found {len(self.tf_names)}")

        self.pfm_files = self._resolve_pfm_files(self.pfm_dir, self.tf_names)
        self._init_moods()

    @staticmethod
    def _load_epigept_model(checkpoint_path: Path, device: torch.device) -> epigept_module.EpiGePT:
        model = epigept_module.EpiGePT(
            epigept_config.WORD_NUM,
            epigept_config.SEQUENCE_DIM,
            epigept_config.TF_DIM,
            batch_size=1,
        )
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(f"Missing epigept parameters in checkpoint: {missing}")
        if unexpected:
            raise ValueError(f"Unexpected epigept parameters in checkpoint: {unexpected}")
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_tf_names(tf_csv: Path) -> List[str]:
        tfs_df = pd.read_csv(tf_csv, sep="\t", index_col=0)
        return [str(tf) for tf in tfs_df.index]

    @staticmethod
    def _resolve_pfm_files(pfm_dir: Path, tf_names: List[str]) -> Dict[str, Optional[Path]]:
        pfm_map = {p.stem: p for p in pfm_dir.glob("*.pfm")}
        missing = [tf for tf in tf_names if tf not in pfm_map]
        if missing:
            logging.warning("Missing PFMs for %d TFs (will be zeroed). First few: %s", len(missing), missing[:10])
        return {tf: pfm_map.get(tf) for tf in tf_names}

    def _init_moods(self) -> None:
        try:
            import MOODS.parsers  # type: ignore
            import MOODS.scan  # type: ignore
            import MOODS.tools  # type: ignore
        except Exception as exc:  # pragma: no cover - import error is runtime dependent
            raise ImportError("MOODS Python package is required for TFBS scanning") from exc

        matrices = []
        tf_indices = []
        for tf_idx, tf_name in enumerate(self.tf_names):
            p = self.pfm_files.get(tf_name)
            if p is None:
                continue
            if self.log_base is not None:
                mat = MOODS.parsers.pfm_to_log_odds(str(p), self.lo_bg, self.pseudocount, self.log_base)
            else:
                mat = MOODS.parsers.pfm_to_log_odds(str(p), self.lo_bg, self.pseudocount)
            matrices.append(mat)
            tf_indices.append(tf_idx)
        if any(len(m) == 0 for m in matrices):
            raise ValueError("Failed to parse one or more PFM files into log-odds matrices")
        matrices_rc = [MOODS.tools.reverse_complement(m, 4) for m in matrices]
        matrices_all = matrices + matrices_rc
        thresholds = [float(self.threshold)] * len(matrices_all)

        scanner = MOODS.scan.Scanner(self.window_size)
        scanner.set_motifs(matrices_all, self.bg, thresholds)

        self._moods_bg = self.bg
        self._moods_scanner = scanner
        self._moods_n_motifs = len(matrices)
        self._moods_tf_indices = tf_indices

    def __call__(
        self,
        dna_seq: Union[str, np.ndarray, torch.Tensor],
        tf_expression: Union[np.ndarray, torch.Tensor, List[float]],
    ) -> torch.Tensor:
        dna_onehot, dna_string = self._normalize_dna_input(dna_seq)
        tf_expression_vec = self._normalize_tf_expression(tf_expression)

        tfbs_sites = self._compute_tfbs_sites(dna_string)

        dna_chunks, tf_feats = self._prepare_epigept_inputs(dna_onehot, tf_expression_vec, tfbs_sites)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds = self.model(dna_chunks, tf_feats)
            else:
                preds = self.model(dna_chunks, tf_feats)

        # preds: (1, 1000, NUM_SIGNALS) -> (1000, NUM_SIGNALS)
        preds = preds.to(device="cpu", dtype=torch.float32)
        return preds.squeeze(0)

    @staticmethod
    def _normalize_dna_input(dna_seq: Union[str, np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, str]:
        if isinstance(dna_seq, str):
            seq = dna_seq.strip().upper()
            if len(seq) != SEQ_LENGTH:
                raise ValueError(f"DNA string must be length {SEQ_LENGTH}, got {len(seq)}")
            if any(base not in "ACGTN" for base in seq):
                raise ValueError("DNA string must contain only A/C/G/T/N")
            onehot = EpigeptInferenceWrapper._dna_string_to_onehot(seq)
            return onehot, seq
        if isinstance(dna_seq, torch.Tensor):
            arr = dna_seq.detach().cpu().numpy()
        else:
            arr = np.asarray(dna_seq)

        if arr.shape != (SEQ_LENGTH, 4):
            raise ValueError(f"DNA one-hot must have shape ({SEQ_LENGTH}, 4); got {arr.shape}")
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("DNA one-hot must contain only 0/1 values")
        row_sums = arr.sum(axis=1)
        if not np.all((row_sums == 1) | (row_sums == 0)):
            raise ValueError("DNA one-hot must have exactly one 1 per position (or all zeros for N)")
        onehot = torch.from_numpy(arr.astype(np.float32))
        seq = EpigeptInferenceWrapper._onehot_to_dna_string(arr)
        return onehot, seq

    @staticmethod
    def _normalize_tf_expression(tf_expression: Union[np.ndarray, torch.Tensor, List[float]]) -> torch.Tensor:
        if isinstance(tf_expression, torch.Tensor):
            vec = tf_expression.detach().cpu().numpy()
        else:
            vec = np.asarray(tf_expression)
        if vec.shape != (epigept_config.TF_DIM,):
            raise ValueError(f"TF expression must have shape ({epigept_config.TF_DIM},); got {vec.shape}")
        return torch.from_numpy(vec.astype(np.float32))

    @staticmethod
    def _dna_string_to_onehot(seq: str) -> torch.Tensor:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        onehot = np.zeros((SEQ_LENGTH, 4), dtype=np.float32)
        for i, base in enumerate(seq):
            if base == "N":
                continue
            onehot[i, mapping[base]] = 1.0
        return torch.from_numpy(onehot)

    @staticmethod
    def _onehot_to_dna_string(arr: np.ndarray) -> str:
        mapping = np.array(["A", "C", "G", "T"])
        row_sums = arr.sum(axis=1)
        idx = arr.argmax(axis=1)
        seq = mapping[idx]
        seq[row_sums == 0] = "N"
        return "".join(seq)

    def _prepare_epigept_inputs(
        self,
        dna_onehot: torch.Tensor,
        tf_expression: torch.Tensor,
        tfbs_sites: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # DNA: single 128 kb chunk
        dna_seq = dna_onehot.unsqueeze(0).to(device=self.device, dtype=torch.float32)  # (1, 128000, 4)
        dna_chunks = dna_seq.permute(0, 2, 1)  # (1, 4, 128000)

        # TF binding: 1000 bins of 128 bp
        tf_binding = torch.from_numpy(tfbs_sites.astype(np.float32)).to(device=self.device)
        tf_binding = tf_binding.unsqueeze(0)  # (1, 1000, 711)

        tf_expr = tf_expression.to(device=self.device).unsqueeze(0).unsqueeze(1)  # (1, 1, 711)
        tf_feats = tf_binding * tf_expr
        return dna_chunks, tf_feats

    def _compute_tfbs_sites(self, dna_seq: str) -> np.ndarray:
        if len(dna_seq) != SEQ_LENGTH:
            raise ValueError(f"DNA string must be length {SEQ_LENGTH}, got {len(dna_seq)}")
        results = self._moods_scanner.scan(dna_seq)
        tfbs = np.zeros((TFBINS, epigept_config.TF_DIM), dtype=np.float16)
        n_motifs = self._moods_n_motifs

        for local_idx in range(n_motifs):
            tf_idx = self._moods_tf_indices[local_idx]
            for matches in (results[local_idx], results[local_idx + n_motifs]):
                for m in matches:
                    bin_idx = int(m.pos) // BINSIZE
                    if bin_idx < 0 or bin_idx >= TFBINS:
                        continue
                    score = np.float16(np.round(float(m.score), 3))
                    if score > tfbs[bin_idx, tf_idx]:
                        tfbs[bin_idx, tf_idx] = score

        return tfbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EpiGePT inference wrapper for single sequence inputs.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EpiGePT .ckpt")
    parser.add_argument("--pfm-dir", type=str, default=str(EPIGEPT_ROOT / "motif_pfms"))
    parser.add_argument("--tf-csv", type=str, default=str(EPIGEPT_ROOT / "tf_expression.csv"))
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--pseudocount", type=float, default=0.01)
    parser.add_argument("--bg", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25])
    parser.add_argument("--lo-bg", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25])
    parser.add_argument("--log-base", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    _ = EpigeptInferenceWrapper(
        checkpoint_path=args.checkpoint,
        pfm_dir=args.pfm_dir,
        tf_csv=args.tf_csv,
        threshold=args.threshold,
        window_size=args.window_size,
        pseudocount=args.pseudocount,
        bg=args.bg,
        lo_bg=args.lo_bg,
        log_base=args.log_base,
    )
    logging.info("Wrapper initialized. Import and call EpigeptInferenceWrapper in Python for inference.")
