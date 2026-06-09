"""
TabPFNBranchWrapper — TabPFN pretrained tabular backbone for ReinterventionRiskModel.

TabPFN (Prior-Labs, Nature 2024) is a transformer pretrained on millions of synthetic
tabular datasets via meta-learning.  At inference it performs in-context learning:
the training set is the context; prediction is a single forward pass.  No gradient
flows through the backbone — it is fitted once before Stage A and then frozen.

Pretrained weights: downloaded automatically to ~/.cache/tabpfn/ from
HuggingFace (Prior-Labs/TabPFN-v2-clf) on first use.  Set HTTPS_PROXY if behind
a corporate proxy (e.g. export HTTPS_PROXY=http://proxy.utwente.nl:3128).

Fine-tuning strategy (built-in, no LoRA needed):
  FinetunedTabPFNClassifier does full weight fine-tuning at lr=1e-5 — very
  conservative, preserves pretrained in-context priors.  Uses a meta-learning
  objective: splits training samples into context + query per step and trains
  the transformer to predict query labels from context examples.
  Early stopping on an internal 10 % validation split (patience=5, ~20 epochs max).

Projection:
  predict_raw_logits() returns per-estimator unnormalised logits with shape
  (n_estimators, n_samples, n_classes).  These are transposed to
  (n_samples, n_estimators * n_classes) and fed into a learnable
  Linear(n_estimators * n_classes → out_dim) projection.

  With n_estimators_final_inference=4 and binary classification:
    projection input = 4 × 2 = 8  (vs. 1 for a scalar probability)
  Each estimator used a different random preprocessing seed, so the 8 values
  carry genuine diversity — differences in logit magnitude and sign across
  estimators are meaningful signal for the downstream fusion head.

Contract
--------
    fit(X_np, y_np)                  # called once by fit_classical_backbone()
    forward(x: Tensor[B, in_features]) → Tensor[B, out_dim]
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

__all__ = ["TabPFNBranchWrapper"]

try:
    from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False

_N_CLASSES = 2   # binary classification throughout this project


class TabPFNBranchWrapper(nn.Module):
    """Drop-in clinical branch using TabPFN as the tabular encoder.

    Satisfies the branch contract ``(B, in_features) → (B, out_dim)``.  The
    TabPFN model is non-differentiable; gradients only flow through the learnable
    ``projection`` layer (Linear(n_estimators*2 → out_dim)), allowing the FiLM
    fusion and prediction head to adapt while the pretrained transformer stays frozen.

    Setting ``_is_non_differentiable = True`` causes ``is_classical_backbone()``
    to return True, which skips the L1 feature gate and its param group in the
    optimizer — consistent with the sklearn classical backends.

    Args:
        in_features:                 clinical_dim + aaa_dim (total tabular width).
        out_dim:                     output embedding size (32 to match FiLM fusion).
        epochs:                      fine-tuning epochs (early stopping may trigger first).
        learning_rate:               AdamW lr for fine-tuning (1e-5 preserves priors).
        early_stopping_patience:     epochs without val improvement before stopping.
        n_estimators_final_inference: TabPFN ensemble size at inference.
                                     Projection input = n_estimators * 2 (both logits
                                     per estimator before softmax / averaging).
        random_state:                reproducibility seed.
        device:                      'cuda' / 'cpu'; auto-detected when None.
    """

    _is_non_differentiable: bool = True   # read by is_classical_backbone()

    def __init__(
        self,
        in_features: int,
        out_dim: int = 32,
        epochs: int = 20,
        learning_rate: float = 1e-5,
        early_stopping_patience: int = 5,
        n_estimators_final_inference: int = 4,
        random_state: int = 42,
        device: str | None = None,
        model_dir: str | None = None,
    ) -> None:
        super().__init__()
        if not _TABPFN_AVAILABLE:
            raise ImportError("tabpfn is not installed. Run: pip install tabpfn")

        self.in_features = in_features
        self.n_estimators = n_estimators_final_inference
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        extra: dict = {}
        if model_dir is not None:
            ckpts = sorted(Path(model_dir).glob("tabpfn-v2.5-classifier*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(
                    f"No TabPFN v2.5 classifier checkpoints found in {model_dir}"
                )
            # Pass a single checkpoint: multi-model mode breaks FinetunedTabPFNClassifier
            # (finetuned_base._fit calls .model_ which raises with multiple models).
            # Fine-tuning tunes one model; n_estimators_final_inference ensembles via
            # different preprocessing seeds on that single fine-tuned model.
            # Prefer the canonical default (matches TabPFN's own default_filename);
            # sorted() puts _default-2 before _default alphabetically, so ckpts[0]
            # would be wrong.
            default_ckpt = next(
                (p for p in ckpts if p.name.endswith("_default.ckpt")), ckpts[0]
            )
            extra = {"model_path": str(default_ckpt)}

        self._clf = FinetunedTabPFNClassifier(
            device=_device,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping=True,
            early_stopping_patience=early_stopping_patience,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_final_inference=n_estimators_final_inference,
            use_fixed_preprocessing_seed=False,
            random_state=random_state,
            eval_metric="roc_auc",
            extra_classifier_kwargs=extra or None,
        )
        # n_estimators * n_classes raw logits before any averaging or softmax
        self.projection = nn.Linear(n_estimators_final_inference * _N_CLASSES, out_dim)
        self._fitted: bool = False

    # ── fit (called once by fit_classical_backbone before Stage A) ────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fine-tune TabPFN on z-normalised clinical + raw AAA features."""
        self._clf.fit(X, y)
        self._fitted = True

    # ── nn.Module forward ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, in_features)`` normalised tabular features.
        Returns:
            ``(B, out_dim)`` projection of per-estimator raw logits.
        """
        if not self._fitted:
            return torch.zeros(x.shape[0], self.projection.out_features,
                               device=x.device, dtype=x.dtype)

        x_np = x.detach().cpu().float().numpy()

        # (E, B, C) — raw unnormalised logits, one row per estimator
        raw = self._clf.finetuned_inference_classifier_.predict_raw_logits(x_np)

        # (B, E*C) — flatten estimator and class axes into a single feature vector
        feat = raw.transpose(1, 0, 2).reshape(len(x_np), -1)
        feat_t = torch.as_tensor(feat, dtype=x.dtype, device=x.device)

        return self.projection(feat_t)   # (B, out_dim)

    # ── checkpoint serialisation ──────────────────────────────────────────────

    def serialize(self) -> bytes:
        """Serialise the fitted FinetunedTabPFNClassifier to bytes."""
        buf = io.BytesIO()
        joblib.dump(self._clf, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize(data: bytes) -> "FinetunedTabPFNClassifier":
        return joblib.load(io.BytesIO(data))
