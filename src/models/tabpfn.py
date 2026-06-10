"""
TabPFNBranchWrapper — TabPFN pretrained tabular backbone for ReinterventionRiskModel.

TabPFN (Prior-Labs, Nature 2024) is a transformer pretrained on millions of synthetic
tabular datasets via meta-learning.  At inference it performs in-context learning:
the training set is the context; prediction is a single forward pass.

Fine-tuning strategy (two phases):
  Phase 1 — pre-Stage A: FinetunedTabPFNClassifier meta-learning fine-tune at lr=1e-5.
    Adapts the full transformer to EVAR task structure via a meta-learning objective
    (context + query splits within the training set).  Early stopping on an internal
    10 % validation split (patience=5, ~20 epochs max).

  Phase 2 — Stage A + B: Last ``n_trainable_layers`` transformer blocks remain
    trainable; all other TabPFN weights are frozen.  Gradients flow through the
    unfrozen blocks and the learnable projection layer.
    The L1 feature gate is also trained because the backbone is differentiable
    (_is_non_differentiable = False).

Differentiable forward:
  TabPFNClassifier exposes ``differentiable_input=True`` + ``fit_with_differentiable_input``
  which sets up an InferenceEngineCachePreprocessing with inference_mode=False.
  Calling ``forward(..., use_inference_mode=True)`` on such a classifier sets
  ``actual_inference_mode = use_inference_mode and not differentiable_input = False``,
  meaning torch.inference_mode() is NOT entered — gradients are preserved through
  the unfrozen layers.

Projection:
  forward() returns per-estimator raw logits shape (n_estimators, B, n_classes).
  These are permuted to (B, n_estimators * n_classes) and fed into a learnable
  Linear(n_estimators * n_classes → out_dim).

  With n_estimators_final_inference=4 and binary classification:
    projection input = 4 × 2 = 8
  Each estimator uses a different random preprocessing seed, so the 8 values
  carry genuine diversity across the ensemble.

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

    Satisfies the branch contract ``(B, in_features) → (B, out_dim)``.

    Phase 1: FinetunedTabPFNClassifier meta-learning fine-tune adapts the
    pretrained transformer to the EVAR task before Stage A begins.

    Phase 2: The last ``n_trainable_layers`` encoder/decoder blocks stay
    trainable during Stage A and B; the rest are frozen.  Gradients flow
    through these blocks and the learnable ``projection`` layer.

    ``_manages_own_grad = True`` signals stages.py to not override the
    per-layer requires_grad flags set by ``_setup_diff_mode()``.

    Args:
        in_features:                 clinical_dim + aaa_dim (total tabular width).
        out_dim:                     output embedding size (32 to match FiLM fusion).
        epochs:                      fine-tuning epochs for phase 1.
        learning_rate:               AdamW lr for phase 1 fine-tuning (1e-5 preserves priors).
        early_stopping_patience:     epochs without val improvement before stopping.
        n_estimators_final_inference: TabPFN ensemble size at inference.
        n_trainable_layers:          last N encoder/decoder blocks to keep trainable
                                     during Stage A + B (default 2).  Set 0 for
                                     projection-only training.
        random_state:                reproducibility seed.
        device:                      'cuda' / 'cpu'; auto-detected when None.
        model_dir:                   path to directory with tabpfn-v2.5-classifier*.ckpt
                                     files; uses HuggingFace cache when None.
    """

    _is_non_differentiable: bool = False  # differentiable → feature gate trained
    _manages_own_grad: bool = True        # stages.py must not clobber per-layer grad flags

    def __init__(
        self,
        in_features: int,
        out_dim: int = 32,
        epochs: int = 20,
        learning_rate: float = 1e-5,
        early_stopping_patience: int = 5,
        n_estimators_final_inference: int = 4,
        n_trainable_layers: int = 2,
        random_state: int = 42,
        device: str | None = None,
        model_dir: str | None = None,
    ) -> None:
        super().__init__()
        if not _TABPFN_AVAILABLE:
            raise ImportError("tabpfn is not installed. Run: pip install tabpfn")

        self.in_features = in_features
        self.n_estimators = n_estimators_final_inference
        self.n_trainable_layers = n_trainable_layers
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
        self._diff_clf = None  # set by _setup_diff_mode() after fit()

    # ── fit (called once by fit_classical_backbone before Stage A) ────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Phase 1 meta-learning fine-tune, then set up differentiable inference."""
        self._clf.fit(X, y)
        self._setup_diff_mode()
        self._fitted = True

    def _setup_diff_mode(self) -> None:
        """Configure differentiable inference from fitted _clf.

        Sets differentiable_input=True on the inference classifier, re-initialises
        its executor with inference_mode=False (preserves autograd), then freezes all
        TabPFN model params except the last n_trainable_layers encoder/decoder blocks.
        """
        diff_clf = self._clf.finetuned_inference_classifier_
        diff_clf.differentiable_input = True

        X_t = torch.as_tensor(self._clf.X_, dtype=torch.float32)
        y_t = torch.as_tensor(self._clf.y_, dtype=torch.long)
        diff_clf.fit_with_differentiable_input(X_t, y_t)

        # Freeze everything, then selectively unfreeze last N blocks
        for m in diff_clf.models_:
            for p in m.parameters():
                p.requires_grad_(False)
            if self.n_trainable_layers > 0:
                # Use decoder if present (it produces the test-sample logits),
                # else fall back to the encoder's last layers
                stack = getattr(m, "transformer_decoder", None) or m.transformer_encoder
                for layer in stack.layers[-self.n_trainable_layers:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)

        self._diff_clf = diff_clf

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

        # differentiable_input=True means actual_inference_mode = False inside
        # TabPFNClassifier.forward(), so torch.inference_mode() is never entered
        # and gradients flow through the unfrozen last N layers.
        raw = self._diff_clf.forward(
            x, use_inference_mode=True, return_raw_logits=True
        )

        # raw: (n_estimators, B, n_classes) → (B, n_estimators * n_classes)
        feat = raw.permute(1, 0, 2).reshape(x.shape[0], -1)
        return self.projection(feat)

    # ── checkpoint serialisation ──────────────────────────────────────────────

    def serialize(self) -> bytes:
        """Serialise the fitted FinetunedTabPFNClassifier to bytes."""
        buf = io.BytesIO()
        joblib.dump(self._clf, buf)
        return buf.getvalue()

    def restore_from_bytes(self, data: bytes) -> None:
        """Restore fitted _clf and re-establish differentiable mode.

        Called by load_checkpoint() in reintervention.py after loading a
        checkpoint that contains 'classical_baseline_bytes'.
        """
        self._clf = joblib.load(io.BytesIO(data))
        self._setup_diff_mode()
        self._fitted = True
