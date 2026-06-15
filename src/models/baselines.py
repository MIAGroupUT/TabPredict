"""
Classical ML tabular baselines: Logistic Regression, Random Forest,
Gradient Boosting, and (optionally) XGBoost.

Two public classes
──────────────────
TabularBaseline
    Standalone sklearn/XGBoost classifier.  Fit on any numpy feature matrix and
    call predict_proba.  Used by train_baselines.py for tabular-only comparison.

ClassicalBranchWrapper  (nn.Module)
    Integrates a TabularBaseline into ReinterventionRiskModel as the
    clinical_branch, replacing MLP or TabM.  A learnable Linear(1 → 32)
    projection makes it compatible with the FiLM fusion and allows gradients
    to flow into the rest of the network.  The sklearn model itself is
    non-differentiable; its output is treated as a fixed external feature.

Usage — standalone
──────────────────
baseline = TabularBaseline("rf")
baseline.fit(X_train, y_train)
probs = baseline.predict_proba(X_val)   # (N,)
baseline.save("rf.pkl")
baseline = TabularBaseline.load("rf.pkl")

Usage — integrated (called automatically by trainer.py)
────────────────────────────────────────────────────────
wrapper = ClassicalBranchWrapper(in_features, "rf")
wrapper.fit(X_train_normalised, y_train)   # fit before training starts
# forward(x_tensor) → (B, 32)  — called from ReinterventionRiskModel.forward()
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier as _XGBClassifier
    from xgboost import DMatrix as _XGBDMatrix
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    _XGBDMatrix = None

AVAILABLE: list[str] = ["lr", "rf", "gbm", "xgboost"]

# Sklearn-only classical backbone keys (TabPFN lives in tabpfn.py)
CLASSICAL_BACKBONES: list[str] = AVAILABLE

_DEFAULTS: dict[str, dict] = {
    "lr": dict(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
    ),
    "rf": dict(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gbm": dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    ),
    "xgboost": dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}


def _build_estimator(model_type: str, **overrides):
    if model_type not in _DEFAULTS:
        raise ValueError(f"Unknown model_type {model_type!r}. Choose from {AVAILABLE}.")
    if model_type == "xgboost" and not _XGB_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Install it with:  pip install xgboost"
        )
    cfg = {**_DEFAULTS[model_type], **overrides}
    cls = {
        "lr":      LogisticRegression,
        "rf":      RandomForestClassifier,
        "gbm":     GradientBoostingClassifier,
        "xgboost": _XGBClassifier if _XGB_AVAILABLE else None,
    }[model_type]
    return cls(**cfg)


class TabularBaseline:
    """Tabular-only classical ML classifier.

    Wraps sklearn/XGBoost with a StandardScaler prefix and median imputation
    for NaN values (missing clinical measurements).

    Args:
        model_type: one of ``"lr"``, ``"rf"``, ``"gbm"``, ``"xgboost"``
        **kwargs:   override any default hyperparameter for the chosen estimator
    """

    def __init__(self, model_type: str, **kwargs) -> None:
        self.model_type = model_type
        self.pipeline   = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    _build_estimator(model_type, **kwargs)),
        ])
        self._col_medians: np.ndarray | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularBaseline":
        """Fit the baseline on training data.

        Median imputation statistics are derived from X (training set only).
        For XGBoost, scale_pos_weight is set automatically from class counts.
        """
        X = self._impute(X, fit=True)
        if self.model_type == "xgboost":
            n_neg = int((y == 0).sum())
            n_pos = int((y == 1).sum())
            self.pipeline.named_steps["clf"].set_params(
                scale_pos_weight=n_neg / max(n_pos, 1)
            )
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probability for each sample, shape (N,)."""
        return self.pipeline.predict_proba(self._impute(X, fit=False))[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TabularBaseline":
        return joblib.load(path)

    # ── imputation ────────────────────────────────────────────────────────────

    def _impute(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = X.copy().astype(np.float32)
        # XGBoost learns the optimal default direction for NaN at each split —
        # don't overwrite with a fixed value and lose that signal.
        if self.model_type == "xgboost":
            return X
        if fit:
            col_medians = np.nanmedian(X, axis=0)
            self._col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        if self._col_medians is None:
            raise RuntimeError("Call fit() before predict_proba().")
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_idx = np.where(nan_mask.any(axis=0))[0]
            for j in col_idx:
                X[nan_mask[:, j], j] = self._col_medians[j]
        return X


# ── integrated backbone wrapper ───────────────────────────────────────────────

class ClassicalBranchWrapper(nn.Module):
    """Drop-in replacement for MLP/TabM as the clinical_branch in
    ReinterventionRiskModel.

    The sklearn/XGBoost model is non-differentiable; gradients flow only
    through the learnable ``projection`` layer.  The projection input width
    varies by model type to preserve richer structural information:

    - ``lr``:      scalar positive-class probability → Linear(1→32)
    - ``rf``:      per-tree positive-class probability (n_estimators,) → Linear(n_est→32)
    - ``gbm``:     per-tree normalised leaf index (n_estimators,) → Linear(n_est→32)
    - ``xgboost``: TreeSHAP values per input feature (in_features,) → Linear(n_feat→32)

    The sklearn model is fit *once* before Stage A using z-normalised clinical
    features + raw AAA scalars (same distribution as runtime inference).

    Args:
        in_features:      clinical_dim + aaa_dim (total tabular input width)
        model_type:       one of CLASSICAL_BACKBONES
        use_tree_output:  False (default) → Linear(1→32) from single calibrated
                          probability. True → richer per-tree projection (see
                          class docstring above). Set via backbone_cfg in YAML.
        **kwargs:         hyperparameter overrides forwarded to TabularBaseline
    """

    _is_non_differentiable: bool = True   # read by is_classical_backbone()

    def __init__(self, in_features: int, model_type: str, use_tree_output: bool = False, **kwargs) -> None:
        super().__init__()
        self.model_type      = model_type
        self.in_features     = in_features
        self.use_tree_output = use_tree_output
        self.baseline        = TabularBaseline(model_type, **kwargs)

        if use_tree_output:
            merged  = {**_DEFAULTS.get(model_type, {}), **kwargs}
            n_est   = int(merged.get('n_estimators', 300))
            proj_in = {'lr': 1, 'rf': n_est, 'gbm': n_est, 'xgboost': in_features}.get(model_type, 1)
        else:
            proj_in = 1  # single calibrated probability for all backends

        self.projection  = nn.Linear(proj_in, 32)
        self._fitted: bool = False

    # ── sklearn fit ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying TabularBaseline on pre-normalised tabular data."""
        self.baseline.fit(X, y)
        if self.model_type == 'gbm':
            # Pre-compute per-tree leaf-index scale; stored on baseline so
            # joblib serialise() includes it automatically.
            X_proc = self._preprocess(X)
            clf    = self.baseline.pipeline.named_steps['clf']
            leaves = clf.apply(X_proc).squeeze(-1).astype(np.float32)  # (N, n_est)
            self.baseline._gbm_leaf_scale = leaves.max(axis=0).clip(min=1.0)
        self._fitted = True

    # ── feature extraction ────────────────────────────────────────────────────

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Impute + StandardScaler — mirrors the sklearn pipeline prefix."""
        X = self.baseline._impute(X, fit=False)
        scaler = self.baseline.pipeline.named_steps.get('scaler')
        if scaler is not None:
            X = scaler.transform(X)
        return X

    def _extract_features(self, x_np: np.ndarray) -> np.ndarray:
        """Return features fed into the projection layer.

        use_tree_output=False (default): single calibrated probability (B, 1).
        use_tree_output=True: richer per-tree representation — see class docstring.
        """
        if not self.use_tree_output:
            return self.baseline.predict_proba(x_np)[:, np.newaxis].astype(np.float32)

        if self.model_type == 'lr':
            return self.baseline.predict_proba(x_np)[:, np.newaxis].astype(np.float32)

        x_proc = self._preprocess(x_np)
        clf    = self.baseline.pipeline.named_steps['clf']

        if self.model_type == 'rf':
            return np.column_stack([
                tree.predict_proba(x_proc)[:, 1] for tree in clf.estimators_
            ]).astype(np.float32)

        if self.model_type == 'gbm':
            leaves = clf.apply(x_proc).squeeze(-1).astype(np.float32)
            return leaves / self.baseline._gbm_leaf_scale

        if self.model_type == 'xgboost':
            dm   = _XGBDMatrix(x_proc)
            shap = clf.get_booster().predict(dm, pred_contribs=True).astype(np.float32)
            return shap[:, :-1]

        return self.baseline.predict_proba(x_np)[:, np.newaxis].astype(np.float32)

    # ── nn.Module forward ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            return torch.zeros(x.shape[0], 32, device=x.device, dtype=x.dtype)
        x_np   = x.detach().cpu().float().numpy()
        feat   = self._extract_features(x_np)
        feat_t = torch.as_tensor(feat, dtype=x.dtype, device=x.device)
        return self.projection(feat_t)

    # ── checkpoint serialisation ──────────────────────────────────────────────

    def serialize(self) -> bytes:
        """Serialise the fitted TabularBaseline to bytes for checkpoint storage."""
        buf = io.BytesIO()
        joblib.dump(self.baseline, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize(data: bytes) -> "TabularBaseline":
        """Restore a TabularBaseline from checkpoint bytes."""
        return joblib.load(io.BytesIO(data))
