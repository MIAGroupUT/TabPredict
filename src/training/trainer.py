"""
Training loop and validation for both model variants.

Entry point: run_pipeline(cfg)
    cfg is a dict loaded from an experiments/*.yaml file.

The function handles both variants transparently:
    - with_complications=False → ReinterventionLoss, 2-output forward
    - with_complications=True  → MultiTaskReinterventionLoss, 4-output forward
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from src.evaluation.metrics import plot_roc_curve
from src.data.loaders import (
    build_loaders,
    build_data_list,
    compute_clin_stats,
    filter_features_by_missingness,
    get_val_runtime_transforms,
    load_splits_and_config,
    RuntimeDataset,
)
from src.models.reintervention import ReinterventionRiskModel, get_model, is_classical_backbone
from src.training.losses import MultiTaskReinterventionLoss, ReinterventionLoss
from src.training.stages import (
    build_optimizer,
    maybe_apply_modality_dropout,
    print_model_param_overview,
    set_all_seeds,
    set_stage_a,
    set_stage_b,
)
from src.xai.explainer import ReinterventionXAI

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── validation ────────────────────────────────────────────────────────────────

def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    complications: list[str],
    size: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 8,
    recall_beta: float = 1.0,
    disable_cta: bool = False,
    split: str = "val",
) -> tuple[float, dict]:
    """Evaluate on a data split; return (avg_loss, metrics_dict).

    split controls the metric key prefix ("val" → "val/...", "train" → "train/...").
    """
    model.eval()
    with_comp  = bool(complications)
    total_loss = 0.0
    all_reint_probs, all_reint_gt = [], []
    all_comp_probs, all_comp_gt   = [], []

    with torch.no_grad():
        for batch in val_loader:
            cta  = batch.get("cta", torch.zeros(batch_size, 4, *size)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa  = batch["aaa_metrics"].to(DEVICE).float()
            reint_gt = batch["reintervention_label"].to(DEVICE).float()

            if disable_cta:
                cta = torch.zeros_like(cta)
            outputs = model(cta, clin, aaa)
            reint_logit = outputs[0]

            if with_comp:
                comp_gt     = batch["complication_labels"].to(DEVICE).float()
                comp_logits = outputs[1]
                loss, _     = criterion(reint_logit, comp_logits, reint_gt, comp_gt)
                all_comp_probs.append(torch.sigmoid(comp_logits).cpu())
                all_comp_gt.append(comp_gt.cpu())
            else:
                loss, _ = criterion(reint_logit, reint_gt)

            total_loss += loss.item()
            all_reint_probs.append(torch.sigmoid(reint_logit).cpu())
            all_reint_gt.append(reint_gt.cpu())

    reint_p = torch.cat(all_reint_probs).numpy().flatten()
    reint_g = torch.cat(all_reint_gt).numpy().flatten().astype(int)

    # Optimal F-beta threshold via PR curve (beta>1 weights recall over precision)
    p_c, r_c, t_c = precision_recall_curve(reint_g, reint_p)
    b2           = recall_beta ** 2
    f_beta       = (1 + b2) * p_c[:-1] * r_c[:-1] / (b2 * p_c[:-1] + r_c[:-1] + 1e-8)
    best_idx     = int(np.argmax(f_beta))
    best_thresh  = float(t_c[best_idx])
    reint_pred   = (reint_p > best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(reint_g, reint_pred, labels=[0, 1]).ravel()
    s = split

    metrics = {
        f"{s}/loss_total":        total_loss / len(val_loader),
        f"{s}/reint_auc":         float(roc_auc_score(reint_g, reint_p)) if len(np.unique(reint_g)) > 1 else 0.5,
        f"{s}/reint_f1":          float(f1_score(reint_g, reint_pred)),
        f"{s}/reint_precision":   float(precision_score(reint_g, reint_pred, zero_division=0)),
        f"{s}/reint_recall":      float(recall_score(reint_g, reint_pred, zero_division=0)),
        f"{s}/reint_accuracy":    float(accuracy_score(reint_g, reint_pred)),
        f"{s}/reint_best_fbeta":  float(f_beta[best_idx]),
        f"{s}/reint_best_thresh": best_thresh,
        f"{s}/recall_beta":       recall_beta,
        f"{s}/reint_TP": int(tp), f"{s}/reint_FP": int(fp),
        f"{s}/reint_TN": int(tn), f"{s}/reint_FN": int(fn),
        f"{s}/reint_TPR": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        f"{s}/reint_TNR": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        f"{s}/reint_FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        f"{s}/reint_FNR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }

    if with_comp:
        comp_p = torch.cat(all_comp_probs).numpy()
        comp_g = torch.cat(all_comp_gt).numpy()
        aucs   = []
        for i, name in enumerate(complications):
            valid = comp_g[:, i] != -1
            if valid.sum() > 0 and len(np.unique(comp_g[valid, i])) > 1:
                auc = float(roc_auc_score(comp_g[valid, i], comp_p[valid, i]))
                metrics[f"{s}/auc_{name}"] = auc
                aucs.append(auc)
        if aucs:
            metrics[f"{s}/comp_mean_auc"] = float(np.mean(aucs))

    wandb.log(metrics)

    print(
        f"[{split.upper()}] loss={metrics[f'{s}/loss_total']:.4f} | "
        f"auc={metrics[f'{s}/reint_auc']:.4f} | "
        f"acc={metrics[f'{s}/reint_accuracy']:.4f} | "
        f"f1={metrics[f'{s}/reint_f1']:.4f} | "
        f"prec={metrics[f'{s}/reint_precision']:.4f} | "
        f"rec={metrics[f'{s}/reint_recall']:.4f} | "
        f"thresh={metrics[f'{s}/reint_best_thresh']:.3f}"
        + (f" | comp_mAUC={metrics.get(f'{s}/comp_mean_auc', 0.0):.4f}" if with_comp else "")
    )
    return total_loss / len(val_loader), metrics, (reint_p, reint_g)


def log_val_table(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    stage: str,
    thresh: float,
    complications: list[str],
    size: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 8,
    split: str = "val",
) -> None:
    """Log a W&B table with per-case predictions (reintervention + complications).

    split distinguishes val vs train tables in W&B ("val" or "train").
    """
    model.eval()
    with_comp = bool(complications)
    columns = ["Epoch", "Case_ID", "Reint_Prob", "Reint_Pred", "Reint_GT", "Label_Type"]
    if with_comp:
        for c in complications:
            columns += [f"P_{c}", f"GT_{c}"]
    table = wandb.Table(columns=columns)

    def _ltype(pred, gt):
        return {(1, 1): "TP", (1, 0): "FP", (0, 0): "TN"}.get((pred, gt), "FN")

    case_counter = 0
    with torch.no_grad():
        for batch in val_loader:
            cta  = batch.get("cta", torch.zeros(batch_size, 4, *size)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa  = batch["aaa_metrics"].to(DEVICE).float()

            outputs  = model(cta=cta, clin=clin, aaa=aaa)
            probs    = torch.sigmoid(outputs[0]).cpu().numpy().flatten()
            reint_gt = batch["reintervention_label"].cpu().numpy().flatten()

            comp_probs = torch.sigmoid(outputs[1]).cpu().numpy() if with_comp else None
            comp_gt    = batch["complication_labels"].cpu().numpy() if with_comp else None

            for i in range(len(probs)):
                pred = int(probs[i] > thresh)
                gt   = int(reint_gt[i])
                row  = [epoch, case_counter, round(float(probs[i]), 4),
                        pred, gt, _ltype(pred, gt)]
                if with_comp:
                    for j in range(len(complications)):
                        row += [round(float(comp_probs[i, j]), 4), int(comp_gt[i, j])]
                table.add_data(*row)
                case_counter += 1

    wandb.log({f"eval/{split}_full_matrix_stage_{stage}": table, "epoch": epoch})
    print(f"Logged {split} table: {case_counter} cases.")


def log_xai_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    stage: str,
    feature_names: list[str],
    complications: list[str],
    n_samples: int = 4,
    use_gradcam: bool = False,
    size: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 8,
) -> None:
    """Run XAI on up to n_samples val patients and log a W&B table + optional GradCAM."""
    xai = ReinterventionXAI(get_model(model), feature_names, complications)
    with_comp = bool(complications)

    columns = [
        "epoch", "stage", "case_idx",
        "reint_prob", "confidence_level", "primary_risk_factor", "causal_chain",
        "imp_CTA", "imp_Clinical", "imp_AAA",
    ]
    if with_comp:
        columns += [f"comp_{c}" for c in complications]

    table = wandb.Table(columns=columns)
    wandb_images: dict = {}

    model.eval()
    count = 0
    for batch in val_loader:
        if count >= n_samples:
            break
        cta  = batch.get("cta", torch.zeros(batch_size, 4, *size)).to(DEVICE).float()
        clin = batch["clin"].to(DEVICE).float()
        aaa  = batch["aaa_metrics"].to(DEVICE).float()

        for i in range(cta.shape[0]):
            if count >= n_samples:
                break
            c_cta, c_clin, c_aaa = cta[i:i+1], clin[i:i+1], aaa[i:i+1]
            xr = xai.full_explain(c_cta, c_clin, c_aaa)
            mod = xr.get("modality_importance", {})

            row = [
                epoch, stage, count,
                xr["reintervention_prob"], xr["confidence_level"],
                xr["primary_risk_factor"], xr["causal_chain"],
                mod.get("CTA", 0.0), mod.get("Clinical", 0.0), mod.get("AAA_metrics", 0.0),
            ]
            if with_comp:
                row += [xr.get("complication_probs", {}).get(c, 0.0) for c in complications]
            table.add_data(*row)

            if use_gradcam:
                cam_np = xai.gradcam(c_cta, c_clin, c_aaa)
                if cam_np is not None:
                    for k, v in xai.get_gradcam_wandb_images(c_cta, cam_np).items():
                        wandb_images[f"xai/{k}_s{count}"] = v
            count += 1

    wandb.log({f"xai/table_stage_{stage}": table, "epoch": epoch, **wandb_images})
    print(f"Logged XAI table: {count} samples (epoch {epoch}, stage {stage}).")


def _log_detailed_metrics(
    stage: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    loss_dict: dict,
    opt: torch.optim.Optimizer,
) -> None:
    log_data = {
        f"stage_{stage}/val_loss":   val_loss,
        f"stage_{stage}/train_loss": train_loss,
        "epoch": epoch,
    }
    for k, v in loss_dict.items():
        log_data[f"metrics/{k}"] = v
    for group in opt.param_groups:
        log_data[f"lr/{group.get('name', 'group')}"] = group["lr"]
    wandb.log(log_data)


# ── temperature calibration ───────────────────────────────────────────────────

def calibrate_temperature(
    model: nn.Module,
    val_loader,
    ckpt_path: Path,
    recall_beta: float,
    size: tuple,
    batch_size: int,
    disable_cta: bool = False,
) -> tuple[float, float]:
    """Fit a single temperature scalar T on the val set (post-hoc calibration).

    Minimises NLL of sigmoid(logit / T) via LBFGS, then re-searches the
    F-beta-optimal threshold on calibrated probabilities and writes both back
    into the checkpoint at ckpt_path.

    Returns (temperature, calibrated_threshold).
    """
    from src.models.reintervention import load_checkpoint

    load_checkpoint(get_model(model), ckpt_path, device=DEVICE)
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            cta  = batch.get("cta", torch.zeros(batch_size, 4, *size)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa  = batch["aaa_metrics"].to(DEVICE).float()
            gt   = batch["reintervention_label"].to(DEVICE).float()
            if disable_cta:
                cta = torch.zeros_like(cta)
            logit = model(cta, clin, aaa)[0]
            all_logits.append(logit.cpu())
            all_labels.append(gt.cpu())

    logits = torch.cat(all_logits).float().flatten()
    labels = torch.cat(all_labels).float().flatten()

    T_param = nn.Parameter(torch.ones(1))
    opt_T   = torch.optim.LBFGS([T_param], lr=0.01, max_iter=200, tolerance_grad=1e-7)

    def closure():
        opt_T.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            logits / T_param.clamp(min=0.01), labels
        )
        loss.backward()
        return loss

    opt_T.step(closure)
    T = float(T_param.clamp(min=0.01).item())

    # Re-search F-beta threshold on calibrated probabilities
    probs  = torch.sigmoid(logits / T).detach().numpy()
    labels_np = labels.numpy().astype(int)
    p_c, r_c, t_c = precision_recall_curve(labels_np, probs)
    b2     = recall_beta ** 2
    f_beta = (1 + b2) * p_c[:-1] * r_c[:-1] / (b2 * p_c[:-1] + r_c[:-1] + 1e-8)
    thresh = float(t_c[int(np.argmax(f_beta))])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["temperature"] = T
    ckpt["threshold"]   = thresh
    torch.save(ckpt, ckpt_path)

    return T, thresh


# ── classical backbone helpers ────────────────────────────────────────────────

def fit_classical_backbone(
    model: nn.Module,
    train_ids: list[str],
    config: dict,
    feature_names: list[str],
    clin_mean: np.ndarray,
    clin_std: np.ndarray,
) -> None:
    """Fit the sklearn model inside ClassicalBranchWrapper on the training split.

    Features are z-normalised (clinical) and left raw (AAA) to exactly match
    what the DataLoader delivers to the model at runtime.
    """
    from src.data.loaders import build_tabular_arrays
    from src.models.reintervention import is_classical_backbone

    if not is_classical_backbone(model):
        return

    branch = get_model(model).clinical_branch

    X_raw, y, _ = build_tabular_arrays(train_ids, config, feature_names)

    # Z-score clinical features; leave NaNs intact so TabularBaseline._impute()
    # can apply median imputation, and XGBoost can use its native NaN handling.
    n_clin = len(feature_names)
    X_clin = (X_raw[:, :n_clin] - clin_mean) / (clin_std + 1e-8)
    X_aaa  = X_raw[:, n_clin:]                      # raw AAA scalars (no z-norm)
    X      = np.concatenate([X_clin, X_aaa], axis=1)

    n_pos  = int(y.sum())
    name   = getattr(branch, "model_type", type(branch).__name__).upper()
    print(f"  Fitting {name} on {len(y)} train samples (pos={n_pos}, neg={len(y)-n_pos})")
    branch.fit(X, y)
    print(f"  Classical backbone fitted and ready.")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict) -> None:
    """
    Full training pipeline driven by a YAML config dict.

    cfg keys (see experiments/*.yaml for the canonical schema):
        name, model, training, paths, wandb
    """
    seed = cfg["training"].get("seed", 3299)
    set_all_seeds(seed)

    base_dir      = Path(cfg["paths"]["base_dir"])
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    complications = list(cfg["model"].get("complications") or [])
    with_comp     = bool(complications)
    batch_size    = cfg["training"].get("batch_size", 8)
    size          = tuple(cfg["training"].get("image_size", [128, 128, 128]))
    spacing       = tuple(cfg["training"].get("voxel_spacing", [1.5, 1.5, 1.5]))
    num_workers   = cfg["training"].get("num_workers", 16)

    # ── data ─────────────────────────────────────────────────────────────────
    splits, config, feature_names = load_splits_and_config(base_dir)
    max_missing_pct = cfg["training"].get("max_missing_pct", 1.0)
    feature_names   = filter_features_by_missingness(
        splits["train"], config, feature_names, max_missing_pct
    )
    clin_mean, clin_std, clin_median = compute_clin_stats(splits["train"], config, feature_names)
    clin_fill = ((clin_median - clin_mean) / clin_std).astype(np.float32)
    print(f"Clinical features : {len(feature_names)}  (max_missing_pct={max_missing_pct})")

    train_loader, val_loader = build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=base_dir,
        batch_size=batch_size, num_workers=num_workers,
        complications=complications,
        train_cache_rate=cfg["training"].get("train_cache_rate", 0.5),
        val_cache_rate=cfg["training"].get("val_cache_rate", 1.0),
        size=size, spacing=spacing, seed=seed,
        clin_fill=clin_fill,
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"].get("run_name", cfg["name"]),
        mode=cfg["wandb"].get("mode", "online"),
        config={
            "clinical_dim":      len(feature_names),
            "num_complications": len(complications),
            "batch_size":        batch_size,
            "with_complications": with_comp,
        },
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = ReinterventionRiskModel(
        clinical_dim=len(feature_names),
        aaa_dim=cfg["model"].get("aaa_dim", 5),
        complications=complications,
        clinical_backbone=cfg["model"].get("clinical_backbone", "mlp"),
        tabm_cfg=cfg["model"].get("backbone_cfg") or cfg["model"].get("tabm") or {},
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ── loss ──────────────────────────────────────────────────────────────────
    _focal_gamma  = cfg["training"].get("focal_gamma", 2.0)
    _label_smooth = cfg["training"].get("label_smoothing", 0.02)
    _pos_weight   = cfg["training"].get("pos_weight", 1.0)
    if with_comp:
        _clw_cfg = cfg["training"].get("comp_loss_weights", {})
        _cpw_cfg = cfg["training"].get("comp_pos_weights",  {})
        comp_loss_weights = [_clw_cfg.get(c, 1.0) for c in complications] if _clw_cfg else None
        comp_pos_weights  = [_cpw_cfg.get(c, 1.0) for c in complications] if _cpw_cfg else None
        criterion = MultiTaskReinterventionLoss(
            focal_gamma=_focal_gamma,
            label_smoothing=_label_smooth,
            pos_weight_reint=_pos_weight,
            comp_loss_weights=comp_loss_weights,
            comp_pos_weights=comp_pos_weights,
            device=DEVICE,
        )
    else:
        criterion = ReinterventionLoss(
            focal_gamma=_focal_gamma,
            label_smoothing=_label_smooth,
            pos_weight=_pos_weight,
            device=DEVICE,
        )

    dropout_cfg  = cfg["training"].get("modality_dropout", {})
    xai_every    = cfg["training"].get("xai_every_n_epochs", 10)
    xai_samples  = cfg["training"].get("xai_samples", 4)
    xai_gradcam  = cfg["training"].get("xai_gradcam", False)
    recall_beta  = cfg["training"].get("recall_beta", 1.0)
    disable_cta  = cfg["training"].get("disable_cta", False)

    # ── classical backbone: fit sklearn model once before any stage ───────────
    if is_classical_backbone(model):
        print("\n>>> FITTING CLASSICAL BACKBONE")
        fit_classical_backbone(
            model, splits["train"], config, feature_names, clin_mean, clin_std
        )

    # ── stage loop ────────────────────────────────────────────────────────────
    for stage_cfg in cfg["training"]["stages"]:
        st = stage_cfg["name"]
        print(f"\n>>> STARTING STAGE {st}")

        if st == "A":
            set_stage_a(model)
            if with_comp and isinstance(criterion, MultiTaskReinterventionLoss):
                criterion.comp_weight = cfg["training"].get("comp_weight_stage_a", 0.1)
        elif st == "B":
            prev_ckpt = checkpoint_dir / "best_stage_a.pt"
            if prev_ckpt.exists():
                from src.models.reintervention import load_checkpoint
                load_checkpoint(get_model(model), prev_ckpt, device=DEVICE)
                print(f"Loaded Stage A checkpoint: {prev_ckpt}")
            freeze_clinical_b = cfg["training"].get("freeze_clinical_in_stage_b", False)
            set_stage_b(model, freeze_clinical=freeze_clinical_b)
            if with_comp and isinstance(criterion, MultiTaskReinterventionLoss):
                criterion.comp_weight = cfg["training"].get("comp_weight_stage_b", 0.3)

        opt = build_optimizer(get_model(model), st,
                              freeze_clinical=cfg["training"].get("freeze_clinical_in_stage_b", False))
        T_0 = stage_cfg.get("T_0", 20 if st == "A" else 30)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=2, eta_min=1e-7)
        scaler    = torch.cuda.amp.GradScaler()

        print_model_param_overview(model)
        model = model.to(DEVICE).to(memory_format=torch.channels_last_3d)

        best_val_fbeta   = 0.0
        patience_counter = 0
        gate_l1_w        = cfg["training"].get("gate_l1_weight", 1e-3)
        apply_dropout    = True  # both stages; Stage A needs it to prevent clinical memorisation

        for epoch in range(stage_cfg["epochs"]):
            model.train()
            train_loss, last_loss_dict = 0.0, {}

            pbar = tqdm(train_loader, desc=f"Stage {st} Epoch {epoch}")
            for i, batch in enumerate(pbar):
                clin = batch["clin"].to(DEVICE).float()
                aaa  = batch["aaa_metrics"].to(DEVICE).float()
                cta  = batch.get("cta", torch.zeros(batch_size, 4, *size)).to(DEVICE).float()

                if disable_cta:
                    cta = torch.zeros_like(cta)
                if apply_dropout:
                    cta, clin, aaa = maybe_apply_modality_dropout(
                        model.training, cta, clin, aaa,
                        p_cta=dropout_cfg.get("p_cta", 0.1),
                        p_aaa_given_cta=dropout_cfg.get("p_aaa_given_cta", 0.15),
                        p_aaa_given_clin=dropout_cfg.get("p_aaa_given_clin", 0.15),
                        p_clinical=dropout_cfg.get("p_clinical", 0.0),
                    )

                reint_gt = batch["reintervention_label"].to(DEVICE).float()

                opt.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs     = model(cta=cta, clin=clin, aaa=aaa)
                    reint_logit = outputs[0]

                    if with_comp:
                        comp_gt = batch["complication_labels"].to(DEVICE).float()
                        loss, last_loss_dict = criterion(
                            reint_logit, outputs[1], reint_gt, comp_gt,
                        )
                    else:
                        loss, last_loss_dict = criterion(reint_logit, reint_gt)

                    # L1 on feature gate → sparse feature selection
                    # (skipped for classical backbones: gate has no gradient path)
                    if not is_classical_backbone(model):
                        loss = loss + gate_l1_w * torch.sigmoid(get_model(model).feature_gate).sum()

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{train_loss/(i+1):.3f}")

            val_loss, val_metrics, _ = validate(
                model, val_loader, criterion, complications, size, batch_size,
                recall_beta=recall_beta,
                disable_cta=disable_cta,
                split="val",
            )
            _, train_metrics, _ = validate(
                model, train_loader, criterion, complications, size, batch_size,
                recall_beta=recall_beta,
                disable_cta=disable_cta,
                split="train",
            )
            _log_detailed_metrics(st, epoch, train_loss / len(train_loader),
                                   val_loss, last_loss_dict, opt)
            scheduler.step()
            tqdm.write(
                f"[Stage {st} | Epoch {epoch:03d}]  "
                f"train_loss={train_loss/len(train_loader):.4f}  val_loss={val_loss:.4f}  "
                f"train_auc={train_metrics.get('train/reint_auc', 0):.4f}  "
                f"val_auc={val_metrics.get('val/reint_auc', 0):.4f}  "
                f"val_f{recall_beta:.0f}={val_metrics.get('val/reint_best_fbeta', 0):.4f}"
            )

            if (epoch + 1) % 3 == 0:
                log_val_table(
                    model, val_loader, epoch, st,
                    val_metrics.get("val/reint_best_thresh", 0.5),
                    complications, size, batch_size, split="val",
                )
                log_val_table(
                    model, train_loader, epoch, st,
                    train_metrics.get("train/reint_best_thresh", 0.5),
                    complications, size, batch_size, split="train",
                )

            if (epoch + 1) % xai_every == 0:
                log_xai_epoch(
                    model, val_loader, epoch, st,
                    feature_names, complications,
                    n_samples=xai_samples,
                    use_gradcam=xai_gradcam,
                    size=size, batch_size=batch_size,
                )

            # Save by F-beta (recall-weighted) rather than AUC: in a safety-critical
            # setting catching all positives (recall) matters more than discrimination.
            val_fbeta = val_metrics.get("val/reint_best_fbeta", 0.0)
            if val_fbeta > best_val_fbeta:
                best_val_fbeta   = val_fbeta
                patience_counter = 0
                ckpt_dict = {
                    "state_dict":    get_model(model).state_dict(),
                    "threshold":     val_metrics.get("val/reint_best_thresh", 0.5),
                    "feature_names": feature_names,
                    "clin_mean":     clin_mean.tolist(),
                    "clin_std":      clin_std.tolist(),
                    "complications":      complications,
                    "clinical_backbone":  cfg["model"].get("clinical_backbone", "mlp"),
                    "tabm_cfg":           cfg["model"].get("backbone_cfg") or cfg["model"].get("tabm") or {},
                    "config":             cfg,
                }
                if is_classical_backbone(model):
                    from src.models.baselines import ClassicalBranchWrapper
                    ckpt_dict["classical_baseline_bytes"] = (
                        get_model(model).clinical_branch.serialize()
                    )
                torch.save(ckpt_dict, checkpoint_dir / f"best_stage_{st.lower()}.pt")
                print(
                    f"  ✓ Checkpoint saved "
                    f"(F{recall_beta:.0f}={best_val_fbeta:.4f}, "
                    f"AUC={val_metrics.get('val/reint_auc', 0):.4f}, "
                    f"rec={val_metrics.get('val/reint_recall', 0):.4f}, "
                    f"thresh={val_metrics.get('val/reint_best_thresh', 0):.3f}@F{recall_beta:.0f})"
                )
            else:
                patience_counter += 1
                if patience_counter >= stage_cfg.get("patience", 8):
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best F{recall_beta:.0f}={best_val_fbeta:.4f})"
                    )
                    break

        # ── ROC curves for this stage (best checkpoint, val + train) ─────────
        import matplotlib.pyplot as plt
        from src.models.reintervention import load_checkpoint as _lc
        best_ckpt = checkpoint_dir / f"best_stage_{st.lower()}.pt"
        if best_ckpt.exists():
            _lc(get_model(model), best_ckpt, device=DEVICE)
            _, roc_metrics, (rp, rg) = validate(
                model, val_loader, criterion, complications, size, batch_size,
                recall_beta=recall_beta, disable_cta=disable_cta, split="val",
            )
            thresh = float(roc_metrics.get("val/reint_best_thresh", 0.5))
            fig = plot_roc_curve(
                rp.tolist(), rg.tolist(),
                title=f"Stage {st} — Val ROC (best checkpoint)",
                threshold=thresh,
            )
            wandb.log({f"roc/val_stage_{st}": wandb.Image(fig)})
            plt.close(fig)
            print(f"  Val ROC curve logged to W&B (roc/val_stage_{st})")

            _, train_roc_metrics, (trp, trg) = validate(
                model, train_loader, criterion, complications, size, batch_size,
                recall_beta=recall_beta, disable_cta=disable_cta, split="train",
            )
            train_thresh = float(train_roc_metrics.get("train/reint_best_thresh", 0.5))
            fig_train = plot_roc_curve(
                trp.tolist(), trg.tolist(),
                title=f"Stage {st} — Train ROC (best checkpoint)",
                threshold=train_thresh,
            )
            wandb.log({f"roc/train_stage_{st}": wandb.Image(fig_train)})
            plt.close(fig_train)
            print(f"  Train ROC curve logged to W&B (roc/train_stage_{st})")

    # ── temperature calibration (post-hoc, after all stages) ─────────────────
    final_ckpt = checkpoint_dir / "best_stage_b.pt"
    if final_ckpt.exists():
        print("\n>>> TEMPERATURE CALIBRATION")
        T, cal_thresh = calibrate_temperature(
            model, val_loader, final_ckpt,
            recall_beta=recall_beta,
            size=size, batch_size=batch_size,
            disable_cta=disable_cta,
        )
        print(f"  T={T:.4f}  calibrated threshold={cal_thresh:.4f}")
        wandb.log({"calibration/temperature": T, "calibration/threshold": cal_thresh})

    wandb.finish()
