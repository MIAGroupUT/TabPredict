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
import wandb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from src.data.loaders import (
    build_loaders,
    build_data_list,
    compute_clin_stats,
    get_val_runtime_transforms,
    load_splits_and_config,
    RuntimeDataset,
)
from src.models.reintervention import ReinterventionRiskModel, get_model
from src.training.losses import MultiTaskReinterventionLoss, ReinterventionLoss
from src.training.stages import (
    build_optimizer,
    maybe_apply_modality_dropout,
    print_model_param_overview,
    set_all_seeds,
    set_stage_a,
    set_stage_b,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── validation ────────────────────────────────────────────────────────────────

def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    complications: list[str],
    size: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 8,
) -> tuple[float, dict]:
    """Evaluate on val split; return (avg_loss, metrics_dict)."""
    model.eval()
    with_comp = bool(complications)
    val_loss  = 0.0
    all_reint_probs, all_reint_gt, all_reint_lv = [], [], []
    all_comp_probs, all_comp_gt = [], []

    with torch.no_grad():
        for batch in val_loader:
            cta  = batch.get("cta", torch.zeros(batch_size, 4, *size)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa  = batch["aaa_metrics"].to(DEVICE).float()
            reint_gt = batch["reintervention_label"].to(DEVICE).float()

            outputs = model(cta, clin, aaa)
            reint_logit, reint_logvar = outputs[0], outputs[1]

            if with_comp:
                comp_gt      = batch["complication_labels"].to(DEVICE).float()
                comp_logits, comp_logvar = outputs[2], outputs[3]
                loss, _      = criterion(reint_logit, reint_logvar,
                                         comp_logits, comp_logvar,
                                         reint_gt, comp_gt)
                all_comp_probs.append(torch.sigmoid(comp_logits).cpu())
                all_comp_gt.append(comp_gt.cpu())
            else:
                loss, _ = criterion(reint_logit, reint_logvar, reint_gt)

            val_loss += loss.item()
            all_reint_probs.append(torch.sigmoid(reint_logit).cpu())
            all_reint_gt.append(reint_gt.cpu())
            all_reint_lv.append(reint_logvar.cpu())

    reint_p  = torch.cat(all_reint_probs).numpy().flatten()
    reint_g  = torch.cat(all_reint_gt).numpy().flatten().astype(int)
    reint_lv = torch.cat(all_reint_lv).numpy().flatten()

    # Optimal F1 threshold via PR curve
    p_c, r_c, t_c = precision_recall_curve(reint_g, reint_p)
    f1s          = 2 * p_c[:-1] * r_c[:-1] / (p_c[:-1] + r_c[:-1] + 1e-8)
    best_idx     = int(np.argmax(f1s))
    best_thresh  = float(t_c[best_idx])
    reint_pred   = (reint_p > best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(reint_g, reint_pred, labels=[0, 1]).ravel()

    metrics = {
        "val/loss_total":        val_loss / len(val_loader),
        "val/reint_auc":         float(roc_auc_score(reint_g, reint_p)) if len(np.unique(reint_g)) > 1 else 0.5,
        "val/reint_f1":          float(f1_score(reint_g, reint_pred)),
        "val/reint_precision":   float(precision_score(reint_g, reint_pred, zero_division=0)),
        "val/reint_recall":      float(recall_score(reint_g, reint_pred, zero_division=0)),
        "val/reint_accuracy":    float(accuracy_score(reint_g, reint_pred)),
        "val/reint_best_f1":     float(f1s[best_idx]),
        "val/reint_best_thresh": best_thresh,
        "val/mean_uncertainty":  float(reint_lv.mean()),
        "val/reint_TP": int(tp), "val/reint_FP": int(fp),
        "val/reint_TN": int(tn), "val/reint_FN": int(fn),
        "val/reint_TPR": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "val/reint_TNR": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "val/reint_FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "val/reint_FNR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }

    if with_comp:
        comp_p = torch.cat(all_comp_probs).numpy()
        comp_g = torch.cat(all_comp_gt).numpy()
        aucs   = []
        for i, name in enumerate(complications):
            valid = comp_g[:, i] != -1
            if valid.sum() > 0 and len(np.unique(comp_g[valid, i])) > 1:
                auc = float(roc_auc_score(comp_g[valid, i], comp_p[valid, i]))
                metrics[f"val/auc_{name}"] = auc
                aucs.append(auc)
        if aucs:
            metrics["val/comp_mean_auc"] = float(np.mean(aucs))

    wandb.log(metrics)

    print(
        f"[VAL] loss={metrics['val/loss_total']:.4f} | "
        f"auc={metrics['val/reint_auc']:.4f} | "
        f"acc={metrics['val/reint_accuracy']:.4f} | "
        f"f1={metrics['val/reint_f1']:.4f} | "
        f"prec={metrics['val/reint_precision']:.4f} | "
        f"rec={metrics['val/reint_recall']:.4f} | "
        f"thresh={metrics['val/reint_best_thresh']:.3f} | "
        f"unc={metrics['val/mean_uncertainty']:.4f}"
        + (f" | comp_mAUC={metrics.get('val/comp_mean_auc', 0.0):.4f}" if with_comp else "")
    )
    return val_loss / len(val_loader), metrics


def log_val_table(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    stage: str,
    thresh: float,
    complications: list[str],
    size: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 8,
) -> None:
    """Log a W&B table with per-case predictions (reintervention + complications)."""
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

            comp_probs = torch.sigmoid(outputs[2]).cpu().numpy() if with_comp else None
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

    wandb.log({f"eval/full_matrix_stage_{stage}": table, "epoch": epoch})
    print(f"Logged val table: {case_counter} cases.")


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


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict) -> None:
    """
    Full training pipeline driven by a YAML config dict.

    cfg keys (see experiments/*.yaml for the canonical schema):
        name, model, training, paths, wandb
    """
    set_all_seeds(cfg["training"].get("seed", 3299))

    base_dir      = Path(cfg["paths"]["base_dir"])
    cache_dir     = Path(cfg["paths"]["cache_dir"])
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    cache_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    complications = list(cfg["model"].get("complications") or [])
    with_comp     = bool(complications)
    batch_size    = cfg["training"].get("batch_size", 8)
    size          = tuple(cfg["training"].get("image_size", [128, 128, 128]))
    spacing       = tuple(cfg["training"].get("voxel_spacing", [1.5, 1.5, 1.5]))
    num_workers   = cfg["training"].get("num_workers", 16)

    # ── data ─────────────────────────────────────────────────────────────────
    splits, config, feature_names = load_splits_and_config(base_dir)
    clin_mean, clin_std = compute_clin_stats(splits["train"], config, feature_names)
    print(f"Clinical features : {len(feature_names)}")

    train_loader, val_loader = build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=base_dir, cache_dir=cache_dir,
        batch_size=batch_size, num_workers=num_workers,
        complications=complications,
        train_cache_rate=cfg["training"].get("train_cache_rate", 0.5),
        val_cache_rate=cfg["training"].get("val_cache_rate", 1.0),
        size=size, spacing=spacing,
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"].get("run_name", cfg["name"]),
        config={
            "clinical_dim":      len(feature_names),
            "num_complications": len(complications),
            "batch_size":        batch_size,
            "lora_r":            cfg["model"].get("lora_r", 8),
            "with_complications": with_comp,
        },
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = ReinterventionRiskModel(
        clinical_dim=len(feature_names),
        aaa_dim=cfg["model"].get("aaa_dim", 5),
        complications=complications,
        lora_r=cfg["model"].get("lora_r", 8),
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ── loss ──────────────────────────────────────────────────────────────────
    if with_comp:
        criterion = MultiTaskReinterventionLoss(
            focal_gamma=cfg["training"].get("focal_gamma", 2.0),
            label_smoothing=cfg["training"].get("label_smoothing", 0.05),
            device=DEVICE,
        )
    else:
        criterion = ReinterventionLoss(
            focal_gamma=cfg["training"].get("focal_gamma", 2.0),
            label_smoothing=cfg["training"].get("label_smoothing", 0.05),
            device=DEVICE,
        )

    dropout_cfg = cfg["training"].get("modality_dropout", {})

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
            set_stage_b(model)
            if with_comp and isinstance(criterion, MultiTaskReinterventionLoss):
                criterion.comp_weight = cfg["training"].get("comp_weight_stage_b", 0.3)

        opt = build_optimizer(get_model(model), st)
        T_0 = stage_cfg.get("T_0", 20 if st == "A" else 30)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=2, eta_min=1e-7)
        scaler    = torch.cuda.amp.GradScaler()

        print_model_param_overview(model)
        model = model.to(DEVICE).to(memory_format=torch.channels_last_3d)

        best_val_auc     = 0.0
        patience_counter = 0
        gate_l1_w        = cfg["training"].get("gate_l1_weight", 1e-3)
        apply_dropout    = (st != "A")

        for epoch in range(stage_cfg["epochs"]):
            model.train()
            train_loss, last_loss_dict = 0.0, {}

            pbar = tqdm(train_loader, desc=f"Stage {st} Epoch {epoch}")
            for i, batch in enumerate(pbar):
                clin = batch["clin"].to(DEVICE).float()
                aaa  = batch["aaa_metrics"].to(DEVICE).float()
                cta  = batch.get("cta", torch.zeros(batch_size, 4, *size)).to(DEVICE).float()

                if apply_dropout:
                    cta, clin, aaa = maybe_apply_modality_dropout(
                        model.training, cta, clin, aaa,
                        p_cta=dropout_cfg.get("p_cta", 0.1),
                        p_aaa_given_cta=dropout_cfg.get("p_aaa_given_cta", 0.15),
                        p_aaa_given_clin=dropout_cfg.get("p_aaa_given_clin", 0.15),
                    )

                reint_gt = batch["reintervention_label"].to(DEVICE).float()

                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs      = model(cta=cta, clin=clin, aaa=aaa)
                    reint_logit, reint_logvar = outputs[0], outputs[1]

                    if with_comp:
                        comp_gt   = batch["complication_labels"].to(DEVICE).float()
                        loss, last_loss_dict = criterion(
                            reint_logit, reint_logvar,
                            outputs[2], outputs[3],
                            reint_gt, comp_gt,
                        )
                    else:
                        loss, last_loss_dict = criterion(reint_logit, reint_logvar, reint_gt)

                    # L1 on feature gate → sparse feature selection
                    loss = loss + gate_l1_w * torch.sigmoid(get_model(model).feature_gate).sum()

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{train_loss/(i+1):.3f}")

            val_loss, val_metrics = validate(
                model, val_loader, criterion, complications, size, batch_size
            )
            _log_detailed_metrics(st, epoch, train_loss / len(train_loader),
                                   val_loss, last_loss_dict, opt)
            scheduler.step()
            tqdm.write(
                f"[Stage {st} | Epoch {epoch:03d}]  "
                f"train={train_loss/len(train_loader):.4f}  val={val_loss:.4f}  "
                f"auc={val_metrics.get('val/reint_auc', 0):.4f}"
            )

            if (epoch + 1) % 3 == 0:
                log_val_table(
                    model, val_loader, epoch, st,
                    val_metrics.get("val/reint_best_thresh", 0.5),
                    complications, size, batch_size,
                )

            val_auc = val_metrics.get("val/reint_auc", 0.5)
            if val_auc > best_val_auc:
                best_val_auc     = val_auc
                patience_counter = 0
                torch.save(
                    {
                        "state_dict":    get_model(model).state_dict(),
                        "threshold":     val_metrics.get("val/reint_best_thresh", 0.5),
                        "feature_names": feature_names,
                        "clin_mean":     clin_mean.tolist(),
                        "clin_std":      clin_std.tolist(),
                        "complications": complications,
                        "config":        cfg,
                    },
                    checkpoint_dir / f"best_stage_{st.lower()}.pt",
                )
                print(f"  ✓ Checkpoint saved (AUC={best_val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= stage_cfg.get("patience", 8):
                    print(f"Early stopping at epoch {epoch} (best AUC={best_val_auc:.4f})")
                    break

    wandb.finish()
