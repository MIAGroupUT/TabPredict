"""
AAA REINTERVENTION RISK PREDICTION WITH COMPLICATIONS
======================================================
Multi-task model predicting:
  1. Reintervention likelihood + confidence (primary task)
  2. Per-complication likelihood + confidence for all 11 complication types
  3. Causal explanation: which complications are driving the reintervention risk

Two-stage training curriculum:
  Stage A — Frozen Backbone (Fusion Warm-Up):
    Merlin/LoRA fully frozen. Fusion, clinical, AAA branches, and both heads train
    at moderate LR. Establishes stable multimodal representations quickly without
    noisy LoRA gradients destabilising the fusion.

  Stage B — LoRA Fine-Tuning (Backbone Adaptation):
    LoRA adapters unfrozen at very low LR (5e-6). Everything else continues at
    low LR. Adapts visual backbone to the AAA task now that fusion is stable.
"""

import json
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, precision_recall_curve, confusion_matrix,
)
import wandb
import torch.nn.functional as F
from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, Resized, MapTransform, ToTensord,
    RandGaussianNoised, RandBiasFieldd, RandScaleIntensityd,
)
from dataclasses import dataclass
from typing import Dict, List, Tuple

from peft import LoraConfig, get_peft_model
from merlin import Merlin

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for server
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR  = Path("/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1")
CACHE_DIR = Path("/deepstore/datasets/mia/AI4EVAR/derivatives/monai_cache_comp")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
SIZE       = (128, 128, 128)
BATCH_SIZE = 8


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_all_seeds(3299)


# --- COMPLICATIONS LIST ---
COMPLICATIONS = [
    # Endoleaks (granular)
    "endoleak_1a", "endoleak_1b", "endoleak_2", "endoleak_3", "endoleak_5",
    # Patency & mechanical
    # "occlusion", "stenosis", "kinking", "fracture",
    # Endograft specific
    "migration", "infection",
]
NUM_COMPLICATIONS = len(COMPLICATIONS)

AAA_METRIC_NAMES = [
    "volume_lumen_ml",
    "volume_ilt_ml",
    "volume_total_ml",
    "max_lumen_to_wall_dist_mm",
    "avg_lumen_to_wall_dist_mm",
]

# --- OUTPUT STRUCTURE ---
@dataclass
class ReinterventionRiskPrediction:
    """Structured output with full per-complication uncertainty breakdown."""

    # Reintervention prediction
    reintervention_prob: float        # 0–1 probability
    reintervention_confidence: float  # 0–1 score (higher = more certain)
    confidence_level: str             # "HIGH" / "MED" / "LOW"

    # Per-complication predictions
    complication_probs: Dict[str, float]   # {name: probability}
    complication_confidence: Dict[str, str]  # {name: "HIGH"/"MED"/"LOW"}

    # XAI
    clinical_importance: List[Tuple[str, float, str]]  # [(feature, grad_score, level)]
    aaa_importance: List[Tuple[str, float, str]]       # [(metric, grad_score, level)]
    modality_importance: Dict[str, float]              # {modality: relative importance}
    causal_chain: str
    primary_risk_factor: str
    gradcam_images: Dict[str, object]                  # {view: wandb.Image}

    def format_report(self) -> str:
        lines = [
            "╔════════════════════════════════════════════════════════════════╗",
            "║     REINTERVENTION RISK + COMPLICATIONS PREDICTION REPORT      ║",
            "╚════════════════════════════════════════════════════════════════╝",
            f"\nREINTERVENTION RISK:",
            f"  Probability : {self.reintervention_prob:.1%}",
            f"  Confidence  : {self.confidence_level} ({self.reintervention_confidence:.2f})",
            f"\nCOMPLICATION BREAKDOWN (sorted by probability):",
        ]
        for comp, prob in sorted(self.complication_probs.items(), key=lambda x: x[1], reverse=True):
            conf = self.complication_confidence.get(comp, "?")
            lines.append(f"  • {comp:<20}  {prob:.1%}  [{conf} confidence]")
        lines += [
            f"\nTOP CLINICAL FACTORS:",
        ]
        for feat, score, level in self.clinical_importance[:5]:
            lines.append(f"  • {feat}: {level} ({score:.3f})")
        lines += [
            f"\nAAA MORPHOLOGY IMPORTANCE:",
        ]
        for metric, score, level in self.aaa_importance:
            lines.append(f"  • {metric}: {level} ({score:.3f})")
        lines += [
            f"\nMODALITY IMPORTANCE:",
        ]
        for mod, imp in sorted(self.modality_importance.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  • {mod}: {imp:.1%}")
        lines += [
            f"\nCAUSAL CHAIN:",
            f"  {self.causal_chain}",
            f"\nPRIMARY RISK FACTOR:  {self.primary_risk_factor}",
        ]
        return "\n".join(lines)


# --- AUGMENTATION ---
def causal_modality_dropout(cta, clin, aaa, p_cta=0.1, p_aaa_given_cta=0.2, p_aaa_given_clin=0.2):
    bsz = cta.size(0)
    cta_mask = torch.rand(bsz, device=cta.device) > p_cta
    cta_d, clin_d, aaa_d = cta.clone(), clin.clone(), aaa.clone()
    for b in range(bsz):
        if not cta_mask[b]:
            cta_d[b] = 0
            continue
        if random.random() < p_aaa_given_cta or random.random() < p_aaa_given_clin:
            aaa_d[b] = 0
    return cta_d, clin_d, aaa_d


def maybe_apply_modality_dropout(training, cta, clin, aaa, p_cta, p_aaa_given_cta, p_aaa_given_clin):
    if not training:
        return cta, clin, aaa
    return causal_modality_dropout(cta, clin, aaa, p_cta, p_aaa_given_cta, p_aaa_given_clin)


# --- DATA LOADING ---
def build_data_list(ids, base_dir, config, feature_keys):
    """Build dataset with reintervention AND per-complication labels."""
    data = []
    for pid in ids:
        pdata    = config.get(pid, {})
        clin_dict = pdata.get("pre_op_features", {})
        clin = np.nan_to_num(
            np.array([clin_dict.get(k, 0.0) for k in feature_keys], dtype=np.float32)
        )

        metrics_path = base_dir / "metricsTr" / f"{pid}.json"
        aaa_metrics  = [0.0] * 5
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
                aaa_metrics = [
                    m.get("volume_lumen_ml", 0.0),
                    m.get("volume_ilt_ml", 0.0),
                    m.get("volume_total_ml", 0.0),
                    m.get("max_lumen_to_wall_dist_mm", 0.0),
                    m.get("avg_lumen_to_wall_dist_mm", 0.0),
                ]

        gt          = pdata.get("ground_truth", {})
        reint_label = float(gt.get("reintervention", 0))

        # Per-complication labels: -1 = missing/unknown, 0 = absent, 1 = present
        comp_labels = np.array(
            [float(gt.get(comp, -1.0)) for comp in COMPLICATIONS],
            dtype=np.float32,
        )

        data.append({
            "ct":  str(base_dir / "imagesTr"   / f"{pid}.nii.gz"),
            "lbl": str(base_dir / "labelsTr"   / f"{pid}.nii.gz"),
            "dst": str(base_dir / "distMapsTr" / f"{pid}.nii.gz"),
            "clin":                clin,
            "aaa_metrics":         aaa_metrics,
            "reintervention_label": np.array([reint_label], dtype=np.float32),
            "complication_labels":  comp_labels,
            "pid":                 pid,
        })
    return data


class BuildCTAd(MapTransform):
    def __call__(self, data):
        ct, lbl, dst = data["ct"], data["lbl"], data["dst"]
        data["cta"] = torch.cat([ct, (lbl == 1).float(), (lbl == 2).float(), dst], dim=0)
        return data


class NormalizeClinicald(MapTransform):
    def __init__(self, keys, mean, std, eps=1e-8):
        super().__init__(keys)
        self.mean = torch.tensor(mean)
        self.std  = torch.tensor(std)
        self.eps  = eps

    def __call__(self, data):
        clin = torch.as_tensor(data["clin"]).float()
        data["clin"] = (clin - self.mean) / (self.std + self.eps)
        return data


def get_cached_transforms():
    """Expensive deterministic I/O + geometry — cached in RAM once."""
    return Compose([
        LoadImaged(keys=["ct", "lbl", "dst"]),
        EnsureChannelFirstd(keys=["ct", "lbl", "dst"]),
        Orientationd(keys=["ct", "lbl", "dst"], axcodes="RAS"),
        Spacingd(
            keys=["ct", "lbl", "dst"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest", "bilinear"),
        ),
        ScaleIntensityRanged(keys=["ct"], a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        Resized(
            keys=["ct", "lbl", "dst"],
            spatial_size=SIZE,
            mode=("trilinear", "nearest", "trilinear"),
        ),
    ])


def get_train_runtime_transforms(clin_mean, clin_std):
    """Applied fresh every iteration: augment ct → build CTA → tensorise."""
    return Compose([
        RandGaussianNoised(keys=["ct"], prob=0.3, mean=0.0, std=0.02),
        RandBiasFieldd(keys=["ct"], prob=0.2),
        RandScaleIntensityd(keys=["ct"], factors=0.1, prob=0.5),
        BuildCTAd(keys=["ct", "lbl", "dst"]),
        ToTensord(keys=["ct", "lbl", "dst", "aaa_metrics", "reintervention_label", "complication_labels"]),
        NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
    ])


def get_val_runtime_transforms(clin_mean, clin_std):
    """Applied fresh every iteration: no augmentation — deterministic val metrics."""
    return Compose([
        BuildCTAd(keys=["ct", "lbl", "dst"]),
        ToTensord(keys=["ct", "lbl", "dst", "aaa_metrics", "reintervention_label", "complication_labels"]),
        NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
    ])


class RuntimeDataset(torch.utils.data.Dataset):
    """Wraps a cached dataset and applies a runtime transform on every access."""
    def __init__(self, base_ds, runtime_transform):
        self.base_ds   = base_ds
        self.runtime   = runtime_transform
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        return self.runtime(self.base_ds[idx])


# --- MODEL ---
def disable_checkpointing(model):
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
        if hasattr(m, "enable_gradient_checkpointing"):
            try:
                m.enable_gradient_checkpointing(False)
            except Exception:
                pass


def binary_entropy_from_logits(logits, eps=1e-8):
    p   = torch.sigmoid(logits)
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return ent / np.log(2)


def combined_uncertainty(learned_unc, logits):
    """Combine predictive entropy with learned uncertainty. Returns value in [0,1]."""
    entropy = binary_entropy_from_logits(logits)
    return 1 - (1 - entropy) * (1 - learned_unc)


class ReinterventionRiskModel(nn.Module):
    """Multi-task model: reintervention risk + per-complication predictions."""

    def __init__(self, clinical_dim=744, aaa_dim=5, num_complications=NUM_COMPLICATIONS):
        super().__init__()
        self.num_complications = num_complications

        # Visual pathway
        self.cta_adapter = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv3d(8, 1, 1)
        )
        self.visual_base   = Merlin(ImageEmbedding=True)
        disable_checkpointing(self.visual_base)
        self.visual_branch = get_peft_model(
            self.visual_base,
            LoraConfig(r=8, target_modules=["query", "value", "query_global", "value_global"])
        )
        self.visual_proj = nn.Linear(2048, 256)

        # Per-feature learned gate. sigmoid(gate) ∈ (0,1) weights each input
        # feature. L1 reg during training pushes unimportant gates toward 0,
        # replacing manual SelectKBest with task-aware end-to-end selection.
        self.feature_gate = nn.Parameter(torch.zeros(clinical_dim))

        # Clinical pathway
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.Dropout(0.2),
        )

        # AAA metrics pathway
        self.aaa_metrics_branch = nn.Sequential(
            nn.Linear(aaa_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),     nn.ReLU(),
        )

        # Fusion
        total_fusion_dim = 256 + 32 + 32  # 320
        self.condition_net = nn.Sequential(
            nn.Linear(total_fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * total_fusion_dim),
        )
        self.fusion_norm = nn.LayerNorm(total_fusion_dim)

        # Reintervention head: logit + logvar (uncertainty)
        self.reintervention_head = nn.Sequential(
            nn.Linear(total_fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

        # Complications head: (logit + logvar) per complication — shared MLP
        self.complications_head = nn.Sequential(
            nn.Linear(total_fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_complications * 2),
        )

    def film_modulate(self, x, gamma_beta):
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * 0.1
        beta  = torch.tanh(beta)  * 0.1
        return x * (1 + gamma) + beta

    def encode_visual(self, cta):
        v = self.visual_branch(cta)
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() == 3 and v.shape[0] == 1:
            v = v.squeeze(0)
        return v.flatten(1) if v.dim() > 2 else v

    def forward(self, cta=None, clin=None, aaa=None, return_fused=False):
        cta   = self.cta_adapter(cta)
        v_emb = self.visual_proj(self.encode_visual(cta))
        c_emb = self.clinical_branch(clin * torch.sigmoid(self.feature_gate))
        a_emb = self.aaa_metrics_branch(aaa)

        fused = torch.cat([v_emb, c_emb, a_emb], dim=1)
        cond  = self.condition_net(fused)   # film_modulate applies its own tanh+scale
        fused = self.film_modulate(fused, cond)
        fused = self.fusion_norm(fused)

        if return_fused:
            return fused

        # Reintervention
        reint_out    = self.reintervention_head(fused)
        reint_logit  = reint_out[:, 0:1]
        reint_logvar = torch.clamp(reint_out[:, 1:2], -5, 5)

        # Complications: (B, C*2) → (B, C, 2) → split logit / logvar
        comp_out    = self.complications_head(fused)
        comp_out    = comp_out.view(comp_out.size(0), self.num_complications, 2)
        comp_logits = comp_out[:, :, 0]                       # (B, C)
        comp_logvar = torch.clamp(comp_out[:, :, 1], -5, 5)  # (B, C)

        return reint_logit, reint_logvar, comp_logits, comp_logvar


# --- MULTI-TASK LOSS ---
class MultiTaskReinterventionLoss(nn.Module):
    """
    Heteroscedastic focal loss for reintervention (primary) + masked focal loss per complication.
    Labels == -1 are masked out (missing ground truth).

    Uncertainty is calibrated via heteroscedastic weighting:
      loss = exp(-logvar) * focal_bce_per_sample + 0.5 * logvar
    Equilibrium: logvar ≈ log(2 * per_sample_loss), so the model learns to be
    confident when correct and uncertain when wrong — replacing the L2 penalty
    that was holding logvar near 0 and producing permanently LOW confidence.
    """

    def __init__(
        self,
        pos_weight_reint: float = 1.0,  # WeightedRandomSampler already balances batches ~50/50
        focal_gamma:      float = 2.0,
        label_smoothing:  float = 0.05,
        comp_weight:      float = 0.3,  # set per-stage in run_pipeline (0.1 → A, 0.3 → B)
    ):
        super().__init__()
        self.focal_gamma     = focal_gamma
        self.label_smoothing = label_smoothing
        self.comp_weight     = comp_weight
        self.register_buffer("pos_weight_reint", torch.tensor([pos_weight_reint], device=DEVICE))

    def _focal_bce_per_sample(self, logit, gt, pos_weight=None):
        """Per-sample focal BCE — needed for heteroscedastic precision weighting."""
        smooth_gt = gt * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            logit, smooth_gt, pos_weight=pos_weight, reduction="none"
        )
        p  = torch.sigmoid(logit).detach()
        pt = p * gt + (1 - p) * (1 - gt)
        return (1 - pt).pow(self.focal_gamma) * bce  # (B, 1)

    def _heteroscedastic(self, logit, logvar, gt, pos_weight=None):
        """Heteroscedastic binary focal loss.
        Clamp logvar (not precision) so gradients always flow; -6 ≈ conf 0.95 ceiling.
        """
        bce      = self._focal_bce_per_sample(logit, gt, pos_weight)
        logvar_c = logvar.clamp(min=-6.0)
        precision = torch.exp(-logvar_c)
        return (precision * bce).mean() + 0.5 * logvar.mean()

    def forward(self, reint_logit, reint_logvar, comp_logits, comp_logvar, reint_gt, comp_gt):
        # --- Reintervention heteroscedastic focal loss ---
        reint_gt_v = reint_gt.view_as(reint_logit)
        loss_reint = self._heteroscedastic(
            reint_logit, reint_logvar, reint_gt_v, pos_weight=self.pos_weight_reint
        )

        # --- Per-complication masked heteroscedastic focal loss ---
        comp_losses = []
        for c in range(comp_logits.shape[1]):
            valid = comp_gt[:, c] != -1  # mask missing labels
            if valid.sum() > 0:
                loss_c = self._heteroscedastic(
                    comp_logits[valid, c].unsqueeze(1),
                    comp_logvar[valid, c].unsqueeze(1),
                    comp_gt[valid, c].unsqueeze(1),
                )
                comp_losses.append(loss_c)

        loss_comp = torch.stack(comp_losses).mean() if comp_losses else \
                    torch.tensor(0.0, device=reint_logit.device)

        total = loss_reint + self.comp_weight * loss_comp

        return total, {
            "loss_total":       total.item(),
            "loss_reint":       loss_reint.item(),
            "loss_comp":        loss_comp.item(),
            "mean_uncertainty": reint_logvar.mean().item(),
        }


# --- XAI ---
class ReinterventionXAI:
    """Gradient-based and ablation-based XAI with full complication breakdown."""

    def __init__(self, model, feature_names):
        self.model         = model
        self.feature_names = np.array(feature_names)

    def explain(self, cta, clin, aaa) -> ReinterventionRiskPrediction:
        self.model.eval()

        with torch.no_grad():
            reint_logit, reint_logvar, comp_logits, comp_logvar = self.model(cta, clin, aaa)

        # Reintervention probability and confidence
        reint_prob  = float(torch.sigmoid(reint_logit).item())
        reint_std   = float(torch.exp(0.5 * reint_logvar).item())
        reint_conf  = 1.0 / (1.0 + reint_std)
        conf_level  = self._conf_level(reint_conf)

        # Per-complication probability and confidence — same formula as reintervention
        comp_probs_np  = torch.sigmoid(comp_logits).squeeze().cpu().numpy()
        comp_logvar_np = comp_logvar.squeeze().cpu().numpy()
        comp_prob_dict = {}
        comp_conf_dict = {}
        for i, name in enumerate(COMPLICATIONS):
            p  = float(comp_probs_np[i] if comp_probs_np.ndim > 0 else comp_probs_np)
            lv = float(comp_logvar_np[i] if comp_logvar_np.ndim > 0 else comp_logvar_np)
            comp_std  = float(np.exp(0.5 * lv))
            comp_conf = 1.0 / (1.0 + comp_std)
            comp_prob_dict[name] = p
            comp_conf_dict[name] = self._conf_level(comp_conf)

        # Gradient-based XAI for all branches
        clin_imp     = self._clinical_importance(cta, clin, aaa)
        aaa_imp      = self._aaa_importance(cta, clin, aaa)
        modality_imp = self._modality_importance(cta, clin, aaa)
        causal_chain = self._causal_chain(reint_prob, comp_prob_dict, comp_conf_dict, clin_imp, modality_imp)
        primary      = self._primary_factor(clin_imp, comp_prob_dict)

        # GradCAM on Merlin's last ResNet layer
        cam_np       = self._merlin_gradcam(cta, clin, aaa)
        gradcam_imgs = self._gradcam_wandb_images(cta, cam_np) if cam_np is not None else {}

        return ReinterventionRiskPrediction(
            reintervention_prob=reint_prob,
            reintervention_confidence=reint_conf,
            confidence_level=conf_level,
            complication_probs=comp_prob_dict,
            complication_confidence=comp_conf_dict,
            clinical_importance=clin_imp,
            aaa_importance=aaa_imp,
            modality_importance=modality_imp,
            causal_chain=causal_chain,
            primary_risk_factor=primary,
            gradcam_images=gradcam_imgs,
        )

    def _aaa_importance(self, cta, clin, aaa) -> List[Tuple[str, float, str]]:
        """Gradient-based importance for each AAA morphology metric."""
        aaa_var = aaa.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            reint_logit, *_ = self.model(cta, clin, aaa_var)
            torch.sigmoid(reint_logit).mean().backward()
        grads = aaa_var.grad.abs().squeeze().cpu().numpy()
        grads = grads / (grads.max() + 1e-8)  # normalise to [0, 1]
        result = []
        for name, score in zip(AAA_METRIC_NAMES, grads.tolist()):
            level = "HIGH" if score > 0.7 else ("MED" if score > 0.3 else "LOW")
            result.append((name, round(score, 4), level))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def _get_i3res(self):
        """Walk the LoRA/PEFT wrapper to find the I3ResNet module."""
        for _, mod in self.model.visual_branch.named_modules():
            if hasattr(mod, 'layer4') and hasattr(mod, 'layer1'):
                return mod
        return None

    def _merlin_gradcam(self, cta, clin, aaa):
        """GradCAM on Merlin's last ResNet layer → normalised (D,H,W) heatmap in [0,1]."""
        i3res = self._get_i3res()
        if i3res is None:
            return None

        store = {}

        def fwd_hook(module, inp, out):
            # retain_grad() works because checkpoint is patched to a passthrough below,
            # so 'out' flows through the graph normally and requires_grad=True.
            store['feat'] = out
            if out.requires_grad:
                out.retain_grad()

        # checkpoint.checkpoint wraps forward in torch.no_grad(), which strips
        # requires_grad from intermediate tensors and breaks register_hook.
        # Patching it to a passthrough for this single forward+backward call lets
        # gradients flow through all layers uninterrupted.
        import torch.utils.checkpoint as cp_module
        _orig_checkpoint = cp_module.checkpoint
        cp_module.checkpoint = lambda fn, *args, **kwargs: fn(*args)

        handle = i3res.layer4[-1].register_forward_hook(fwd_hook)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                reint_logit, *_ = self.model(cta=cta, clin=clin, aaa=aaa)
                reint_logit.sum().backward()
        finally:
            handle.remove()
            cp_module.checkpoint = _orig_checkpoint

        if 'feat' not in store or store['feat'].grad is None:
            return None

        acts    = store['feat'].detach().cpu()
        grads   = store['feat'].grad.detach().cpu()
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam     = torch.relu((weights * acts).sum(dim=1, keepdim=True))

        cam_up = F.interpolate(cam, size=cta.shape[2:], mode='trilinear', align_corners=False)
        cam_np = cam_up.squeeze().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np  # (D, H, W) in [0, 1]

    def _gradcam_wandb_images(self, cta, cam_np) -> Dict[str, object]:
        """Create WandB images: center axial/sagittal/coronal slices with GradCAM + seg overlay."""
        ct    = cta[0, 0].cpu().numpy()  # (D, H, W) — HU normalised to [0,1]
        lumen = cta[0, 1].cpu().numpy() > 0.5  # binary lumen mask
        ilt   = cta[0, 2].cpu().numpy() > 0.5  # binary ILT mask
        D, H, W = ct.shape

        views = {
            'axial':    (ct[D//2], cam_np[D//2], lumen[D//2], ilt[D//2]),
            'sagittal': (ct[:, H//2], cam_np[:, H//2], lumen[:, H//2], ilt[:, H//2]),
            'coronal':  (ct[:, :, W//2], cam_np[:, :, W//2], lumen[:, :, W//2], ilt[:, :, W//2]),
        }

        images = {}
        for view, (ct_sl, cam_sl, lumen_sl, ilt_sl) in views.items():
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(ct_sl,  cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.imshow(cam_sl, cmap='jet',  alpha=0.40, vmin=0, vmax=1, interpolation='nearest')

            # Segmentation overlays: lumen = cyan, ILT = yellow
            lumen_rgba       = np.zeros((*lumen_sl.shape, 4))
            lumen_rgba[lumen_sl] = [0.0, 1.0, 1.0, 0.35]
            ilt_rgba             = np.zeros((*ilt_sl.shape, 4))
            ilt_rgba[ilt_sl]     = [1.0, 1.0, 0.0, 0.35]
            ax.imshow(lumen_rgba, interpolation='nearest')
            ax.imshow(ilt_rgba,   interpolation='nearest')

            ax.set_title(f'GradCAM {view}', fontsize=9)
            ax.axis('off')
            fig.tight_layout(pad=0.1)
            images[f'gradcam_{view}'] = wandb.Image(fig)
            plt.close(fig)

        return images

    def _conf_level(self, conf: float) -> str:
        if conf > 0.75: return "HIGH"
        if conf > 0.55: return "MED"
        return "LOW"

    def _clinical_importance(self, cta, clin, aaa):
        clin_var = clin.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            reint_logit, *_ = self.model(cta, clin_var, aaa)
            score = torch.sigmoid(reint_logit).mean()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        grads = clin_var.grad.abs().squeeze().cpu().numpy()
        top   = np.argsort(grads)[::-1][:20]
        result = []
        for idx in top:
            s = float(grads[idx])
            pct = float((grads >= s).mean())
            level = "HIGH" if pct > 0.9 else ("MED" if pct > 0.7 else "LOW")
            result.append((self.feature_names[idx], s, level))
        return result

    def _modality_importance(self, cta, clin, aaa) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            base_logit, *_ = self.model(cta, clin, aaa)
            base_p         = float(torch.sigmoid(base_logit))
            no_cta, *_     = self.model(torch.zeros_like(cta), clin, aaa)
            no_clin, *_    = self.model(cta, torch.zeros_like(clin), aaa)
            no_aaa, *_     = self.model(cta, clin, torch.zeros_like(aaa))
        impacts = {
            "CTA":         abs(base_p - float(torch.sigmoid(no_cta))),
            "Clinical":    abs(base_p - float(torch.sigmoid(no_clin))),
            "AAA_metrics": abs(base_p - float(torch.sigmoid(no_aaa))),
        }
        total = sum(impacts.values()) or 1.0
        return {k: v / total for k, v in impacts.items()}

    def _causal_chain(self, reint_prob, comp_probs, comp_conf, clin_imp, modality_imp) -> str:
        top_comps = [
            (name, prob) for name, prob in
            sorted(comp_probs.items(), key=lambda x: x[1], reverse=True)
            if prob > 0.15
        ][:3]
        top_clin = clin_imp[0][0] if clin_imp else "clinical features"
        top_mod  = max(modality_imp, key=modality_imp.get)
        comp_str = ", ".join(
            f"{name} ({prob:.0%}, {comp_conf.get(name, '?')} conf)"
            for name, prob in top_comps
        ) or "no dominant complication"

        if reint_prob > 0.5:
            return (f"Elevated {top_clin} detected via {top_mod} → "
                    f"probable complications: {comp_str} → "
                    f"Likely reintervention ({reint_prob:.0%})")
        return (f"{top_clin} and {comp_str} — watchful monitoring advised "
                f"(reintervention risk {reint_prob:.0%})")

    def _primary_factor(self, clin_imp, comp_probs) -> str:
        top_comp  = max(comp_probs, key=comp_probs.get) if comp_probs else None
        top_prob  = comp_probs.get(top_comp, 0) if top_comp else 0
        if top_prob > 0.4:
            return f"Probable {top_comp} ({top_prob:.0%})"
        if clin_imp:
            return f"{clin_imp[0][0]} ({clin_imp[0][2]} impact)"
        return "Unknown"


# --- VALIDATION ---
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_reint_probs, all_reint_gt, all_reint_logvar = [], [], []
    all_comp_probs, all_comp_gt = [], []

    with torch.no_grad():
        for batch in val_loader:
            cta      = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            clin     = batch["clin"].to(DEVICE).float()
            aaa      = batch["aaa_metrics"].to(DEVICE).float()
            reint_gt = batch["reintervention_label"].to(DEVICE).float()
            comp_gt  = batch["complication_labels"].to(DEVICE).float()

            reint_logit, reint_logvar, comp_logits, comp_logvar = model(cta, clin, aaa)
            loss, _ = criterion(
                reint_logit, reint_logvar, comp_logits, comp_logvar, reint_gt, comp_gt
            )
            val_loss += loss.item()

            all_reint_probs.append(torch.sigmoid(reint_logit).cpu())
            all_reint_gt.append(reint_gt.cpu())
            all_reint_logvar.append(reint_logvar.cpu())
            all_comp_probs.append(torch.sigmoid(comp_logits).cpu())
            all_comp_gt.append(comp_gt.cpu())

    reint_p       = torch.cat(all_reint_probs).numpy().flatten()    # (N,)
    reint_g       = torch.cat(all_reint_gt).numpy().flatten().astype(int)  # (N,)
    reint_lv_np   = torch.cat(all_reint_logvar).numpy().flatten()
    comp_p        = torch.cat(all_comp_probs).numpy()   # (N, C)
    comp_g        = torch.cat(all_comp_gt).numpy()      # (N, C), -1 = missing

    # Find threshold that maximises F1 on the PR curve instead of using fixed 0.5
    p, r, t = precision_recall_curve(reint_g, reint_p)
    f1s          = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-8)
    best_f1_idx  = np.argmax(f1s)
    best_f1      = float(f1s[best_f1_idx])
    best_thresh  = float(t[best_f1_idx])
    reint_pred   = (reint_p > best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(reint_g, reint_pred, labels=[0, 1]).ravel()

    metrics = {
        "val/loss_total":        val_loss / len(val_loader),
        "val/reint_auc":         roc_auc_score(reint_g, reint_p) if len(np.unique(reint_g)) > 1 else 0.5,
        "val/reint_f1":          f1_score(reint_g, reint_pred),
        "val/reint_precision":   precision_score(reint_g, reint_pred, zero_division=0),
        "val/reint_accuracy":    accuracy_score(reint_g, reint_pred),
        "val/reint_recall":      recall_score(reint_g, reint_pred, zero_division=0),
        "val/reint_best_f1":     best_f1,
        "val/reint_best_thresh": best_thresh,
        "val/mean_uncertainty":  float(reint_lv_np.mean()),
        "val/reint_TP": int(tp),
        "val/reint_FP": int(fp),
        "val/reint_TN": int(tn),
        "val/reint_FN": int(fn),
        "val/reint_TPR": tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # sensitivity/recall
        "val/reint_TNR": tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # specificity
        "val/reint_FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,  # fall-out
        "val/reint_FNR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,  # miss rate
    }

    # Per-complication AUC (only where both classes are present in val set)
    comp_aucs = []
    for i, name in enumerate(COMPLICATIONS):
        valid_mask = comp_g[:, i] != -1
        if valid_mask.sum() > 0:
            y_t = comp_g[valid_mask, i]
            y_p = comp_p[valid_mask, i]
            if len(np.unique(y_t)) > 1:
                auc = roc_auc_score(y_t, y_p)
                metrics[f"val/auc_{name}"] = auc
                comp_aucs.append(auc)
    if comp_aucs:
        metrics["val/comp_mean_auc"] = float(np.mean(comp_aucs))

    wandb.log(metrics)
    print(
        f"[VAL] loss={metrics['val/loss_total']:.4f} | "
        f"reint_auc={metrics['val/reint_auc']:.4f} | "
        f"acc={metrics['val/reint_accuracy']:.4f} | "
        f"f1={metrics['val/reint_f1']:.4f} | "
        f"prec={metrics['val/reint_precision']:.4f} | "
        f"rec={metrics['val/reint_recall']:.4f} | "
        f"best_f1={metrics['val/reint_best_f1']:.4f} | "
        f"thresh={metrics['val/reint_best_thresh']:.3f} | "
        f"comp_mAUC={metrics.get('val/comp_mean_auc', 0.0):.4f}"
    )
    return val_loss / len(val_loader), metrics


def get_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m


# --- STAGE SETUP ---
def set_stage_a(model):
    """Stage A: Frozen Backbone — Fusion Warm-Up.

    Merlin/LoRA fully frozen. Both prediction heads, fusion, clinical, and AAA
    branches train at moderate LR. Goal: stable multimodal representations
    before adapting the visual backbone.
    """
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze everything EXCEPT LoRA (Merlin backbone stays frozen)
    for p in model.cta_adapter.parameters():
        p.requires_grad = True
    model.visual_proj.requires_grad_(True)
    model.feature_gate.requires_grad = True
    for p in model.clinical_branch.parameters():
        p.requires_grad = True
    for p in model.aaa_metrics_branch.parameters():
        p.requires_grad = True
    for p in model.condition_net.parameters():
        p.requires_grad = True
    for p in model.fusion_norm.parameters():
        p.requires_grad = True
    for p in model.reintervention_head.parameters():
        p.requires_grad = True
    for p in model.complications_head.parameters():
        p.requires_grad = True


def set_stage_b(model):
    """Stage B: LoRA Fine-Tuning — Backbone Adaptation.

    Adds LoRA adapters at very low LR on top of the stable Stage A fusion.
    Everything else continues at low LR.
    """
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False

    # Same as Stage A
    for p in model.cta_adapter.parameters():
        p.requires_grad = True
    model.visual_proj.requires_grad_(True)
    model.feature_gate.requires_grad = True
    for p in model.clinical_branch.parameters():
        p.requires_grad = True
    for p in model.aaa_metrics_branch.parameters():
        p.requires_grad = True
    for p in model.condition_net.parameters():
        p.requires_grad = True
    for p in model.fusion_norm.parameters():
        p.requires_grad = True
    for p in model.reintervention_head.parameters():
        p.requires_grad = True
    for p in model.complications_head.parameters():
        p.requires_grad = True

    # PLUS: LoRA adapters in Merlin at very low LR
    for name, p in model.visual_branch.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True


def build_optimizer(model, stage):
    model = get_model(model)

    if stage == "A":
        # Frozen backbone: moderate LR for all branches, no LoRA
        param_groups = [
            {"params": model.cta_adapter.parameters(),        "lr": 2e-4, "name": "cta_adapter"},
            {"params": model.visual_proj.parameters(),         "lr": 2e-4, "name": "visual_proj"},
            {"params": [model.feature_gate],                   "lr": 1e-3, "name": "feature_gate"},
            {"params": model.clinical_branch.parameters(),     "lr": 3e-4, "name": "clinical"},
            {"params": model.aaa_metrics_branch.parameters(),  "lr": 3e-4, "name": "aaa"},
            {"params": model.condition_net.parameters(),       "lr": 2e-4, "name": "condition"},
            {"params": model.fusion_norm.parameters(),         "lr": 2e-4, "name": "fusion_norm"},
            {"params": model.reintervention_head.parameters(), "lr": 2e-4, "name": "reint_head"},
            {"params": model.complications_head.parameters(),  "lr": 2e-4, "name": "comp_head"},
        ]
    elif stage == "B":
        # Joint fine-tuning: low LR for everything, very low for LoRA
        param_groups = [
            {"params": [model.feature_gate],                   "lr": 3e-4, "name": "feature_gate"},
            {"params": model.cta_adapter.parameters(),         "lr": 1e-5, "name": "cta_adapter"},
            {"params": model.visual_proj.parameters(),          "lr": 1e-5, "name": "visual_proj"},
            {"params": model.clinical_branch.parameters(),      "lr": 5e-5, "name": "clinical"},
            {"params": model.aaa_metrics_branch.parameters(),   "lr": 5e-5, "name": "aaa"},
            {"params": model.condition_net.parameters(),        "lr": 5e-5, "name": "condition"},
            {"params": model.fusion_norm.parameters(),          "lr": 5e-5, "name": "fusion_norm"},
            {"params": [p for n, p in model.visual_branch.named_parameters() if "lora" in n.lower()],
             "lr": 5e-6, "name": "lora"},
            {"params": model.reintervention_head.parameters(),  "lr": 5e-5, "name": "reint_head"},
            {"params": model.complications_head.parameters(),   "lr": 5e-5, "name": "comp_head"},
        ]

    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# --- HELPERS ---
def print_trainable_params(obj, text="Trainable params:"):
    total = 0
    trainable = 0

    # Case 1: single Parameter
    if isinstance(obj, torch.nn.Parameter):
        total = obj.numel()
        trainable = obj.numel() if obj.requires_grad else 0

    # Case 2: Module
    elif isinstance(obj, nn.Module):
        for _, p in obj.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

    else:
        print(f"{text} Unsupported type: {type(obj)}")
        return

    pct = 100 * trainable / max(total, 1)
    print(f"{text} {trainable} / {total} ({pct:.2f}%)")

def print_model_param_overview(model):
    model = get_model(model)
    print_trainable_params(model.clinical_branch, "Trainable params (clinical 1/3):")
    print_trainable_params(model.feature_gate, "Trainable params (clinical 2/3):")
    print_trainable_params(model.aaa_metrics_branch, "Trainable params (clinical 2/3):")
    print_trainable_params(model.visual_branch, "Trainable params (visual 1/2):")
    print_trainable_params(model.visual_proj, "Trainable params (visual 2/2):")
    print_trainable_params(model.complications_head, "Trainable params (complications_head):")
    print_trainable_params(model.fusion_norm, "Trainable params (fusion):")
    print_trainable_params(model.condition_net, "Trainable params (condition):")
    print_trainable_params(model.reintervention_head, "Trainable params (reintervention):")
    

def clean_clinical_features(feat_dict):
    cleaned = {}
    for k, v in feat_dict.items():
        try:
            cleaned[k] = float(v) if v not in ["", None] else 0.0
        except Exception:
            cleaned[k] = 0.0
    return cleaned


def log_detailed_metrics(stage, epoch, train_loss, val_loss, loss_dict, opt):
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


def log_val_table(model, val_loader, epoch, stage, thresh=0.5):
    """WandB table: reintervention + all complication predictions for every val case."""
    model.eval()
    columns = ["Epoch", "Case_ID", "Reint_Prob", "Reint_Pred", "Reint_GT", "Reint_Label_Type"]
    for comp in COMPLICATIONS:
        columns += [f"P_{comp}", f"Pred_{comp}", f"GT_{comp}"]

    def _label_type(pred, gt):
        if pred == 1 and gt == 1: return "TP"
        if pred == 1 and gt == 0: return "FP"
        if pred == 0 and gt == 0: return "TN"
        return "FN"

    table        = wandb.Table(columns=columns)
    case_counter = 0

    with torch.no_grad():
        for batch in val_loader:
            cta      = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            clin     = batch["clin"].to(DEVICE).float()
            aaa      = batch["aaa_metrics"].to(DEVICE).float()
            reint_gt = batch["reintervention_label"].cpu().numpy()
            comp_gt  = batch["complication_labels"].cpu().numpy()

            reint_logit, _, comp_logits, _ = model(cta=cta, clin=clin, aaa=aaa)
            probs      = torch.sigmoid(reint_logit).cpu().numpy().flatten()
            comp_probs = torch.sigmoid(comp_logits).cpu().numpy()  # (B, C)

            for i in range(len(probs)):
                pred_i = int(probs[i] > thresh)
                gt_i   = int(reint_gt[i])
                row = [epoch, case_counter, round(float(probs[i]), 4),
                       pred_i, gt_i, _label_type(pred_i, gt_i)]
                for j in range(NUM_COMPLICATIONS):
                    row += [round(float(comp_probs[i, j]), 4), int(comp_probs[i, j] > 0.5), int(comp_gt[i, j])]
                table.add_data(*row)
                case_counter += 1

    wandb.log({f"eval/full_matrix_stage_{stage}": table, "epoch": epoch})
    print(f"Logged full matrix for {case_counter} cases.")


# --- MAIN TRAINING PIPELINE ---
def run_pipeline():
    with open(BASE_DIR / "dataset_splits.json") as f:
        splits = json.load(f)
    with open(BASE_DIR / "clinical_data/radar_multimodal_config_v1.json") as f:
        config = json.load(f)

    feature_names = sorted(config[list(config.keys())[0]]["pre_op_features"].keys())

    for pid in config:
        if "pre_op_features" in config[pid]:
            config[pid]["pre_op_features"] = clean_clinical_features(config[pid]["pre_op_features"])

    clin_matrix = np.stack([
        np.array([config[pid]["pre_op_features"].get(k, 0.0) for k in feature_names], dtype=np.float32)
        for pid in splits["train"]
    ])

    # All features kept — the learned feature_gate selects end-to-end.
    # Zero-variance features normalize to 0 ((x-mean)/(0+1e-8)=0) so they cannot
    # influence the gate output; L1 reg will push their gate values to 0.
    # feature_names order is fixed here and saved in the checkpoint so inference
    # can reconstruct the clinical vector in exactly the same column order.
    CLIN_MEAN = clin_matrix.mean(axis=0)
    CLIN_STD  = clin_matrix.std(axis=0) + 1e-8
    print(f"Clinical features: {len(feature_names)} total (no pre-filtering; gate selects end-to-end)")

    train_files = build_data_list(splits["train"], BASE_DIR, config, feature_names)
    val_files   = build_data_list(splits["val"],   BASE_DIR, config, feature_names)

    # Class-balanced sampling on the primary task (reintervention)
    train_labels   = [int(item["reintervention_label"][0]) for item in train_files]
    class_counts   = np.bincount(train_labels)
    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    train_sampler  = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    _cached = get_cached_transforms()
    train_ds = RuntimeDataset(
        CacheDataset(data=train_files, transform=_cached, cache_rate=0.5, num_workers=8),
        get_train_runtime_transforms(CLIN_MEAN, CLIN_STD),
    )
    val_ds = RuntimeDataset(
        CacheDataset(data=val_files, transform=_cached, cache_rate=1.0, num_workers=8),
        get_val_runtime_transforms(CLIN_MEAN, CLIN_STD),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    wandb.init(
        project="AAA-Reintervention-Risk",
        name=f"multitask_complications_v1_c{len(feature_names)}",
        config={
            "clinical_dim":      len(feature_names),
            "num_complications": NUM_COMPLICATIONS,
            "batch_size":        BATCH_SIZE,
            "lora_r":            8,
            "staging":           "A=frozen_backbone | B=lora_finetuning",
        },
    )

    model    = ReinterventionRiskModel(clinical_dim=len(feature_names)).to(DEVICE)
    save_dir = Path("/home/ranema/AI4EVAR/checkpoints_reint_comp")
    save_dir.mkdir(exist_ok=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = MultiTaskReinterventionLoss()
    xai       = ReinterventionXAI(get_model(model), feature_names)

    stages = [
        {"name": "A", "epochs": 40,  "patience": 8},   # Frozen backbone warm-up
        {"name": "B", "epochs": 120, "patience": 12},  # LoRA fine-tuning
    ]

    for stage_cfg in stages:
        st = stage_cfg["name"]
        print(f"\n>>> STARTING STAGE {st}")

        if st == "A":
            set_stage_a(get_model(model))
            # Low comp_weight: complications are sparse + visual backbone frozen →
            # noisy gradients from 11 sparse targets would destabilise the trunk.
            criterion.comp_weight = 0.1
        elif st == "B":
            # get_model(model).load_state_dict(torch.load(save_dir / "best_stage_a.pt"))
            get_model(model).load_state_dict(torch.load(save_dir / "best_stage_a.pt")["state_dict"])
            set_stage_b(get_model(model))
            # Visual features now available — complications become more learnable.
            criterion.comp_weight = 0.3

        opt       = build_optimizer(get_model(model), st)
        T_0       = 20 if st == "A" else 30
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=2, eta_min=1e-7)

        best_val_metric  = 0.0
        patience_counter = 0
        scaler           = torch.cuda.amp.GradScaler()

        model = model.to(DEVICE)
        model = model.to(memory_format=torch.channels_last_3d)

        print_model_param_overview(model)

        for epoch in range(stage_cfg["epochs"]):
            model.train()
            train_loss     = 0.0
            last_loss_dict = {}

            pbar = tqdm(train_loader, desc=f"Stage {st} Epoch {epoch}")
            for i, batch in enumerate(pbar):
                clin = batch["clin"].to(DEVICE).float()
                aaa  = batch["aaa_metrics"].to(DEVICE).float()
                cta  = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()

                # Modality dropout only in Stage B (fusion already stable)
                if st == "B":
                    cta, clin, aaa = maybe_apply_modality_dropout(
                        model.training, cta, clin, aaa,
                        p_cta=0.1, p_aaa_given_cta=0.15, p_aaa_given_clin=0.15,
                    )

                reint_gt = batch["reintervention_label"].to(DEVICE).float()
                comp_gt  = batch["complication_labels"].to(DEVICE).float()

                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    reint_logit, reint_logvar, comp_logits, comp_logvar = model(
                        cta=cta, clin=clin, aaa=aaa
                    )
                    loss, last_loss_dict = criterion(
                        reint_logit, reint_logvar,
                        comp_logits, comp_logvar,
                        reint_gt, comp_gt,
                    )
                    # Encourage sparse gating: L1 on sigmoid(gate) sums active features
                    loss = loss + 1e-3 * torch.sigmoid(get_model(model).feature_gate).sum()

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{train_loss/(i+1):.3f}")

            # Validation + logging
            val_loss, val_metrics = validate(model, val_loader, criterion)
            log_detailed_metrics(st, epoch, train_loss / len(train_loader), val_loss, last_loss_dict, opt)
            scheduler.step()
            tqdm.write(
                f"[Stage {st} | Epoch {epoch:03d}]  "
                f"train={train_loss/len(train_loader):.4f}  "
                f"val={val_loss:.4f}  "
                f"auc={val_metrics.get('val/reint_auc', 0):.4f}  "
                f"comp_auc={val_metrics.get('val/comp_mean_auc', 0):.4f}"
            )

            # Every 3 epochs: log full case table + XAI report
            if (epoch + 1) % 3 == 0:
                log_val_table(model, val_loader, epoch, st,
                              thresh=val_metrics.get("val/reint_best_thresh", 0.5))

                sample = next(iter(val_loader))
                expl   = xai.explain(
                    sample.get("cta", torch.zeros(1))[:1].to(DEVICE),
                    sample["clin"][:1].to(DEVICE),
                    sample["aaa_metrics"][:1].to(DEVICE),
                )
                wandb.log({
                    f"xai_{st}/causal_chain": wandb.Html(f"<p>{expl.causal_chain}</p>"),
                    f"xai_{st}/reint_prob":   expl.reintervention_prob,
                    f"xai_{st}/conf_level":   expl.confidence_level,
                    f"xai_{st}/primary_factor": expl.primary_risk_factor,
                    f"xai_{st}/top_features": ", ".join(f"{f[0]} ({f[2]})" for f in expl.clinical_importance[:3]),
                    f"xai_{st}/top_aaa":      ", ".join(f"{a[0]} ({a[2]})" for a in expl.aaa_importance[:3]),
                    **{f"xai_{st}/aaa_{name}": score for name, score, _ in expl.aaa_importance},
                    **{f"xai_{st}/{k}": v for k, v in expl.gradcam_images.items()},
                    **{f"xai_{st}/comp_{n}":  p for n, p in expl.complication_probs.items()},
                })
                print(expl.format_report())

            # Early stopping on reintervention AUC
            val_auc = val_metrics.get("val/reint_auc", 0.5)
            if val_auc > best_val_metric:
                best_val_metric  = val_auc
                patience_counter = 0
                # torch.save(get_model(model).state_dict(), save_dir / f"best_stage_{st.lower()}.pt")
                torch.save(
                    {
                        "state_dict": get_model(model).state_dict(),
                        "threshold": val_metrics.get("val/reint_best_thresh", 0.5),
                        "feature_names": feature_names,
                        "clin_mean": CLIN_MEAN.tolist(),
                        "clin_std": CLIN_STD.tolist(),
                    },
                    save_dir / f"best_stage_{st.lower()}.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= stage_cfg["patience"]:
                    print(f"Early stopping at epoch {epoch} (best AUC={best_val_metric:.4f})")
                    break

    wandb.finish()


if __name__ == "__main__":
    run_pipeline()
