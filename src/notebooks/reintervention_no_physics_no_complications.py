"""
AAA REINTERVENTION RISK PREDICTION MODEL
=========================================

Predicts:
1. Likelihood of reintervention (primary)
2. Which complications will occur (secondary)
3. Causal chains and feature importance (XAI)

Output includes confidence scores and extensive explanations.
"""

# NOTE: Physics is always zero array and never used in the model!

import json, os
import random
import torch
import torch.nn as nn
import numpy as np
import pyvista as pv
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
import wandb
import torch.nn.functional as F
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    Resized,
    MapTransform,
    ToTensord,
    RandGaussianNoised,
    RandBiasFieldd, 
    RandScaleIntensityd,   
)
from dataclasses import dataclass
from typing import Dict, List, Tuple

from peft import LoraConfig, get_peft_model
from merlin import Merlin

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1")
CACHE_DIR = Path("/deepstore/datasets/mia/AI4EVAR/derivatives/monai_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
SIZE = (128, 128, 128)  # Target size for resampling
BATCH_SIZE=8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_all_seeds(3299)

# --- COMPLICATIONS LIST ---
COMPLICATIONS = [
    # --- ENDOLEAKS (Granular) ---
    "endoleak_1a",
    "endoleak_1b",
    "endoleak_2",
    "endoleak_3",
    "endoleak_5",
    
    # --- PATENCY & MECHANICAL ---
    "occlusion",
    "stenosis",
    "kinking",
    "fracture",
    
    # --- ENDOGRAFT SPECIFIC ---
    "migration",
    "infection",
]

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
    """Structured output for reintervention risk prediction with full XAI."""
    
    # Main prediction
    reintervention_prob: float  # 0-1 probability
    reintervention_confidence: float  # 0-1 confidence (lower std = higher confidence)
    confidence_level: str  # "HIGH", "MED", "LOW"
    
    # Feature importance
    clinical_importance: List[Tuple[str, float, str]]  # [(feature, score, impact_level)]
    aaa_importance: List[Tuple[str, float, str]]       # [(metric, score, level)]
    modality_importance: Dict[str, float]  # {modality: percentage}

    # Causal explanation
    causal_chain: str  # Natural language explanation
    primary_risk_factor: str  # Most important factor

    # GradCAM
    gradcam_images: Dict[str, object]  # {view: wandb.Image}
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "reintervention": {
                "probability": self.reintervention_prob,
                "confidence": self.confidence_level
            },
            "clinical_factors": self.clinical_importance[:10],
            "modality_importance": self.modality_importance,
            "causal_chain": self.causal_chain,
        }
    
    def format_report(self) -> str:
        """Generate human-readable report."""
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║          REINTERVENTION RISK PREDICTION REPORT                 ║
╚════════════════════════════════════════════════════════════════╝

REINTERVENTION RISK:
  Probability: {self.reintervention_prob:.1%}
  Confidence: {self.confidence_level}
  Range: {self.reintervention_prob - self.reintervention_confidence:.1%} to {self.reintervention_prob + self.reintervention_confidence:.1%}

PRIMARY RISK FACTOR:
  {self.primary_risk_factor}
"""
        report += f"""
TOP CLINICAL FACTORS:
"""
        for feat, score, level in self.clinical_importance[:5]:
            report += f"  • {feat}: {level} impact ({score:.3f})\n"

        report += f"""
AAA MORPHOLOGY IMPORTANCE:
"""
        for metric, score, level in self.aaa_importance:
            report += f"  • {metric}: {level} ({score:.3f})\n"

        report += f"""
MODALITY IMPORTANCE:
"""
        for mod, imp in sorted(self.modality_importance.items(), key=lambda x: x[1], reverse=True):
            report += f"  • {mod}: {imp:.1%}\n"

        report += f"""
CAUSAL CHAIN:
  {self.causal_chain}

"""

def causal_modality_dropout(
    cta, clin, aaa,
    p_cta=0.1,
    p_aaa_given_cta=0.2,
    p_aaa_given_clin=0.2
):
    bsz = cta.size(0)

    cta_mask = torch.rand(bsz, device=cta.device) > p_cta

    cta_d = cta.clone()
    clin_d = clin.clone()
    aaa_d = aaa.clone()

    for b in range(bsz):

        # --- CTA missing ---
        if not cta_mask[b]:
            cta_d[b] = 0
            # AAA still possible via clinical → keep
            continue

        # --- AAA depends on CTA OR clinical ---
        aaa_drop = (
            random.random() < p_aaa_given_cta
        ) or (
            random.random() < p_aaa_given_clin
        )

        if aaa_drop:
            aaa_d[b] = 0

    return cta_d, clin_d, aaa_d

def maybe_apply_modality_dropout(training, cta, clin, aaa, p_cta, p_aaa_given_cta, p_aaa_given_clin):
    if not training:
        return cta, clin, aaa
    return causal_modality_dropout(cta, clin, aaa, p_cta, p_aaa_given_cta, p_aaa_given_clin)

# --- DATA LOADING ---
def build_data_list(ids, base_dir, config, feature_keys):
    """Build dataset with reintervention and complication labels."""
    data = []
    for pid in ids:
        pdata = config.get(pid, {})
        clin_dict = pdata.get("pre_op_features", {})
        clin = np.nan_to_num(
            np.array(
                [clin_dict.get(k, 0.0) for k in feature_keys],
                dtype=np.float32
            )
        )
        
        # AAA metrics
        metrics_path = base_dir / "metricsTr" / f"{pid}.json"
        aaa_metrics = [0.0] * 5
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                m = json.load(f)
                aaa_metrics = [
                    m.get("volume_lumen_ml", 0.0),
                    m.get("volume_ilt_ml", 0.0),
                    m.get("volume_total_ml", 0.0),
                    m.get("max_lumen_to_wall_dist_mm", 0.0),
                    m.get("avg_lumen_to_wall_dist_mm", 0.0)
                ]
        
        gt = pdata.get("ground_truth", {})
        
        # Reintervention label
        reint_label = float(gt.get("reintervention", 0))
        
        data.append({
            "ct": str(base_dir / "imagesTr" / f"{pid}.nii.gz"),
            "lbl": str(base_dir / "labelsTr" / f"{pid}.nii.gz"),
            "dst": str(base_dir / "distMapsTr" / f"{pid}.nii.gz"),
            "clin": clin,
            "aaa_metrics": aaa_metrics,
            "reintervention_label": np.array([reint_label], dtype=np.float32),
            "pid": pid,
        })
    return data


class BuildCTAd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        ct = data["ct"]
        lbl = data["lbl"]
        dst = data["dst"]
        cta = torch.cat([ct, (lbl == 1).float(), (lbl == 2).float(), dst], dim=0)
        data["cta"] = cta
        return data

class NormalizeClinicald(MapTransform):
    def __init__(self, keys, mean, std, eps=1e-8):
        super().__init__(keys)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.eps = eps

    def __call__(self, data):
        clin = torch.as_tensor(data["clin"]).float()
        clin = (clin - self.mean) / (self.std + self.eps)
        data["clin"] = clin
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
        ScaleIntensityRanged(
            keys=["ct"], a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True,
        ),
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
        ToTensord(keys=["ct", "lbl", "dst", "aaa_metrics", "reintervention_label"]),
        NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
    ])


def get_val_runtime_transforms(clin_mean, clin_std):
    """Applied fresh every iteration: no augmentation — deterministic val metrics."""
    return Compose([
        BuildCTAd(keys=["ct", "lbl", "dst"]),
        ToTensord(keys=["ct", "lbl", "dst", "aaa_metrics", "reintervention_label"]),
        NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
    ])


class RuntimeDataset(torch.utils.data.Dataset):
    """Wraps a cached dataset and applies a runtime transform on every access."""
    def __init__(self, base_ds, runtime_transform):
        self.base_ds = base_ds
        self.runtime = runtime_transform
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        return self.runtime(self.base_ds[idx])


# --- MODEL ---
def disable_checkpointing(model):
    import torch.utils.checkpoint as cp
    cp.use_reentrant = False
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
        if hasattr(m, "enable_gradient_checkpointing"):
            try:
                m.enable_gradient_checkpointing(False)
            except:
                pass

def binary_entropy_from_logits(logits, eps=1e-8):
    p = torch.sigmoid(logits)
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return ent / np.log(2)  # normalize to [0,1]

def combined_uncertainty(learned_unc, logits):
    """
    learned_unc: tensor or float from model head
    logits: raw logits for entropy computation
    """
    entropy = binary_entropy_from_logits(logits)
    return 1 - (1 - entropy) * (1 - learned_unc)

class ReinterventionRiskModel(nn.Module):
    """Multi-task model predicting reintervention risk and complications."""
    
    def __init__(self, clinical_dim=744, aaa_dim=5, num_complications=len(COMPLICATIONS)):
        super().__init__()
        
        # Visual pathway
        self.cta_adapter = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1)
        )
        self.visual_base = Merlin(ImageEmbedding=True)
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

        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.Dropout(0.2),
        )
        
        # AAA metrics pathway
        self.aaa_metrics_branch = nn.Sequential(
            nn.Linear(aaa_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Clinical-only head (Stage A)
        self.clinical_head = nn.Sequential(
            # nn.Linear(64 + 32, 128),
            nn.Linear(32 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_complications + 2)  # +2 for reint + sac_growth
        )
        
        # Fusion
        total_fusion_dim = 256 + 32 + 32
        self.condition_net = nn.Sequential(
            nn.Linear(total_fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * total_fusion_dim),
        )
        self.fusion_norm = nn.LayerNorm(total_fusion_dim)
        
        # Output heads
        self.reintervention_head = nn.Sequential(
            nn.Linear(total_fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # logit + uncertainty
        )
        
    def film_modulate(self, x, gamma_beta):
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * 0.1
        beta = torch.tanh(beta) * 0.1
        return x * (1 + gamma) + beta

    def encode_visual(self, cta):
        v = self.visual_branch(cta)
        if v.dim() == 3 and v.shape[0] == 1:
            v = v.squeeze(0)
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() > 2:
            v = v.flatten(1)
        return v

    def forward_from_fused(self, fused):
        """
        Compute model outputs directly from fused representation.
        Used for XAI / intervention-style analysis.
        """

        # --- Reintervention head ---
        reint_output = self.reintervention_head(fused)
        reint_logit = reint_output[:, 0:1]
        reint_logvar = reint_output[:, 1:2]

        return reint_logit, reint_logvar#, comp_logits, comp_unc#, sac_growth, sac_unc
    
    def forward_from_fused_minimal(self, fused):
        reint_output = self.reintervention_head(fused)
        reint_logit = reint_output[:, 0:1]
        comp_output = self.complications_head(fused)
        comp_output = comp_output.view(comp_output.size(0), self.num_complications, 2)
        comp_logits = comp_output[:, :, 0]
        return reint_logit, comp_logits

    def forward(self, cta=None, clin=None, aaa=None, stage="B", return_uncertainties=True, return_fused=False):
        """
        Forward pass.
        
        Returns (if return_uncertainties=True):
            reint_logit, reint_logvar, comp_logits, comp_unc, sac_growth, sac_unc
        """
        
        # Stage B/C: Full multimodal
        cta = self.cta_adapter(cta)
        v_out = self.encode_visual(cta)
        if isinstance(v_out, (tuple, list)):
            v_out = v_out[0]
        v_emb = v_out.flatten(1)
        v_emb = self.visual_proj(v_emb)
        
        c_emb = self.clinical_branch(clin * torch.sigmoid(self.feature_gate))
        a_emb = self.aaa_metrics_branch(aaa)

        # Fusion
        fused = torch.cat([v_emb, c_emb, a_emb], dim=1)
        cond = self.condition_net(fused)   # film_modulate applies its own tanh+scale
        fused = self.film_modulate(fused, cond)
        fused = self.fusion_norm(fused)
        if return_fused:
            return fused
        
        # Outputs
        reint_output = self.reintervention_head(fused)  
        reint_logit = reint_output[:, 0:1]
        reint_logvar = reint_output[:, 1:2]
        reint_logvar = torch.clamp(reint_logvar, -2, 2)
       
        if return_uncertainties:
            return reint_logit, reint_logvar
        else:
            return reint_logit
    
class ReinterventionLoss(nn.Module):
    def __init__(self, pos_weight_val=1.0, focal_gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        # WeightedRandomSampler already balances batches ~50/50, so pos_weight=1.0
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val], device=DEVICE))

    def forward(self, reint_logit, reint_logvar, reint_gt):
        reint_gt = reint_gt.view_as(reint_logit)

        smooth_gt = reint_gt * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            reint_logit, smooth_gt, pos_weight=self.pos_weight, reduction="none"
        )
        p  = torch.sigmoid(reint_logit).detach()
        pt = p * reint_gt + (1 - p) * (1 - reint_gt)
        per_sample = (1 - pt).pow(self.focal_gamma) * bce  # (B, 1)

        # Heteroscedastic NLL. Clamp matches forward clamp (-2, 2) so precision ≤ exp(2) ≈ 7.4×.
        # L2 term softly penalises |logvar| so it cannot collapse to the boundary.
        logvar_c  = reint_logvar.clamp(-2.0, 2.0)
        precision = torch.exp(-logvar_c)
        total = (precision * per_sample).mean() + 0.5 * reint_logvar.mean() \
                + 0.05 * reint_logvar.pow(2).mean()

        return total, {
            "loss_total": total.item(),
            "loss_reint": per_sample.mean().item(),
            "mean_uncertainty": reint_logvar.mean().item()
        }

# --- XAI SYSTEM ---
class ReinterventionXAI:
    """Advanced XAI for reintervention risk prediction."""
    
    def __init__(self, model, feature_names, complications):
        self.model = model
        self.feature_names = np.array(feature_names)
        self.complications = complications

    def explain(self, cta, clin, aaa) -> ReinterventionRiskPrediction:
        """Generate comprehensive explanation for reintervention prediction."""
        self.model.eval()
        
        with torch.no_grad():
            reint_logit, reint_logvar = \
                self.model(cta, clin, aaa, return_uncertainties=True)
                
        # Convert to probabilities
        reint_prob = torch.sigmoid(reint_logit).item()
        
        # Confidence levels
        reint_std = torch.exp(0.5 * reint_logvar)
        reint_conf = 1.0 / (1.0 + reint_std.item())
        conf_level = self._get_confidence_level(reint_conf)
        
        # Feature importance
        clin_imp     = self._clinical_importance(cta, clin, aaa)
        aaa_imp      = self._aaa_importance(cta, clin, aaa)
        modality_imp = self._modality_importance(cta, clin, aaa)

        # Causal chain
        causal_chain   = self._generate_causal_chain(reint_prob, clin_imp, modality_imp)
        primary_factor = self._identify_primary_factor(clin_imp)

        # GradCAM on Merlin's last ResNet layer
        cam_np       = self._merlin_gradcam(cta, clin, aaa)
        gradcam_imgs = self._gradcam_wandb_images(cta, cam_np) if cam_np is not None else {}

        return ReinterventionRiskPrediction(
            reintervention_prob=reint_prob,
            reintervention_confidence=reint_conf,
            confidence_level=conf_level,
            clinical_importance=clin_imp,
            aaa_importance=aaa_imp,
            modality_importance=modality_imp,
            causal_chain=causal_chain,
            primary_risk_factor=primary_factor,
            gradcam_images=gradcam_imgs,
        )
    
    def _aaa_importance(self, cta, clin, aaa) -> List[Tuple[str, float, str]]:
        """Gradient-based importance for each AAA morphology metric."""
        aaa_var = aaa.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            reint_logit, *_ = self.model(cta, clin, aaa_var, return_uncertainties=True)
            torch.sigmoid(reint_logit).mean().backward()
        grads = aaa_var.grad.abs().squeeze().cpu().numpy()
        grads = grads / (grads.max() + 1e-8)
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
        return cam_np

    def _gradcam_wandb_images(self, cta, cam_np) -> Dict[str, object]:
        """WandB images: center axial/sagittal/coronal slices with GradCAM + seg overlay."""
        ct    = cta[0, 0].cpu().numpy()
        lumen = cta[0, 1].cpu().numpy() > 0.5
        ilt   = cta[0, 2].cpu().numpy() > 0.5
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
            lumen_rgba          = np.zeros((*lumen_sl.shape, 4))
            lumen_rgba[lumen_sl] = [0.0, 1.0, 1.0, 0.35]
            ilt_rgba            = np.zeros((*ilt_sl.shape, 4))
            ilt_rgba[ilt_sl]    = [1.0, 1.0, 0.0, 0.35]
            ax.imshow(lumen_rgba, interpolation='nearest')
            ax.imshow(ilt_rgba,   interpolation='nearest')
            ax.set_title(f'GradCAM {view}', fontsize=9)
            ax.axis('off')
            fig.tight_layout(pad=0.1)
            images[f'gradcam_{view}'] = wandb.Image(fig)
            plt.close(fig)
        return images

    def _get_confidence_level(self, conf: float) -> str:
        """Map confidence score to level."""
        if conf > 0.8:
            return "HIGH"
        elif conf > 0.6:
            return "MED"
        else:
            return "LOW"
    
    def _get_complication_confidences(self, comp_probs, comp_unc) -> Dict[str, str]:
        result = {}
        # Convert to numpy and ensure it is at least 1D
        comp_unc_np = comp_unc.detach().cpu().numpy().flatten()
        
        for i, comp in enumerate(self.complications):
            # If model provides 1 unc per comp, use i. If it's a scalar, use [0]
            unc_val = comp_unc_np[i] if len(comp_unc_np) > 1 else comp_unc_np[0]
            
            confidence_score = float(np.exp(-unc_val))
            result[comp] = self._get_confidence_level(confidence_score)
        return result
    
    def _clinical_importance(self, cta, clin, aaa):
        clin_var = clin.clone().detach().requires_grad_(True)

        reint_logit, *_ = self.model(
            cta, clin_var, aaa, return_uncertainties=True
        )

        # scalar target (batch-aggregated)
        score = torch.sigmoid(reint_logit).mean()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        grads = clin_var.grad.abs().squeeze().cpu().numpy()

        top_indices = np.argsort(grads)[::-1][:20]

        result = []
        for idx in top_indices:
            score = float(grads[idx])
            level = self._get_impact_level(score, grads)
            result.append((self.feature_names[idx], score, level))

        return result
    
    def _get_impact_level(self, score: float, all_scores: np.ndarray) -> str:
        """Determine impact level based on score percentile."""
        percentile = (all_scores >= score).mean()
        if percentile > 0.9:
            return "HIGH"
        elif percentile > 0.7:
            return "MED"
        else:
            return "LOW"
    
    # def _modality_importance(self, cta, phys, clin, aaa) -> Dict[str, float]:
    def _modality_importance(self, cta, clin, aaa) -> Dict[str, float]:
        """Compute importance of each modality."""
        self.model.eval()
        
        # Ablation approach
        with torch.no_grad():
            reint_logit_full, *_ = self.model(cta, clin, aaa)
            reint_full = torch.sigmoid(reint_logit_full).item()
        
        # Zero each modality and measure impact
        impacts = {}
        
        with torch.no_grad():
            cta_zero = torch.zeros_like(cta)
            reint_no_cta, *_ = self.model(cta_zero, clin, aaa)
            impacts['CTA'] = abs(reint_full - torch.sigmoid(reint_no_cta).item())
            
            clin_zero = torch.zeros_like(clin)
            reint_no_clin, *_ = self.model(cta, clin_zero, aaa)
            impacts['Clinical'] = abs(reint_full - torch.sigmoid(reint_no_clin).item())
            
            aaa_zero = torch.zeros_like(aaa)
            reint_no_aaa, *_ = self.model(cta, clin, aaa_zero)
            impacts['AAA_metrics'] = abs(reint_full - torch.sigmoid(reint_no_aaa).item())
        
        # Normalize
        total = sum(impacts.values())
        if total > 0:
            impacts = {k: v / total for k, v in impacts.items()}
        
        return impacts
    
    def _generate_causal_chain(self, reint_prob, clin_imp, modality_imp) -> str:
        """Generate natural language causal explanation."""
        
        # Find dominant clinical feature
        top_clin_feat = clin_imp[0][0] if clin_imp else "unknown factors"
        
        # Find dominant modality
        top_modality = max(modality_imp, key=modality_imp.get)
        
        # Generate explanation
        if reint_prob > 0.5:
            chain = f"Elevated {top_clin_feat} detected by {top_modality} → Likely reintervention needed"
        else:
            chain = f"Clinical factors ({top_clin_feat}) suggest monitoring advised, low complication risk"
        
        return chain
    
    def _identify_primary_factor(self, clin_imp: List) -> str:
        """Identify the single most important factor."""
        if not clin_imp:
            return "Unable to determine"
        
        top_feat = clin_imp[0][0]
        impact_level = clin_imp[0][2]
        
        return f"{top_feat} ({impact_level} impact)"


# --- UTILITIES ---
def validate(model, val_loader, criterion, stage="B"):
    model.eval()
    val_loss = 0
    
    # Storage for metrics
    all_reint_probs, all_reint_gt, all_reint_logvar = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            # Setup inputs
            cta = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa = batch["aaa_metrics"].to(DEVICE).float()
            
            reint_gt = batch["reintervention_label"].to(DEVICE).float()
            
            # 1. Capture ALL outputs, including uncertainties
            reint_logit, reint_logvar = model(
                cta, clin, aaa, return_uncertainties=True
            )
            
            # 2. Pass the uncertainty tensors to the criterion instead of None
            loss, loss_dict = criterion(
                reint_logit, reint_logvar, 
                reint_gt
            )
            
            val_loss += loss.item()

            # Collect for sklearn metrics
            all_reint_probs.append(torch.sigmoid(reint_logit).cpu())
            all_reint_gt.append(reint_gt.cpu())
            all_reint_logvar.append(reint_logvar.cpu())

    # Concatenate all results — flatten to 1D so sklearn doesn't see (N,1) as multilabel
    reint_p = torch.cat(all_reint_probs).numpy().flatten()
    reint_g = torch.cat(all_reint_gt).numpy().flatten().astype(int)
    reint_logvar = torch.cat(all_reint_logvar).numpy().flatten()

    # Find threshold that maximises F1 on the PR curve instead of using fixed 0.5
    p, r, t = precision_recall_curve(reint_g, reint_p)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-8)
    best_f1_idx = np.argmax(f1s)
    best_f1 = float(f1s[best_f1_idx])
    best_thresh = float(t[best_f1_idx])
    reint_pred = (reint_p > best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(reint_g, reint_pred, labels=[0, 1]).ravel()

    # Calculate final stats
    metrics = {
        "val/loss_total": val_loss / len(val_loader),
        "val/reint_auc": roc_auc_score(reint_g, reint_p) if len(np.unique(reint_g)) > 1 else 0.5,
        "val/reint_f1": f1_score(reint_g, reint_pred),
        "val/reint_precision": precision_score(reint_g, reint_pred, zero_division=0),
        "val/reint_accuracy": accuracy_score(reint_g, reint_pred),
        "val/reint_recall": recall_score(reint_g, reint_pred, zero_division=0),
        "val/reint_best_f1": best_f1,
        "val/reint_best_thresh": best_thresh,
        "val/mean_uncertainty": reint_logvar.mean(),
        "val/reint_TP": int(tp),
        "val/reint_FP": int(fp),
        "val/reint_TN": int(tn),
        "val/reint_FN": int(fn),
        "val/reint_TPR": tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # sensitivity/recall
        "val/reint_TNR": tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # specificity
        "val/reint_FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,  # fall-out
        "val/reint_FNR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,  # miss rate
    }
    
    wandb.log(metrics)

    print(
        f"[VAL] loss={metrics['val/loss_total']:.4f} | "
        f"auc={metrics['val/reint_auc']:.4f} | "
        f"acc={metrics['val/reint_accuracy']:.4f} | "
        f"f1={metrics['val/reint_f1']:.4f} | "
        f"prec={metrics['val/reint_precision']:.4f} | "
        f"rec={metrics['val/reint_recall']:.4f} | "
        f"best_f1={metrics['val/reint_best_f1']:.4f} | "
        f"thresh={metrics['val/reint_best_thresh']:.3f} | "
        f"mean_uncertainty={metrics['val/mean_uncertainty']:.4f}"
    )
    return val_loss / len(val_loader), metrics

def get_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def set_stage_a(model):
    """Stage A: Frozen Backbone — Fusion Warm-Up.

    Merlin/LoRA is fully frozen. All other branches (cta_adapter, visual_proj,
    clinical, AAA, fusion, prediction head) train at moderate LR.
    Goal: establish stable multimodal fusion before adapting the visual backbone.
    """
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze everything EXCEPT LoRA — Merlin backbone stays frozen
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


def set_stage_b(model):
    """Stage B: LoRA Fine-Tuning — Backbone Adaptation.

    Adds LoRA adapter training on top of Stage A. Very low LR for LoRA
    to avoid destabilizing the stable fusion learned in Stage A.
    The condition_net is re-initialised so that the Stage A clinical bias
    does not suppress the visual pathway once LoRA features become available.
    """
    model = get_model(model)
    # Start by freezing everything, then selectively open the gates
    for p in model.parameters():
        p.requires_grad = False

    # Reset FiLM condition network so it relearns with LoRA visual features.
    for m in model.condition_net.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        
    # Unfreeze Tabular Pathways
    model.feature_gate.requires_grad = True
    for p in model.clinical_branch.parameters():
        p.requires_grad = True
    for p in model.aaa_metrics_branch.parameters():
        p.requires_grad = True

    # Unfreeze Fusion Layers
    for p in model.condition_net.parameters():
        p.requires_grad = True
        
    # Unfreeze Visual Pathways (LoRA adapter + Input Adapters)
    for p in model.cta_adapter.parameters():
        p.requires_grad = True
    for name, p in model.visual_branch.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True
    model.visual_proj.requires_grad_(True)
    
    # Unfreeze Prediction/Uncertainty Heads
    for p in model.reintervention_head.parameters():
        p.requires_grad = True
    for p in model.fusion_norm.parameters():
        p.requires_grad = True

def build_optimizer(model, stage, lr_dict=None):
    """Build optimizer with stage-specific learning rates."""
    model = get_model(model)

    if stage == "A":
        # Frozen backbone: all fusion + clinical + AAA at moderate LR, no LoRA
        param_groups = [
            {"params": model.cta_adapter.parameters(),        "lr": 2e-4, "name": "cta_adapter"},
            {"params": model.visual_proj.parameters(),         "lr": 2e-4, "name": "visual_proj"},
            {"params": [model.feature_gate],                   "lr": 1e-3, "name": "feature_gate"},
            {"params": model.clinical_branch.parameters(),     "lr": 3e-4, "name": "clinical"},
            {"params": model.aaa_metrics_branch.parameters(),  "lr": 3e-4, "name": "aaa"},
            {"params": model.condition_net.parameters(),       "lr": 2e-4, "name": "condition"},
            {"params": model.fusion_norm.parameters(),         "lr": 2e-4, "name": "fusion_norm"},
            {"params": model.reintervention_head.parameters(), "lr": 2e-4, "name": "reint_head"},
        ]
    elif stage == "B":
        # Joint fine-tuning: lower LR throughout to preserve stable stage-A weights
        param_groups = [
            {"params": [model.feature_gate],                   "lr": 3e-4, "name": "feature_gate"},
            {"params": model.clinical_branch.parameters(),     "lr": 5e-5, "name": "clinical"},
            {"params": model.aaa_metrics_branch.parameters(),  "lr": 5e-5, "name": "aaa"},
            {"params": model.condition_net.parameters(),       "lr": 5e-5, "name": "condition"},
            {"params": model.fusion_norm.parameters(),         "lr": 5e-5, "name": "fusion_norm"},
            {"params": model.cta_adapter.parameters(),         "lr": 1e-5, "name": "cta_adapter"},
            {"params": [p for name, p in model.visual_branch.named_parameters() if "lora" in name.lower()], "lr": 5e-6, "name": "lora"},
            {"params": model.visual_proj.parameters(),         "lr": 1e-5, "name": "visual_proj"},
            {"params": model.reintervention_head.parameters(), "lr": 5e-5, "name": "reint_head"},
        ]

    return torch.optim.AdamW(param_groups, weight_decay=1e-4)

def denorm(x, mean, std):
    """Denormalize sac growth."""
    return x * std + mean

def clean_clinical_features(feat_dict):
    """Clean up clinical feature dictionary."""
    cleaned = {}
    for k, v in feat_dict.items():
        try:
            cleaned[k] = float(v) if v not in ["", None] else 0.0
        except:
            cleaned[k] = 0.0
    return cleaned
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
    print_trainable_params(model.clinical_branch, "Trainable params (clinical 1/4):")
    print_trainable_params(model.feature_gate, "Trainable params (clinical 2/4):")
    print_trainable_params(model.aaa_metrics_branch, "Trainable params (clinical 3/4):")
    print_trainable_params(model.clinical_head, "Trainable params (clinical 4/4):")
    print_trainable_params(model.visual_branch, "Trainable params (visual 1/2):")
    print_trainable_params(model.visual_proj, "Trainable params (visual 2/2):")
    print_trainable_params(model.fusion_norm, "Trainable params (fusion):")
    print_trainable_params(model.condition_net, "Trainable params (condition):")
    print_trainable_params(model.reintervention_head, "Trainable params (reintervention):")
    
# --- ADD THESE HELPER LOGGING FUNCTIONS ---

def log_detailed_metrics(stage, epoch, train_loss, val_loss, loss_dict, opt):
    """Logs learning rates and split losses."""
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
        
    wandb.log(log_data)

# def log_val_table(model, val_loader, epoch, stage, sac_mean, sac_std):
def log_val_table(model, val_loader, epoch, stage, thresh=0.5):
    """Generates a WandB table with reintervention pred/GT for all val cases."""
    model.eval()

    columns = ["Epoch", "Case_ID", "Reint_Prob", "Reint_Pred", "Reint_GT",
               "Reint_Label_Type", "Error_Reint", "Logit_Reint"]
    table = wandb.Table(columns=columns)

    def _label_type(pred, gt):
        if pred == 1 and gt == 1: return "TP"
        if pred == 1 and gt == 0: return "FP"
        if pred == 0 and gt == 0: return "TN"
        return "FN"

    case_counter = 0
    with torch.no_grad():
        for batch in val_loader:
            cta  = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa  = batch["aaa_metrics"].to(DEVICE).float()
            reint_gt = batch["reintervention_label"].cpu().numpy()

            reint_logit, *_ = model(cta=cta, clin=clin, aaa=aaa, stage=stage)
            probs = torch.sigmoid(reint_logit).cpu().numpy().flatten()
            preds = (probs > thresh).astype(int)

            for i in range(len(probs)):
                pred_i = int(preds[i])
                gt_i   = int(reint_gt[i])
                row_data = [
                    epoch, case_counter,
                    round(float(probs[i]), 4), pred_i, gt_i,
                    _label_type(pred_i, gt_i),
                    abs(float(probs[i]) - gt_i),
                    round(float(reint_logit[i].cpu().numpy()), 4),
                ]
                table.add_data(*row_data)
                case_counter += 1

    wandb.log({f"eval/full_matrix_stage_{stage}": table, "epoch": epoch})
    print(f"Successfully logged full matrix for {case_counter} cases.")

# --- MAIN TRAINING PIPELINE ---
def run_pipeline():
    """Main training pipeline."""
    
    # Load configuration
    with open(BASE_DIR / "dataset_splits.json") as f:
        splits = json.load(f)
    with open(BASE_DIR / "clinical_data/radar_multimodal_config_v1.json") as f:
        config = json.load(f)
    
    feature_names = sorted(config[list(config.keys())[0]]['pre_op_features'].keys())
    
    # Clean features
    for pid in config.keys():
        if "pre_op_features" in config[pid]:
            config[pid]["pre_op_features"] = clean_clinical_features(
                config[pid]["pre_op_features"]
            )
    
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
    CLIN_STD = clin_matrix.std(axis=0) + 1e-8
    print(f"Clinical features: {len(feature_names)} total (no pre-filtering; gate selects end-to-end)")

    # Build datasets
    train_files = build_data_list(splits["train"], BASE_DIR, config, feature_names)
    val_files = build_data_list(splits["val"], BASE_DIR, config, feature_names)

    # Weighted sampler: ensures each batch has balanced positives/negatives
    # Without this, batches of 8 average ~1.4 positives (18% rate) → noisy gradients
    train_labels = [int(item["reintervention_label"][0]) for item in train_files]
    class_counts = np.bincount(train_labels)
    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
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

    # train_loader_a = DataLoader(train_ds_a, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    # Initialize W&B
    wandb.init(
        project="AAA-Reintervention-Risk",
        name=f"multimodal_v2_c{len(feature_names)}_binary_sac_regregression",
        config={
            "clinical_dim": len(feature_names),
            "num_complications": len(COMPLICATIONS),
            "batch_size": BATCH_SIZE,
            "lora_r": 16,
        }
    )
    
    # Build model
    model = ReinterventionRiskModel(clinical_dim=len(feature_names)).to(DEVICE)
    save_dir = Path("/home/ranema/AI4EVAR/checkpoints_reint")
    save_dir.mkdir(exist_ok=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    criterion = ReinterventionLoss()
    xai = ReinterventionXAI(model, feature_names, COMPLICATIONS)

    stages = [
        {"name": "A", "epochs": 40, "patience": 8},    # Warm start — stop early if overfitting
        {"name": "B", "epochs": 120, "patience": 12},  # Joint fine-tuning
    ]

    for stage_cfg in stages:
        st = stage_cfg["name"]
        print(f"\n>>> STARTING STAGE {st}")
        
        if st == "A": set_stage_a(model)
        elif st == "B": 
            model.load_state_dict(torch.load(save_dir / "best_stage_a.pt")["state_dict"])
            set_stage_b(model)
        elif st == "C":
            model.load_state_dict(torch.load(save_dir / "best_stage_b.pt")["state_dict"])

            set_stage_c(model)

        opt = build_optimizer(model, st)
        T_0 = 20 if st == "A" else 30
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=2, eta_min=1e-7)
        best_val_metric = 0.0  # Track AUC for early stopping (more stable than loss)
        patience_counter = 0

        scaler = torch.cuda.amp.GradScaler()
        print_model_param_overview(model)

        model = model.to(DEVICE)
        model = model.to(memory_format=torch.channels_last_3d)
        
        for epoch in range(stage_cfg["epochs"]):
            model.train()
            train_loss = 0
            last_loss_dict = {}

            pbar = tqdm(train_loader, desc=f"Stage {st} Epoch {epoch}")
            for i, batch in enumerate(pbar):
                # Standard training step...
                clin = batch["clin"].to(DEVICE).float()
                aaa = batch["aaa_metrics"].to(DEVICE).float()
                cta = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
                
                if st != "A":
                    cta, clin, aaa = maybe_apply_modality_dropout(
                        model.training, cta, clin, aaa,
                        p_cta=0.1 if st == "B" else 0.05,
                        p_aaa_given_cta = 0.15,
                        p_aaa_given_clin = 0.15,
                    )
                
                reint_gt = batch["reintervention_label"].to(DEVICE).float()
                
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    reint_logit, reint_logvar = model(
                        cta=cta,
                        clin=clin,
                        aaa=aaa,
                        stage=st,
                    )

                    loss, last_loss_dict = criterion(
                        reint_logit, reint_logvar,
                        reint_gt,
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

            # --- VALIDATION & LOGGING ---
            val_loss, val_metrics = validate(model, val_loader, criterion, stage=st)

            # 1. Log numerical metrics
            log_detailed_metrics(st, epoch, train_loss/len(train_loader), val_loss, last_loss_dict, opt)
            scheduler.step()
            tqdm.write(
                f"[Stage {st} | Epoch {epoch:03d}]  "
                f"train={train_loss/len(train_loader):.4f}  "
                f"val={val_loss:.4f}  "
                f"auc={val_metrics.get('val/reint_auc', 0):.4f}  "
                f"acc={val_metrics.get('val/reint_accuracy', 0):.4f}"
            )

            # 2. Log Table & XAI every 10 epochs
            if (epoch + 1) % 3 == 0:
                log_val_table(model, val_loader, epoch, st,
                              thresh=val_metrics.get("val/reint_best_thresh", 0.5))
                
                # Sample XAI for the first case in the val_loader
                sample = next(iter(val_loader))
                expl = xai.explain(
                    sample.get("cta", torch.zeros(1))[:1].to(DEVICE), 
                    sample["clin"][:1].to(DEVICE), 
                    sample["aaa_metrics"][:1].to(DEVICE)
                )
                wandb.log({
                    f"xai_{st}/causal_chain": wandb.Html(f"<b>Causal Chain:</b> <p>{expl.causal_chain}</p>"),
                    f"xai_{st}/reint_prob":   expl.reintervention_prob,
                    f"xai_{st}/conf_level":   expl.confidence_level,
                    f"xai_{st}/primary_factor": expl.primary_risk_factor,
                    f"xai_{st}/top_features": ", ".join(f"{f[0]} ({f[2]})" for f in expl.clinical_importance[:3]),
                    f"xai_{st}/top_aaa":      ", ".join(f"{a[0]} ({a[2]})" for a in expl.aaa_importance[:3]),
                    **{f"xai_{st}/aaa_{name}": score for name, score, _ in expl.aaa_importance},
                    **{f"xai_{st}/{k}": v for k, v in expl.gradcam_images.items()},
                })
                print(expl.format_report())

            # Early Stopping Logic (monitor AUC — more stable than loss with imbalanced data)
            val_auc = val_metrics.get("val/reint_auc", 0.5)
            if val_auc > best_val_metric:
                best_val_metric = val_auc
                patience_counter = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
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