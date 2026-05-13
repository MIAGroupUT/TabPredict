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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import wandb
import torch.nn.functional as F
from monai.data import PersistentDataset
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
    "endoleak",
    "migration",
    "infection",
    # "erosion",
    "limb_obstruction",
    "claudication",
    # "occlusion",
    # "rupture"
    "sac_expansion_using_min"
]

# --- OUTPUT STRUCTURE ---
@dataclass
class ReinterventionRiskPrediction:
    """Structured output for reintervention risk prediction with full XAI."""
    
    # Main prediction
    reintervention_prob: float  # 0-1 probability
    reintervention_confidence: float  # 0-1 confidence (lower std = higher confidence)
    confidence_level: str  # "HIGH", "MED", "LOW"
    
    # Complication predictions
    complication_probs: Dict[str, float]  # {name: probability}
    complication_confidence: Dict[str, float]  # {name: confidence_level}
    
    # Feature importance
    clinical_importance: List[Tuple[str, float, str]]  # [(feature, score, impact_level)]
    modality_importance: Dict[str, float]  # {modality: percentage}
    
    # Complication drivers
    complication_drivers: Dict[str, List[Tuple[str, float]]]  # {complication: [(feature, importance)]}
    complication_influence_on_reint: Dict[str, float]  # {complication: influence_score}
    
    # Causal explanation
    causal_chain: str  # Natural language explanation
    primary_risk_factor: str  # Most important factor
    
    # Sac growth (auxiliary)
    # sac_growth_mm: float
    # sac_growth_confidence: float
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "reintervention": {
                "probability": self.reintervention_prob,
                "confidence": self.confidence_level
            },
            "complications": self.complication_probs,
            "clinical_factors": self.clinical_importance[:10],
            "modality_importance": self.modality_importance,
            "causal_chain": self.causal_chain,
            "primary_risk_factor": self.primary_risk_factor,
            # "sac_growth_mm": self.sac_growth_mm
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

PREDICTED COMPLICATIONS (by influence on reintervention):
"""
        # Sort by influence on reintervention
        sorted_comps = sorted(
            self.complication_influence_on_reint.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for comp, influence in sorted_comps[:5]:
            prob = self.complication_probs.get(comp, 0)
            conf = self.complication_confidence.get(comp, "UNKNOWN")
            report += f"  • {comp}: {prob:.1%} ({conf} confidence, {influence:.1%} influence on reint)\n"

        report += f"""
TOP CLINICAL FACTORS:
"""
        for feat, score, level in self.clinical_importance[:5]:
            report += f"  • {feat}: {level} impact ({score:.3f})\n"

        report += f"""
MODALITY IMPORTANCE:
"""
        for mod, imp in sorted(self.modality_importance.items(), key=lambda x: x[1], reverse=True):
            report += f"  • {mod}: {imp:.1%}\n"

        report += f"""
CAUSAL CHAIN:
  {self.causal_chain}

"""
# SAC GROWTH PREDICTION:
#   Expected change: {self.sac_growth_mm:.2f} mm
#   Confidence: {self.sac_growth_confidence:.1%}
        return report


# --- AUGMENTATION ---
class PhysicsAwareFlipd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        cta, phys = data["cta"], data["phys"]
        if random.random() < 0.5:
            cta = torch.flip(cta, [1])
            phys = torch.flip(phys, [1])
            phys[0] *= -1
        data["cta"], data["phys"] = cta, phys
        return data

def modality_dropout(cta, phys, clin, aaa, p=0.1):
    """Randomly drops entire modalities."""
    cta_d, phys_d, clin_d, aaa_d = cta.clone(), phys.clone(), clin.clone(), aaa.clone()
    
    for b in range(cta.size(0)):
        if random.random() < p:
            cta_d[b] = torch.zeros_like(cta[b])
        if random.random() < p:
            phys_d[b] = torch.zeros_like(phys[b])
        if random.random() < p:
            clin_d[b] = torch.zeros_like(clin[b])
        if random.random() < p:
            aaa_d[b] = torch.zeros_like(aaa[b])
            
    return cta_d, phys_d, clin_d, aaa_d

# def causal_modality_dropout(
#     cta, phys, clin, aaa,
#     p_cta=0.1,
#     p_phys_given_cta=0.2,
#     p_aaa_given_cta=0.2,
#     p_aaa_given_clin=0.2
# ):
#     bsz = cta.size(0)

#     cta_mask = torch.rand(bsz, device=cta.device) > p_cta

#     cta_d = cta.clone()
#     phys_d = phys.clone()
#     clin_d = clin.clone()
#     aaa_d = aaa.clone()

#     for b in range(bsz):

#         # --- CTA missing ---
#         if not cta_mask[b]:
#             cta_d[b] = 0
#             phys_d[b] = 0  # CFD blocked
#             # AAA still possible via clinical → keep
#             continue

#         # --- CFD depends on CTA ---
#         if random.random() < p_phys_given_cta:
#             phys_d[b] = 0

#         # --- AAA depends on CTA OR clinical ---
#         aaa_drop = (
#             random.random() < p_aaa_given_cta
#         ) or (
#             random.random() < p_aaa_given_clin
#         )

#         if aaa_drop:
#             aaa_d[b] = 0

#     return cta_d, phys_d, clin_d, aaa_d

def causal_modality_dropout(
    cta, phys, clin, aaa,
    p_cta=0.1,
    p_phys_given_cta=0.2,
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

    return cta_d, phys, clin_d, aaa_d

def maybe_apply_modality_dropout(training, cta, phys, clin, aaa, p_cta, p_phys_given_cta, p_aaa_given_cta, p_aaa_given_clin):
    if not training:
        return cta, phys, clin, aaa
    return causal_modality_dropout(cta, phys, clin, aaa, p_cta, p_phys_given_cta, p_aaa_given_cta, p_aaa_given_clin)

def get_train_transforms():
    return Compose([
        PhysicsAwareFlipd(keys=["cta", "phys"]),
    ])

# --- DATA LOADING ---
# def build_data_list(ids, base_dir, config, feature_keys, sac_mean, sac_std):
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
        
        # Complication labels
        comp_labels = np.array([
            float(gt.get(comp, -1.0))  # -1 = missing, 0/1 = label
            for comp in COMPLICATIONS
        ], dtype=np.float32)
        
        # Sac regression (auxiliary)
        # sac_delta = float(gt.get("diam_diff_using_max", 0))
        # sac_expansion = float(1.0 if float(gt.get("diam_diff_using_min", 0)) >= 0 else 0.0)
        # sac_delta = float(gt.get("diam_diff_using_min", 0))
        # sac_delta_norm = (sac_delta - sac_mean) / (sac_std + 1e-8)
        # print(f"PID {pid}: sac_delta={sac_delta:.2f} mm, norm={sac_delta_norm:.3f}, expansion={sac_expansion}")
        
        data.append({
            "ct": str(base_dir / "imagesTr" / f"{pid}.nii.gz"),
            "lbl": str(base_dir / "labelsTr" / f"{pid}.nii.gz"),
            "dst": str(base_dir / "distMapsTr" / f"{pid}.nii.gz"),
            "clin": clin,
            "aaa_metrics": aaa_metrics,
            "reintervention_label": np.array([reint_label], dtype=np.float32),
            "complication_labels": comp_labels,
            # "sac_growth": np.array([sac_delta_norm], dtype=np.float32),
            # "sac_expansion": sac_expansion,
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


class LoadPhysicsd(MapTransform):
    def __init__(self, keys, base_dir):
        super().__init__(keys)
        self.base_dir = Path(base_dir)

    def __call__(self, data):
        pid = data["pid"]
        phys = self._load_vtp_as_grid(
            self.base_dir / "cfdTr" / f"{pid}.vtp"
        )
        if phys is None:
            phys = np.zeros((4, 128, 128, 128), dtype=np.float32)
        data["phys"] = torch.from_numpy(phys).float()
        return data

    def _load_vtp_as_grid(self, path):
        if not path or not path.exists():
            return None

        mesh = pv.read(str(path))
        nx, ny, nz = SIZE

        grid = pv.ImageData()
        grid.dimensions = (nx, ny, nz)
        grid.origin = (mesh.bounds[0], mesh.bounds[2], mesh.bounds[4])
        grid.spacing = (
            (mesh.bounds[1] - mesh.bounds[0]) / (nx - 1),
            (mesh.bounds[3] - mesh.bounds[2]) / (ny - 1),
            (mesh.bounds[5] - mesh.bounds[4]) / (nz - 1),
        )

        sampled = grid.sample(mesh)

        required_keys = ["tawss_true", "osi_true", "mean_wss_true", "wall_mask"]
        channels = []

        for k in required_keys:
            if k not in sampled.point_data:
                return None
            
            arr = np.asarray(sampled.point_data[k], dtype=np.float32)
            
            arr = np.nan_to_num(arr, nan=0.0)

            if k == "mean_wss_true":
                if arr.ndim == 2 and arr.shape[1] == 3:
                    if arr.shape[0] != nx * ny * nz:
                        return None
                    arr = arr.reshape(nx, ny, nz, 3)
                    arr = np.transpose(arr, (3, 0, 1, 2))
                    channels.extend(arr)
                    continue
                elif arr.ndim == 1 and arr.size == nx * ny * nz * 3:
                    arr = arr.reshape(nx, ny, nz, 3)
                    arr = np.transpose(arr, (3, 0, 1, 2))
                    channels.extend(arr)
                    continue
                else:
                    return None

            expected_size = nx * ny * nz
            if arr.size != expected_size:
                return None
            arr = arr.reshape(nx, ny, nz)
            channels.append(arr)

        phys = np.stack(channels, axis=0).astype(np.float32)
        wall = phys[-1:]
        phys[:-1] = phys[:-1] * wall

        return phys


class NormalizePhysicsd(MapTransform):
    def __init__(self, keys, eps=1e-8):
        super().__init__(keys)
        self.eps = eps

    def __call__(self, data):
        phys = data["phys"]
        mean = phys.mean(dim=(1, 2, 3), keepdim=True)
        std = phys.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        phys = (phys - mean) / (std + self.eps)
        data["phys"] = phys
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


def get_base_transforms(base_dir, clin_mean, clin_std, stage="B"):
    if stage == "A":
        return Compose([
            ToTensord(keys=["aaa_metrics"]),
            NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
        ])
    else:
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
            # ScaleIntensityRanged(keys=["ct"], a_min=-400, a_max=800, b_min=0, b_max=1, clip=False),
            Resized(
                keys=["ct", "lbl", "dst"],
                spatial_size=SIZE,
                mode=("trilinear", "nearest", "trilinear"),
            ),
            RandGaussianNoised(keys=["ct"], prob=0.3, mean=0.0, std=0.02),
            RandBiasFieldd(keys=["ct"], prob=0.2),
            RandScaleIntensityd(keys=["ct"], factors=0.1, prob=0.5),
            BuildCTAd(keys=["ct", "lbl", "dst"]),
            ToTensord(keys=["ct", "lbl", "dst", "aaa_metrics"]),
            # LoadPhysicsd(keys=["pid"], base_dir=base_dir),
            # NormalizePhysicsd(keys=["phys"]),
            NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
        ])


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
    # return torch.clamp(entropy + learned_unc, 0, 1) 

class ReinterventionRiskModel(nn.Module):
    """Multi-task model predicting reintervention risk and complications."""
    
    def __init__(self, clinical_dim=744, aaa_dim=5, num_complications=len(COMPLICATIONS)):
        super().__init__()
        
        self.num_complications = num_complications
        
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
            LoraConfig(r=16, target_modules=["query", "value", "query_global", "value_global"])
        )
        self.visual_proj = nn.Linear(2048, 256)
        
        # # Physics pathway
        # self.physics_branch = nn.Sequential(
        #     nn.Conv3d(6, 32, 3, padding=1),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(),
        #     nn.Conv3d(32, 64, 3, padding=1),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, 3, padding=1),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        #     nn.Linear(128, 128),
        #     nn.ReLU()
        # )
        
        # Clinical pathway
        # self.clinical_branch = nn.Sequential(
        #     nn.Linear(clinical_dim, 512), nn.LayerNorm(512), nn.ReLU(),
        #     nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64)
        # )
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 32)
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
        # total_fusion_dim = 256 + 128 + 64 + 32
        # total_fusion_dim = 256 + 128 + 32 + 32
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
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # logit + uncertainty
        )
        
        self.complications_head = nn.Linear(total_fusion_dim, num_complications * 2)    # Add sac-regression label
        
        # self.sac_growth_head = nn.Sequential(
        #     nn.Linear(total_fusion_dim, 128), nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 2)  # mean + log_var
        # )

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

        # --- Complications head ---
        comp_output = self.complications_head(fused)

        # reshape: (B, num_complications, 2)
        comp_output = comp_output.view(
            comp_output.size(0),
            self.num_complications,
            2
        )

        comp_logits = comp_output[:, :, 0]
        learned_comp_unc = torch.sigmoid(comp_output[:, :, 1])

        # optional: keep same uncertainty logic as full forward
        comp_unc = combined_uncertainty(learned_comp_unc, comp_logits)

        # --- SAC growth head ---
        # sac_output = self.sac_growth_head(fused)
        # sac_growth = sac_output[:, 0:1]
        # learned_sac_unc = torch.sigmoid(sac_output[:, 1:2])
        # sac_unc = learned_sac_unc

        return reint_logit, reint_logvar, comp_logits, comp_unc#, sac_growth, sac_unc
    
    def forward_from_fused_minimal(self, fused):
        reint_output = self.reintervention_head(fused)
        reint_logit = reint_output[:, 0:1]
        comp_output = self.complications_head(fused)
        comp_output = comp_output.view(comp_output.size(0), self.num_complications, 2)
        comp_logits = comp_output[:, :, 0]
        return reint_logit, comp_logits

    def forward(self, cta=None, phys=None, clin=None, aaa=None, stage="B", return_uncertainties=True, return_fused=False):
        """
        Forward pass.
        
        Returns (if return_uncertainties=True):
            reint_logit, reint_logvar, comp_logits, comp_unc, sac_growth, sac_unc
        """
        
        if stage == "A":  # Clinical only
            c_emb = self.clinical_branch(clin)
            a_emb = self.aaa_metrics_branch(aaa)
            combined = torch.cat([c_emb, a_emb], dim=1)
            output = self.clinical_head(combined)

            if return_uncertainties:
                reint_logit = output[:, 0:1]
                reint_logvar = output[:, 1:2]

                comp_logits = output[:, 2:]

                # dummy values for unused heads
                comp_unc = torch.zeros_like(comp_logits)
                # sac_growth = torch.zeros_like(reint_logit)
                # sac_unc = torch.zeros_like(reint_logit)

                return reint_logit, reint_logvar, comp_logits, comp_unc#, sac_growth, sac_unc
            else:
                reint_logit = output[:, 0:1]
                comp_logits = output[:, 2:]
                return reint_logit, comp_logits
        
        # Stage B/C: Full multimodal
        cta = self.cta_adapter(cta)
        v_out = self.encode_visual(cta)
        if isinstance(v_out, (tuple, list)):
            v_out = v_out[0]
        v_emb = v_out.flatten(1)
        v_emb = self.visual_proj(v_emb)
        
        # p_emb = self.physics_branch(phys)
        c_emb = self.clinical_branch(clin)
        a_emb = self.aaa_metrics_branch(aaa)
        
        # Fusion
        # fused = torch.cat([v_emb, p_emb, c_emb, a_emb], dim=1)
        fused = torch.cat([v_emb, c_emb, a_emb], dim=1)
        # cond = self.condition_net(fused)
        cond = torch.tanh(self.condition_net(fused)) * 0.1
        fused = self.film_modulate(fused, cond)
        fused = self.fusion_norm(fused)
        if return_fused:
            return fused
        
        # Outputs
        reint_output = self.reintervention_head(fused)  
        reint_logit = reint_output[:, 0:1]
        reint_logvar = reint_output[:, 1:2]
        reint_logvar = torch.clamp(reint_logvar, -5, 5)

        comp_output = self.complications_head(fused)  
        # (B, num_complications * 2)

        comp_output = comp_output.view(
            comp_output.size(0),
            self.num_complications,
            2
        )

        comp_logits = comp_output[:, :, 0]
        learned_comp_unc = torch.sigmoid(comp_output[:, :, 1])

        comp_unc = combined_uncertainty(learned_comp_unc, comp_logits)#.mean(dim=1)

        # sac_output = self.sac_growth_head(fused)
        # sac_growth = sac_output[:, 0:1]
        # learned_sac_unc = torch.sigmoid(sac_output[:, 1:2])
        # sac_unc = learned_sac_unc  # or proper regression variance if you upgrade
                
        if return_uncertainties:
            return reint_logit, reint_logvar, comp_logits, comp_unc#, sac_growth, sac_unc
        else:
            return reint_logit, comp_logits

def compute_pos_weight(dataset):
    all_labels = []

    for item in dataset:
        comp = item["complication_labels"]
        all_labels.append(comp)

    all_labels = np.stack(all_labels)
    mask = all_labels != -1
    pos = ((all_labels == 1) & mask).sum(axis=0)
    neg = ((all_labels == 0) & mask).sum(axis=0)

    pos_weight = neg / (pos + 1e-6)

    return torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

class StableFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        """
        targets: (B, C) with {0,1,-1}
        """

        mask = (targets != -1).float()
        targets_clamped = targets.clamp(0, 1)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_clamped, reduction="none"
        )

        probs = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)

        pt = torch.where(
            targets_clamped == 1,
            probs,
            1 - probs
        )

        focal = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets_clamped + (1 - self.alpha) * (1 - targets_clamped)

        loss = alpha_t * focal * bce
        loss = loss * mask

        denom = mask.sum().clamp_min(1.0)

        return loss.sum() / denom
    
class ReinterventionLoss(nn.Module):
    def __init__(self, reint_weight=3.0, comp_weight=1.5):
    # def __init__(self, reint_weight=3.0, comp_weight=1.5, sac_weight=0.5):
        super().__init__()

        self.reint_weight = reint_weight
        self.comp_weight = comp_weight
        # self.sac_weight = sac_weight

        # use ONLY focal loss (cleaner + better for rare complications)
        self.comp_criterion = StableFocalLoss(alpha=0.25, gamma=2.0)

        # self.sac_criterion = nn.HuberLoss(delta=1.0)

    def forward(
        self,
        reint_logit, reint_logvar,
        comp_logits, comp_unc,
        # sac_growth,
        reint_gt, comp_gt,# sac_gt,
        stage="B"
    ):

        # -------------------------
        # 1. Reintervention (heteroscedastic NLL)
        # -------------------------
        reint_gt = reint_gt.view_as(reint_logit)

        nll = F.binary_cross_entropy_with_logits(
            reint_logit, reint_gt, reduction="none"
        )

        reint_logvar = torch.clamp(reint_logvar, -5, 5)

        loss_reint = (nll * torch.exp(-reint_logvar) + reint_logvar).mean()

        # -------------------------
        # 2. Complications (FOCAL LOSS)
        # -------------------------
        loss_comp = self.comp_criterion(comp_logits, comp_gt)

        # -------------------------
        # 3. Sac growth
        # -------------------------
        # loss_sac = self.sac_criterion(sac_growth, sac_gt)

        # -------------------------
        # 4. Comp uncertainty alignment
        # -------------------------
        mask = (comp_gt != -1).float()

        pred_prob = torch.sigmoid(comp_logits).detach()
        gt_clean = comp_gt.clamp(0, 1)

        comp_error = (pred_prob - gt_clean).abs()

        comp_unc = comp_unc.view_as(comp_logits)

        loss_comp_unc = ((comp_unc - comp_error) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)

        # -------------------------
        # 5. Stage weighting
        # -------------------------
        if stage == "A":
            total = (
                self.reint_weight * loss_reint +
                self.comp_weight * loss_comp
            )
        else:
            total = (
                self.reint_weight * loss_reint +
                self.comp_weight * loss_comp +
                # self.sac_weight * loss_sac +
                0.1 * loss_comp_unc
            )

        return total, {
            "loss_total": total.item(),
            "loss_reint": loss_reint.item(),
            "loss_comp": loss_comp.item(),
            # "loss_sac": loss_sac.item(),
            "loss_comp_unc": loss_comp_unc.item(),
        }

# --- XAI SYSTEM ---
class ReinterventionXAI:
    """Advanced XAI for reintervention risk prediction."""
    
    # def __init__(self, model, feature_names, complications, sac_mean, sac_std):
    def __init__(self, model, feature_names, complications):
        self.model = model
        self.feature_names = np.array(feature_names)
        self.complications = complications
        # self.sac_mean = sac_mean
        # self.sac_std = sac_std
    
    def explain(self, cta, phys, clin, aaa) -> ReinterventionRiskPrediction:
        """Generate comprehensive explanation for reintervention prediction."""
        self.model.eval()
        
        with torch.no_grad():
            # reint_logit, reint_logvar, comp_logits, comp_unc, sac_growth, sac_unc = \
            reint_logit, reint_logvar, comp_logits, comp_unc = \
                self.model(cta, phys, clin, aaa, return_uncertainties=True)
        
        # Convert to probabilities
        reint_prob = torch.sigmoid(reint_logit).item()
        comp_probs = torch.sigmoid(comp_logits).squeeze().cpu().detach().numpy()
        
        # Confidence levels
        reint_std = torch.exp(0.5 * reint_logvar)
        reint_conf = 1.0 / (1.0 + reint_std.item())
        conf_level = self._get_confidence_level(reint_conf)
        
        comp_confidence = self._get_complication_confidences(comp_probs, comp_unc)
        
        # Feature importance
        clin_imp = self._clinical_importance(cta, phys, clin, aaa)
        modality_imp = self._modality_importance(cta, phys, clin, aaa)
        
        # Complication analysis
        comp_drivers = self._complication_drivers(cta, phys, clin, aaa, comp_logits)
        comp_influence = self._complication_influence(cta, phys, clin, aaa)
        
        # Causal chain
        causal_chain = self._generate_causal_chain(
            reint_prob, comp_probs, clin_imp, modality_imp
        )
        primary_factor = self._identify_primary_factor(clin_imp, comp_influence)
        
        return ReinterventionRiskPrediction(
            reintervention_prob=reint_prob,
            reintervention_confidence=reint_conf,
            confidence_level=conf_level,
            complication_probs={
                self.complications[i]: float(comp_probs[i])
                for i in range(len(self.complications))
            },
            complication_confidence=comp_confidence,
            clinical_importance=clin_imp,
            modality_importance=modality_imp,
            complication_drivers=comp_drivers,
            complication_influence_on_reint=comp_influence,
            causal_chain=causal_chain,
            primary_risk_factor=primary_factor,
            # sac_growth_mm=denorm(sac_growth, self.sac_mean, self.sac_std).item(),
            # sac_growth_confidence=1.0 - sac_unc.item(),
        )
    
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
    
    def _clinical_importance(self, cta, phys, clin, aaa):
        clin_var = clin.clone().detach().requires_grad_(True)

        # reint_logit, _, _, _, _, _ = self.model(
        reint_logit, *_ = self.model(
            cta, phys, clin_var, aaa, return_uncertainties=True
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
    
    def _modality_importance(self, cta, phys, clin, aaa) -> Dict[str, float]:
        """Compute importance of each modality."""
        self.model.eval()
        
        # Ablation approach
        with torch.no_grad():
            reint_logit_full, *_ = self.model(cta, phys, clin, aaa)
            reint_full = torch.sigmoid(reint_logit_full).item()
        
        # Zero each modality and measure impact
        impacts = {}
        
        with torch.no_grad():
            cta_zero = torch.zeros_like(cta)
            reint_no_cta, *_ = self.model(cta_zero, phys, clin, aaa)
            impacts['CTA'] = abs(reint_full - torch.sigmoid(reint_no_cta).item())
            
            # phys_zero = torch.zeros_like(phys)
            # reint_no_phys, _, _, _, _, _ = self.model(cta, phys_zero, clin, aaa)
            # impacts['CFD'] = abs(reint_full - torch.sigmoid(reint_no_phys).item())
            
            clin_zero = torch.zeros_like(clin)
            reint_no_clin, *_ = self.model(cta, phys, clin_zero, aaa)
            impacts['Clinical'] = abs(reint_full - torch.sigmoid(reint_no_clin).item())
            
            aaa_zero = torch.zeros_like(aaa)
            reint_no_aaa, *_ = self.model(cta, phys, clin, aaa_zero)
            impacts['AAA_metrics'] = abs(reint_full - torch.sigmoid(reint_no_aaa).item())
        
        # Normalize
        total = sum(impacts.values())
        if total > 0:
            impacts = {k: v / total for k, v in impacts.items()}
        
        return impacts
    
    def _complication_drivers(self, cta, phys, clin, aaa, comp_logits) -> Dict[str, List]:
        """Find which features drive each complication."""
        result = {}
        
        for comp_idx, comp_name in enumerate(self.complications):
            clin_var = clin.clone().detach().requires_grad_(True)
            
            with torch.enable_grad():
                _, _, comp_logits_var, *_ = self.model(cta, phys, clin_var, aaa)
                loss = torch.sigmoid(comp_logits_var[:, comp_idx]).sum()
                self.model.zero_grad(set_to_none=True)
                loss.backward()
            
            grads = clin_var.grad.abs().squeeze().cpu().detach().numpy()
            top_indices = np.argsort(grads)[::-1][:5]
            
            result[comp_name] = [
                (self.feature_names[i], float(grads[i]))
                for i in top_indices
            ]
        
        return result
    
    def _complication_influence(self, cta, phys, clin, aaa):
        self.model.eval()

        with torch.enable_grad():

            # 1. forward full model
            fused = self.model(cta, phys, clin, aaa, return_fused=True)
            fused = fused.requires_grad_(True)

            # 2. compute outputs from fused
            reint_logit, comp_logits = self.model.forward_from_fused_minimal(fused)

            # 3. define scalar target
            reint = reint_logit.sum()

            # 4. gradient wrt fused (NOT comp_logits)
            grads = torch.autograd.grad(
                outputs=reint,
                inputs=fused,
                retain_graph=True,
                create_graph=False
            )[0]

        # 5. weight by complication probabilities
        comp_probs = torch.sigmoid(comp_logits).detach()

        # collapse fused → importance proxy per comp via projection
        influence = (grads.mean(dim=1, keepdim=True) * comp_probs).abs()

        influence = influence.squeeze().cpu().numpy()

        return {
            self.complications[i]: float(influence[i])
            for i in range(len(self.complications))
        }
        
    def _generate_causal_chain(self, reint_prob, comp_probs, clin_imp, modality_imp) -> str:
        """Generate natural language causal explanation."""
        
        # Find dominant complication
        top_comp_idx = np.argmax(comp_probs)
        top_comp = self.complications[top_comp_idx]
        top_comp_prob = comp_probs[top_comp_idx]
        
        # Find dominant clinical feature
        top_clin_feat = clin_imp[0][0] if clin_imp else "unknown factors"
        
        # Find dominant modality
        top_modality = max(modality_imp, key=modality_imp.get)
        
        # Generate explanation
        if reint_prob > 0.5:
            chain = f"Elevated {top_clin_feat} detected by {top_modality} → High risk of {top_comp} ({top_comp_prob:.0%}) → Likely reintervention needed"
        elif top_comp_prob > 0.3:
            chain = f"Possible {top_comp} ({top_comp_prob:.0%}) detected via {top_modality} → Moderate reintervention risk ({reint_prob:.0%})"
        else:
            chain = f"Clinical factors ({top_clin_feat}) suggest monitoring advised, low complication risk"
        
        return chain
    
    def _identify_primary_factor(self, clin_imp: List, comp_influence: Dict) -> str:
        """Identify the single most important factor."""
        if not clin_imp:
            return "Unable to determine"
        
        top_feat = clin_imp[0][0]
        impact_level = clin_imp[0][2]
        
        # Check if complications are dominant
        if comp_influence:
            top_comp = max(comp_influence, key=comp_influence.get)
            comp_score = comp_influence[top_comp]
            
            if comp_score > 0.3:
                return f"High risk of {top_comp}"
        
        return f"{top_feat} ({impact_level} impact)"


# --- UTILITIES ---
# def validate(model, val_loader, criterion, sac_mean, sac_std, stage="B"):
def validate(model, val_loader, criterion, stage="B"):
    model.eval()
    val_loss = 0
    
    # Storage for metrics
    all_reint_probs, all_reint_gt = [], []
    all_comp_probs, all_comp_gt = [], []
    # all_sac_preds, all_sac_gt = [], []

    with torch.no_grad():
        for batch in val_loader:
            # Setup inputs
            cta = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            phys = batch.get("phys", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa = batch["aaa_metrics"].to(DEVICE).float()
            
            reint_gt = batch["reintervention_label"].to(DEVICE).float()
            comp_gt = batch["complication_labels"].to(DEVICE).float()
            # sac_gt = batch["sac_growth"].to(DEVICE).float()

            # 1. Capture ALL outputs, including uncertainties
            # reint_logit, reint_logvar, comp_logits, comp_unc, sac_growth, sac_unc = model(
            reint_logit, reint_logvar, comp_logits, comp_unc = model(
                cta, phys, clin, aaa, return_uncertainties=True
            )
            
            # 2. Pass the uncertainty tensors to the criterion instead of None
            loss, loss_dict = criterion(
                reint_logit, reint_logvar, 
                comp_logits, comp_unc, 
                # sac_growth, 
                # reint_gt, comp_gt, sac_gt, stage=stage
                reint_gt, comp_gt, stage=stage
            )
            
            val_loss += loss.item()

            # Collect for sklearn metrics
            all_reint_probs.append(torch.sigmoid(reint_logit).cpu())
            all_reint_gt.append(reint_gt.cpu())
            all_comp_probs.append(torch.sigmoid(comp_logits).cpu())
            all_comp_gt.append(comp_gt.cpu())
            # all_sac_preds.append(sac_growth.cpu())
            # all_sac_gt.append(sac_gt.cpu())

    # Concatenate all results
    reint_p = torch.cat(all_reint_probs).numpy()
    reint_g = torch.cat(all_reint_gt).numpy()
    comp_p = torch.cat(all_comp_probs).numpy()
    comp_g = torch.cat(all_comp_gt).numpy()
    # sac_p = denorm(torch.cat(all_sac_preds), sac_mean, sac_std).numpy()
    # sac_g = denorm(torch.cat(all_sac_gt), sac_mean, sac_std).numpy()

    # Calculate final stats
    metrics = {
        "val/loss_total": val_loss / len(val_loader),
        "val/reint_auc": roc_auc_score(reint_g, reint_p) if len(np.unique(reint_g)) > 1 else 0.5,
        "val/reint_f1": f1_score(reint_g, (reint_p > 0.5).astype(int)),
        # "val/sac_mae": np.abs(sac_p - sac_g).mean(),
        "val/comp_mAP": average_precision_score(comp_g, comp_p, average='macro')
    }
    
    # Log per-complication AUC (only if there are positive samples in the val set)
    for i, name in enumerate(COMPLICATIONS):
        if len(np.unique(comp_g[:, i])) > 1:
            valid_mask = comp_g[:, i] != -1
            y_true = comp_g[valid_mask, i]
            y_pred = comp_p[valid_mask, i]

            if len(np.unique(y_true)) > 1:
                metrics[f"val/auc_{name}"] = roc_auc_score(y_true, y_pred)

    wandb.log(metrics)
    return val_loss / len(val_loader)

def get_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def set_stage_a(model):
    """Freeze all except clinical branches."""
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.clinical_branch.parameters():
        p.requires_grad = True
    for p in model.aaa_metrics_branch.parameters():
        p.requires_grad = True
    for p in model.clinical_head.parameters():
        p.requires_grad = True


def set_stage_b(model):
    """Freeze clinical, train physics and visual."""
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False
    # for p in model.physics_branch.parameters():
    #     p.requires_grad = True
    for p in model.reintervention_head.parameters():
        p.requires_grad = True
    for p in model.complications_head.parameters():
        p.requires_grad = True
    # for p in model.sac_growth_head.parameters():
    #     p.requires_grad = True
    for p in model.condition_net.parameters():
        p.requires_grad = True
    for name, p in model.visual_branch.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True
    model.visual_proj.requires_grad_(True)


def set_stage_c(model):
    """Train everything except Merlin."""
    model = get_model(model)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.clinical_branch.parameters():
        p.requires_grad = True
    for p in model.aaa_metrics_branch.parameters():
        p.requires_grad = True
    # for p in model.physics_branch.parameters():
    #     p.requires_grad = True
    for p in model.reintervention_head.parameters():
        p.requires_grad = True
    for p in model.complications_head.parameters():
        p.requires_grad = True
    # for p in model.sac_growth_head.parameters():
    #     p.requires_grad = True
    for p in model.condition_net.parameters():
        p.requires_grad = True
    for name, p in model.visual_branch.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True
    model.visual_proj.requires_grad_(True)


def build_optimizer(model, stage, lr_dict=None):
    """Build optimizer with stage-specific learning rates."""
    model = get_model(model)
    if stage == "A":
        param_groups = [
            {"params": model.clinical_branch.parameters(), "lr": 3e-3},
            {"params": model.aaa_metrics_branch.parameters(), "lr": 3e-3},
            {"params": model.clinical_head.parameters(), "lr": 3e-3},
        ]
    elif stage == "B":
        param_groups = [
            # {"params": model.physics_branch.parameters(), "lr": 1e-4},
            {"params": model.reintervention_head.parameters(), "lr": 1e-4},
            {"params": model.complications_head.parameters(), "lr": 1e-4},
            # {"params": model.sac_growth_head.parameters(), "lr": 1e-4},
            {"params": model.condition_net.parameters(), "lr": 1e-4},
            {"params": model.visual_branch.parameters(), "lr": 1e-5},
        ]
    elif stage == "C":
        param_groups = [
            {"params": model.clinical_branch.parameters(), "lr": 5e-5},
            {"params": model.aaa_metrics_branch.parameters(), "lr": 5e-5},
            # {"params": model.physics_branch.parameters(), "lr": 1e-4},
            {"params": model.reintervention_head.parameters(), "lr": 5e-5},
            {"params": model.complications_head.parameters(), "lr": 5e-5},
            # {"params": model.sac_growth_head.parameters(), "lr": 5e-5},
            {"params": model.condition_net.parameters(), "lr": 5e-5},
            {"params": model.visual_branch.parameters(), "lr": 1e-5},
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

def print_trainable_params(model, text="Trainable params:"):
    total = 0
    trainable = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"{text} {trainable} / {total} ({100*trainable/total:.2f}%)")

def print_model_param_overview(model):
    model = get_model(model)
    print_trainable_params(model.clinical_branch, "Trainable params (clinical 1/3):")
    print_trainable_params(model.aaa_metrics_branch, "Trainable params (clinical 2/3):")
    print_trainable_params(model.clinical_head, "Trainable params (clinical 3/3):")
    print_trainable_params(model.visual_branch, "Trainable params (visual 1/2):")
    print_trainable_params(model.visual_proj, "Trainable params (visual 2/2):")
    # print_trainable_params(model.physics_branch, "Trainable params (physical):")
    print_trainable_params(model.condition_net, "Trainable params (condition):")
    print_trainable_params(model.reintervention_head, "Trainable params (reintervention):")
    print_trainable_params(model.complications_head, "Trainable params (complications):")
    # print_trainable_params(model.sac_growth_head, "Trainable params (SAC growth):")

# --- ADD THESE HELPER LOGGING FUNCTIONS ---

def log_detailed_metrics(stage, epoch, train_loss, val_loss, loss_dict, opt):
    """Logs learning rates and split losses."""
    log_data = {
        f"stage_{stage}/total_val_loss": val_loss,
        f"stage_{stage}/train_loss_step": train_loss,
        "epoch": epoch,
    }
    # Add individual task losses from the criterion's dict
    for k, v in loss_dict.items():
        log_data[f"metrics/{k}"] = v
        
    # Log Learning Rates for all param groups
    for i, group in enumerate(opt.param_groups):
        name = group.get("name", f"group_{i}")
        log_data[f"lr/{name}"] = group["lr"]
        
    wandb.log(log_data)

# def log_val_table(model, val_loader, epoch, stage, sac_mean, sac_std):
def log_val_table(model, val_loader, epoch, stage):
    """Generates a WandB table with EVERY complication pred/GT for all cases."""
    model.eval()
    
    # 1. Dynamically create column headers
    # Format: Case_ID, Reint_Prob, Reint_GT, Sac_Pred, Sac_GT, [Comp1_P, Comp1_GT, Comp2_P, Comp2_GT...]
    # columns = ["Epoch", "Case_ID", "Reint_Prob", "Reint_GT", "Sac_Pred_mm", "Sac_GT_mm"]
    columns = ["Epoch", "Case_ID", "Reint_Prob", "Reint_GT"]
    for comp in COMPLICATIONS:
        columns.append(f"P_{comp}")
        columns.append(f"GT_{comp}")
    
    table = wandb.Table(columns=columns)
    
    case_counter = 0
    with torch.no_grad():
        for batch in val_loader:
            cta = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            phys = batch.get("phys", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
            clin = batch["clin"].to(DEVICE).float()
            aaa = batch["aaa_metrics"].to(DEVICE).float()
            
            reint_gt = batch["reintervention_label"].cpu().numpy()
            # sac_gt = batch["sac_growth"].cpu().numpy()
            comp_gt = batch["complication_labels"].cpu().numpy()
            
            # Forward Pass
            # reint_logit, _, comp_logits, _, sac_growth, _ = model(
            reint_logit, _, comp_logits, _ = model(
                cta = cta if stage != "A" else torch.zeros(BATCH_SIZE, 4, *SIZE).to(DEVICE),
                phys = phys if stage != "A" else torch.zeros(BATCH_SIZE, 4, *SIZE).to(DEVICE),
                clin=clin, aaa=aaa, stage=stage
            )
            
            probs = torch.sigmoid(reint_logit).cpu().numpy().flatten()
            comp_probs = torch.sigmoid(comp_logits).cpu().numpy() # Shape: [Batch, 8]
            # sac_preds = denorm(sac_growth, sac_mean, sac_std).cpu().numpy().flatten()
            # sac_gt = denorm(batch["sac_growth"], sac_mean, sac_std).cpu().numpy().flatten()

            for i in range(len(probs)):
                # Start the row with the core metrics
                row_data = [
                    epoch,
                    case_counter,
                    round(float(probs[i]), 4),
                    int(reint_gt[i]),
                    # round(float(sac_preds[i]), 2),
                    # round(float(sac_gt[i]), 2)
                ]
                
                # Append every complication pair
                for j in range(len(COMPLICATIONS)):
                    p_val = round(float(comp_probs[i][j]), 4)
                    gt_val = int(comp_gt[i][j])
                    row_data.append(p_val)
                    row_data.append(gt_val)
                
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
    
    feature_names = list(config[list(config.keys())[0]]['pre_op_features'].keys())
    
    # Clean features
    for pid in config.keys():
        if "pre_op_features" in config[pid]:
            config[pid]["pre_op_features"] = clean_clinical_features(
                config[pid]["pre_op_features"]
            )
    
    # Compute statistics
    # all_sac = [
    #     float(config[pid]["ground_truth"].get("AAArem_con_diamdiff_baseline465p", 0))
    #     for pid in splits['train']
    # ]
    # SAC_MEAN = np.mean(all_sac)
    # SAC_STD = np.std(all_sac) + 1e-8
    
    clin_matrix = np.stack([
        np.array([config[pid]["pre_op_features"].get(k, 0.0) for k in feature_names], dtype=np.float32)
        for pid in splits["train"]
    ])
    CLIN_MEAN = clin_matrix.mean(axis=0)
    CLIN_STD = clin_matrix.std(axis=0) + 1e-8
    
    # Build datasets
    # train_files = build_data_list(splits["train"], BASE_DIR, config, feature_names, SAC_MEAN, SAC_STD)
    # val_files = build_data_list(splits["val"], BASE_DIR, config, feature_names, SAC_MEAN, SAC_STD)
    train_files = build_data_list(splits["train"], BASE_DIR, config, feature_names)
    val_files = build_data_list(splits["val"], BASE_DIR, config, feature_names)
    
    train_ds_a = PersistentDataset(
        data=train_files,
        transform=get_base_transforms(BASE_DIR, CLIN_MEAN, CLIN_STD, stage="A"),
        cache_dir=str(CACHE_DIR / "train_A")
    )
    
    train_ds = PersistentDataset(
        data=train_files,
        transform=get_base_transforms(BASE_DIR, CLIN_MEAN, CLIN_STD, stage="B"),
        cache_dir=str(CACHE_DIR / "train_B")
    )
    
    val_ds = PersistentDataset(
        data=val_files,
        transform=get_base_transforms(BASE_DIR, CLIN_MEAN, CLIN_STD, stage="B"),
        cache_dir=str(CACHE_DIR / "val")
    )

    train_loader_a = DataLoader(train_ds_a, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
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
    # xai = ReinterventionXAI(model, feature_names, COMPLICATIONS, SAC_MEAN, SAC_STD)
    xai = ReinterventionXAI(model, feature_names, COMPLICATIONS)
    
    # ---------------------------------------------------------
    # STAGES LOOP (A, B, then C)
    # ---------------------------------------------------------
    stages = [
        # {"name": "A", "epochs": 1, "patience": 10},
        # {"name": "B", "epochs": 1, "patience": 5},
        # {"name": "C", "epochs": 1, "patience": 20}

        {"name": "A", "epochs": 40, "patience": 10},
        {"name": "B", "epochs": 20, "patience": 5},
        {"name": "C", "epochs": 120, "patience": 20}

        # {"name": "A", "epochs": 40, "patience": 40},
        # {"name": "B", "epochs": 20, "patience": 20},
        # {"name": "C", "epochs": 120, "patience": 120}
    ]

    for stage_cfg in stages:
        st = stage_cfg["name"]
        print(f"\n>>> STARTING STAGE {st}")
        
        if st == "A": set_stage_a(model)
        elif st == "B": 
            model.load_state_dict(torch.load(save_dir / "best_stage_a.pt"))
            set_stage_b(model)
        elif st == "C":
            model.load_state_dict(torch.load(save_dir / "best_stage_b.pt"))
            set_stage_c(model)

        opt = build_optimizer(model, st)
        best_val = float("inf")
        patience_counter = 0

        scaler = torch.cuda.amp.GradScaler()
        print_model_param_overview(model)

        model = model.to(DEVICE)
        model = model.to(memory_format=torch.channels_last_3d)
        
        for epoch in range(stage_cfg["epochs"]):
            model.train()
            train_loss = 0
            last_loss_dict = {}

            pbar = tqdm(train_loader if st != "A" else train_loader_a, desc=f"Stage {st} Epoch {epoch}")
            for i, batch in enumerate(pbar):
                # Standard training step...
                clin = batch["clin"].to(DEVICE).float()
                aaa = batch["aaa_metrics"].to(DEVICE).float()
                cta = batch.get("cta", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()
                phys = batch.get("phys", torch.zeros(BATCH_SIZE, 4, *SIZE)).to(DEVICE).float()

                if st != "A":
                    cta, phys, clin, aaa = maybe_apply_modality_dropout(
                        model.training, cta, phys, clin, aaa,
                        p_cta=0.1 if st == "B" else 0.05,
                        p_phys_given_cta = 0.15,
                        p_aaa_given_cta = 0.15,
                        p_aaa_given_clin = 0.15,
                    )
                
                reint_gt = batch["reintervention_label"].to(DEVICE).float()
                comp_gt = batch["complication_labels"].to(DEVICE).float()
                # sac_gt = batch["sac_growth"].to(DEVICE).float()

                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    # reint_logit, reint_logvar, comp_logits, comp_unc, sac_growth, _ = model(
                    reint_logit, reint_logvar, comp_logits, comp_unc = model(
                        cta = cta if st != "A" else torch.zeros(BATCH_SIZE, 4, *SIZE).to(DEVICE),
                        phys = phys if st != "A" else torch.zeros(BATCH_SIZE, 4, *SIZE).to(DEVICE),
                        clin=clin,
                        aaa=aaa,
                        stage=st
                    )

                    loss, last_loss_dict = criterion(
                        reint_logit, reint_logvar,
                        comp_logits, comp_unc,
                        # sac_growth,
                        reint_gt, comp_gt,# sac_gt,
                        stage=st
                    )

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                train_loss += loss.item()

            # --- VALIDATION & LOGGING ---
            # val_loss = validate(model, val_loader, criterion, SAC_MEAN, SAC_STD, stage=st)
            val_loss = validate(model, val_loader, criterion, stage=st)
            
            # 1. Log numerical metrics
            log_detailed_metrics(st, epoch, train_loss/len(pbar), val_loss, last_loss_dict, opt)

            # 2. Log Table & XAI every 10 epochs
            if (epoch + 1) % 10 == 0:
                # log_val_table(model, val_loader, epoch, st, SAC_MEAN, SAC_STD)
                log_val_table(model, val_loader, epoch, st)
                
                # Sample XAI for the first case in the val_loader
                sample = next(iter(val_loader))
                expl = xai.explain(
                    sample.get("cta", torch.zeros(1))[:1].to(DEVICE), 
                    sample.get("phys", torch.zeros(1))[:1].to(DEVICE), 
                    sample["clin"][:1].to(DEVICE), 
                    sample["aaa_metrics"][:1].to(DEVICE)
                )
                wandb.log({
                    f"xai_stage_{st}/causal_chain": wandb.Html(f"<b>Causal Chain:</b> <p>{expl.causal_chain}</p>"),
                    f"xai_stage_{st}/reint_prob": expl.reintervention_prob,
                    f"xai_stage_{st}/conf_level": expl.confidence_level,
                    f"xai_stage_{st}/primary_factor": expl.primary_risk_factor,
                    # f"xai_stage_{st}/sac_growth": expl.sac_growth_mm,
                    # Log top 3 clinical features as a summary string
                    f"xai_stage_{st}/top_features": ", ".join([f"{f[0]} ({f[2]})" for f in expl.clinical_importance[:3]])
                })
                # Print the formatted report to console for immediate feedback
                print(expl.format_report())

            # Early Stopping Logic
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_dir / f"best_stage_{st.lower()}.pt")
            else:
                patience_counter += 1
                if patience_counter >= stage_cfg["patience"]:
                    break

    wandb.finish()


if __name__ == "__main__":
    run_pipeline()