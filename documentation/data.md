# Data

## Deepstore Layout

All processed data lives under:

```
/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1/
├── imagesTr/           # CTA volumes (.nii.gz), one file per patient
├── labelsTr/           # nnUNet segmentation masks (.nii.gz)
│                       # label 1 = aortic lumen, label 2 = intraluminal thrombus (ILT)
├── distMapsTr/         # Distance maps from lumen surface (.nii.gz)
├── cfdTr/              # CFD hemodynamics (.vtp meshes, optional)
├── metricsTr/          # AAA geometry metrics (.json, one file per patient)
└── clinical_data/
    ├── radar_multimodal_config_v1.json   # per-patient clinical features + labels
    └── dataset_splits.json               # stratified train / val / test split
```

Raw source data is under:
```
/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/
```

---

## Clinical Data — `radar_multimodal_config_v1.json`

Each patient is stored as a top-level key `<patient_id>_<site>` (e.g. `0042_RNST`). The structure:

```json
{
  "0042_RNST": {
    "metadata": {
      "id": "0042",
      "original_nr": 42.0,
      "site": "001 lokatie Arnhem",
      "site_short": "RNST",
      "img": "0042_RNST.nii.gz",
      "cfd": "0042_RNST.vtp"          // null if no CFD available
    },
    "paths": {
      "pre_op_cta": "/deepstore/.../imagesTr_pre-op/0042_RNST.nii.gz",
      "pre_op_cfd": "/deepstore/.../CFDs/renamed/0042_RNST.vtp"
    },
    "pre_op_features": {
      "Bl_AAAdiam": 52.0,
      "Bl_NeckLen": 23.0,
      // ... all Bl_* RADAR baseline columns
    },
    "post_op_features": {
      // all non-baseline RADAR columns
    },
    "ground_truth": {
      "reintervention":  1,
      "death":           0,
      "death_surg":      0,
      "death_hosp":      0,
      "sac_regression":  0,
      "sac_stable":      1,
      "sac_expansion":   0,
      "diam_diff":       2.5,          // follow-up minus baseline diameter (mm); null if missing
      "endoleak_1a":     0,
      "endoleak_1b":     0,
      "endoleak_2":      1,
      "endoleak_3":      0,
      "endoleak_5":      0,
      "occlusion":       0,
      "stenosis":        0,
      "kinking":         0,
      "fracture":        0,
      "migration":       0,
      "infection":       0
    }
  }
}
```

### Clinical Feature Preprocessing

Raw RADAR fields are cleaned by `_clean_clinical_features` before any statistics are computed or the model sees them:

| Input value | Treatment | Reason |
|---|---|---|
| `999`, `9999`, `99999` (any numeric type or string) | → `NaN` | RADAR sentinel codes for "not measured / not applicable". Feeding them to `compute_clin_stats` inflates the mean to ~8000 for affected features and destroys z-normalisation. |
| `None` / empty string | → `NaN` | Missing record — imputed to training mean after normalisation. |
| Non-numeric string (e.g. `"2010-12-07"`) | → `NaN` | Date fields and free text. Previously silently became `0.0`. |
| `PatientNr` key | Excluded entirely | Database ID (up to ~1e8). Has no clinical meaning and is dropped from `feature_names` so it never enters the feature vector. |
| Any other float | Passed through | Real measurement. |

**Missing-value imputation flow:**

```
raw 9999  →  _clean_clinical_features  →  NaN
                                             ↓
                               build_data_list (NaN preserved)
                                             ↓
                              NormalizeClinicald: z = (NaN − mean) / std  →  NaN
                                             ↓
                              torch.nan_to_num(nan=0.0)  →  z = 0.0 = training mean
```

Imputing to z=0 (the training mean) is the most agnostic choice: the feature carries no signal for this patient, and the L1 feature gate can cheaply zero it out without penalising the model.

**Statistics are computed on observed values only** (`np.nanmean` / `np.nanstd`). Features that are 100% missing in the training split receive `mean=0, std=1` so they always normalise to z=0 and add no gradient.

Of the 240 raw features (after dropping `PatientNr`), 198 (82%) are ≥50% missing and 148 (62%) are ≥90% missing in training patients. Only **42 features have <50% missingness** and carry real clinical signal (age, BMI, ASA score, smoking history, AAA diameter, cardiac/renal comorbidities, vascular anatomy flags).

**Missingness filter:** `filter_features_by_missingness()` (`src/data/loaders.py`) is called during training to drop features above the configured threshold before `compute_clin_stats` runs. This is controlled by `training.max_missing_pct` in the YAML (default 0.5). The filter is computed on the training split only. The resulting `feature_names` list (~42 entries) is stored in the checkpoint, so inference automatically uses the same subset — no YAML changes needed at test time.

### Ground Truth Notes

- **`sac_regression`** — 1 if max aortic diameter decreased ≥ 5 mm vs. baseline
- **`sac_expansion`** — 1 if max aortic diameter increased ≥ 5 mm vs. baseline
- **`sac_stable`** — 1 if |change| < 5 mm
- **`diam_diff`** — raw difference (follow-up − baseline, in mm); can be null if either measurement is absent
- Diameter pairs are selected in order of preference: CT-CT (preferred) → US-US → cross-modality (CT-US or US-CT) → GEN-* fallback (generic `Bl_AAAdiam` column for older Rijnstate patients where no modality-specific baseline exists)

---

## Dataset Splits — `dataset_splits.json`

Stratified split by re-intervention label and site. Format:

```json
{
  "train": ["0042_RNST", "0055_MST", ...],
  "val":   ["0017_AUMC", ...],
  "test":  ["0031_UMCG", ...]
}
```

Split is generated by `preprocessing/case_matching/07_generate_data_split.py` and should not be regenerated unless the dataset changes, to avoid test set contamination.

---

## AAA Geometry Metrics — `metricsTr/<patient_id>.json`

```json
{
  "volume_lumen_ml":            45.3,
  "volume_ilt_ml":              12.1,
  "volume_total_ml":            57.4,
  "max_lumen_to_wall_dist_mm":  28.6,
  "avg_lumen_to_wall_dist_mm":  14.2
}
```

These are extracted from nnUNet segmentation masks by `preprocessing/case_matching/05_nnUNet_labels_metrics_generation.py`. If the file is missing or a key is absent, the loader substitutes 0.0 for that metric.

---

## MONAI Data Pipeline

The pipeline is split into two stages to avoid re-running slow preprocessing on every epoch.

### Cached Transforms (`get_cached_transforms`)

Run once and stored by MONAI's `CacheDataset` (in-RAM caching). Applied to every sample before training begins:

1. **Load NIfTI** — loads `imagesTr/*.nii.gz`, `labelsTr/*.nii.gz`, `distMapsTr/*.nii.gz`
2. **RAS orientation** — reorients all volumes to standard (Right-Anterior-Superior)
3. **(1.5, 1.5, 3) mm resampling** — matches Merlin's pretrained geometry
4. **HU clipping** — clips CT intensities to [-1000, 1000] HU
5. **Pad + crop to 224×224×160** — `SpatialPadd` zero-pads volumes smaller than the target; `CenterSpatialCropd` trims any larger volume to exactly 224×224×160. Replaces `Resized` to preserve voxel structure rather than stretching.
6. **`BuildCTAd`** — assembles the 4-channel tensor (see below)

### Runtime Transforms (`get_train_runtime_transforms`, `get_val_runtime_transforms`)

Applied on-the-fly at training time via `RuntimeDataset`, on top of the cached tensors.

**Training — spatial augmentations** (applied to `ct`, `lbl`, and `dst` together, before `BuildCTAd`):

All three volumes must be warped identically so the lumen mask and distance map stay geometrically aligned with the CT. Spatial transforms therefore use the key list `["ct", "lbl", "dst"]` and run before `BuildCTAd` merges them into the 4-channel tensor.

| Transform | Config | Purpose |
|---|---|---|
| `RandFlipd` axis=0 | p=0.5 | Left-right flip (R axis) |
| `RandFlipd` axis=1 | p=0.5 | Anterior-posterior flip (A axis) |
| `RandFlipd` axis=2 | p=0.4 | Superior-inferior flip (S axis) |
| `RandRotate90d` axes=(0,1) | p=0.5, max_k=3 | 0/90/180/270° axial-plane rotation |
| `RandAffined` | p=0.3, ±10° all axes, ±10/10/5 vox translation | Patient tilt and table offset |
| `RandZoomd` | p=0.3, 85–115%, keep_size=True | Body size variation |

Note: `RandZoomd` and `RandAffined` do not recompute the pre-stored `aaa_metrics` scalars — those remain the clinically authoritative values.

**Training — intensity augmentations** (applied to `ct` only — `lbl` is a label map and `dst` a distance field):

| Transform | Config | Purpose |
|---|---|---|
| `RandAdjustContrastd` | p=0.3, γ=0.7–1.3 | Scanner-to-scanner contrast variation |
| `RandShiftIntensityd` | p=0.3, ±0.1 | HU calibration offset |
| `RandScaleIntensityd` | p=0.5, ±15% | Multiplicative intensity scaling |
| `RandBiasFieldd` | p=0.2, coeff=0–0.3 | Low-frequency field inhomogeneity |
| `RandGaussianNoised` | p=0.3, σ=0.05 | Detector noise |
| `RandGaussianSmoothd` | p=0.2, σ_z=0.25–0.5 (half of σ_xy) | Reconstruction kernel variation; σ_z is narrower to match the 3 mm axial slice spacing |

With all augmentations active, at least one fires on ≈98% of training samples.

**Validation:**
- `BuildCTAd` + `ToTensord` + `NormalizeClinicald` only — no augmentation.

### `BuildCTAd`

Assembles the 4-channel input tensor from the cached volumes:

| Channel | Source | Description |
|---|---|---|
| 0 | `imagesTr/*.nii.gz` | CT intensity (HU, clipped and resized) |
| 1 | `labelsTr/*.nii.gz` (label==1) | Binary lumen mask |
| 2 | `labelsTr/*.nii.gz` (label==2) | Binary ILT mask |
| 3 | `distMapsTr/*.nii.gz` | Signed distance map from lumen surface |

Output key: `"cta"`, shape `(4, 224, 224, 160)`.

### `NormalizeClinicald`

Z-normalises the clinical feature vector in-place:

```
clin_normalised = (clin - clin_mean) / (clin_std + ε)
```

Mean and std are computed from the training split only (`compute_clin_stats`) and stored in the checkpoint to ensure consistent normalisation at inference.

---

## Preprocessing Pipeline

Scripts in `preprocessing/case_matching/` must be run in order. They are designed to run on the HPC via `data/prepare_data.sh`.

### Step-by-step

| Script | Purpose |
|---|---|
| `00_dicom_to_nifti_renaming.py` | Converts DICOM series to NIfTI and renames files to `<id>_<site>.nii.gz` |
| `01_cfd_name_cleaning.py` | Standardises CFD `.vtp` filenames to match the RADAR patient ID format |
| `02_umcg_name_cleaning.py` | Site-specific name fixing for UMCG files (different ID convention) |
| `03_data_matching_v1.py` | **Main matching script** — joins image files to SPSS clinical records; outputs `radar_multimodal_config_v1.json` and `radar_multimodal_config_v1_analysis.txt` |
| `04_data_organizing.py` | Copies matched images, labels, and CFD files into `training_subset_v1/` |
| `05_nnUNet_labels_metrics_generation.py` | Runs nnUNet segmentation and extracts AAA geometry metrics |
| `06_crop_full_body_scans.py` | Crops each CT axially to the nnUNet-segmented aortic region + padding, ensuring the axial physical extent is ≥ 480 mm (160 slices × 3 mm) so the target volume is filled with anatomy rather than empty air |
| `07_generate_data_split.py` | Creates stratified train/val/test split JSON |

### Site Configuration (`03_data_matching_v1.py`)

Four sites are supported, each with its own filename pattern and ID normalisation rule:

| Site key | File pattern | SPSS site string |
|---|---|---|
| `RNST` | `(\d+)_RNST` | `001 lokatie Arnhem` |
| `MST` | `(\d+)_MST` | `007 - Medisch Spectrum Twente (MST)` |
| `AUMC` | `(\d+)_AUMC` | `008 - AmsterdamUMC` |
| `UMCG` | `(\d+)_UMCG` | `009 - UMCG` |

Adding a new site requires adding an entry to `SITE_CONFIG` in `03_data_matching_v1.py`.

### Analysis Report

`03_data_matching_v1.py` writes a detailed analysis to:
```
clinical_data/radar_multimodal_config_v1_analysis.txt
```

This includes a unified ground-truth table (total + per-site columns) and a data quality section listing patients with missing baseline or follow-up diameter measurements.
