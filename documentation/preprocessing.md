# Preprocessing Pipeline (v1)

All scripts live in `preprocessing/case_matching_v1/` and must be run in order.
Each step reads from and writes to `/deepstore/datasets/mia/AI4EVAR/`.

```
Raw XNAT data  ──►  00  ──►  01  ──►  02  ──►  03  ──►  04  ──►  05  ──►  06  ──►  07
DICOM on disk      NIfTI   Session  CFD      Config   Organise  nnUNet   Crop    Split
                   convert  index   rename   JSIndex DON     files     + AAA    SI      JSON
```

---

## Step 00 — DICOM → NIfTI (`00_dicom_to_nifti_renaming.py`)

**Input**: raw XNAT archive under `raw_data/RADAR_XNat/data/`  
**Output**: `raw_data/RADAR_XNat/imagesTr_all/{folder_id}_{site}_{session_id}.nii.gz`

Converts every imaging session from DICOM to NIfTI using SimpleITK. Two XNAT layouts
are handled:

| Layout | Sites | Structure |
|--------|-------|-----------|
| **Layered** | AUMC, UMCG | `patient/session_per_modality/scans/…/DICOM/` |
| **Flat** | MST, RNST | `patient/patient/scans/…/DICOM/` — all modalities as series inside one dir |

For flat-session sites a synthetic session ID `{patient_id}_{modality}_1` is created
so the output filename convention is uniform across all sites.

### DICOM Series Selection

When a session contains multiple series (e.g. thin-slice + thick-slice + scout), the
script scores each candidate and picks the highest-scoring one.

### Score Components

| Category | Condition | Points |
|---|---|---|
| **Orientation** | Axial plane (`ImageOrientationPatient` ≈ [1,0,0,0,1,0]) | +5 |
| | Non-axial (coronal/sagittal reconstruction) | −3 |
| **Series description** | Contains "arter" | +4 |
| | Contains "cta" | +4 |
| | Contains "contrast" or "ce" | +2 |
| | Contains "ax" | +2 |
| | Contains "cor" or "sag" | −5 |
| | Contains "scout", "localizer", "topogram", "surview" | −10 |
| **Slice thickness** | ≤ 0.75 mm (sub-mm, e.g. 0.625 mm) | +8 |
| | ≤ 1.25 mm | +6 |
| | ≤ 2.0 mm | +4 |
| | ≤ 5.0 mm | 0 |
| | > 5.0 mm | −6 |
| **Slice count** | > 300 slices | +6 |
| | > 150 slices | +4 |
| | > 80 slices | +2 |
| | > 30 slices | +1 |
| | < 20 slices | −5 |

### Worked example — typical patient with 3 series

| Series | Orientation | Description | Thickness | Slices | **Score** |
|---|---|---|---|---|---|
| Arterial CTA 0.625 mm | Axial +5 | "CTA arterial" +8 | 0.625 mm +8 | 450 +6 | **27** ✓ |
| Venous recon 3 mm | Axial +5 | "venous" +0 | 3 mm +0 | 160 +4 | **9** |
| Scout localizer | Non-axial −3 | "topogram" −10 | 5 mm +0 | 12 −5 | **−18** |

Thickness and slice count together carry up to **14 points**, which is more than any
single description keyword (+4). A 0.625 mm series with no useful description still
outscores a 3 mm series named "CTA arterial" (+8+6=14 vs +4+4=8).

For non-CT modalities (US, MR, XR) the scoring is simpler: pick the series with the
most files, discarding anything with fewer than 5 files (scouts).

---

## Step 01 — Session Index (`01_build_session_index.py`)

**Input**: same XNAT archive  
**Output**: `clinical_data/session_index_v1.json`

Walks the raw data and builds a structured JSON index of every imaging session per
patient. Does not convert anything — purely metadata extraction.

Key tasks:
- Reads `Modality` DICOM tag from the first readable file in each session (normalises CR/DX → XR, PT → PET)
- Reads `StudyDate` (0008,0020) → stored as ISO `YYYY-MM-DD` per session
- For AUMC: uses session name as a fallback if the DICOM is unreadable (`{id}_{MODALITY}_{n}`)
- For UMCG: uses the raw numeric suffix as the visit order index (already encodes chronology)
- For AUMC/MST/RNST: sorts sessions by `scan_date` (nulls last) and assigns `modality_index` 1, 2, 3…

Output structure per patient:
```json
{
  "AUMC": {
    "02637442": {
      "patient_dir": "/deepstore/.../02637442",
      "sessions": [
        {
          "session_id":     "02637442_CT_1",
          "modality":       "CT",
          "modality_index": 1,
          "scan_date":      "2018-04-23",
          "session_dir":    "/deepstore/..."
        }
      ]
    }
  }
}
```

---

## Step 02 — CFD Name Cleaning (`02_cfd_name_cleaning.py`)

**Input**: `raw_data/CFDs/ensemble_predictions_v3--AAA-full-all/`  
**Output**: `raw_data/CFDs/renamed_v1/`

Short utility script. CFD mesh files from the fluid-dynamics simulation arrive named
`RADAR_CTA_{ID}_....vtp`. This script renames them to match the NIfTI naming convention
used by the rest of the pipeline:

```
RADAR_CTA_118_....vtp  →  118_MST_118_CT_1.vtp
```

Pattern: `RADAR_CTA_(\d+)_.*\.vtp` → `{id}_MST_{id}_CT_1.vtp`

Currently only processes MST CFD files (the only site with CFD data in v1).

---

## Step 03 — Data Matching (`03_data_matching_v1.py`)

**Input**: SPSS `.sav` clinical database + `session_index_v1.json`  
**Output**: `clinical_data/radar_multimodal_config_v1.json` + `_analysis.txt`

The central matching step. Joins imaging metadata with clinical/outcome data from SPSS
and classifies every scan as pre-op, post-op, or undated.

### Pre-op / post-op classification

The hard boundary is `P_IP_P_Date` (implant procedure date) from SPSS:

```
scan_date <= P_IP_P_Date  →  pre-op
scan_date >  P_IP_P_Date  →  post-op
scan_date unavailable     →  undated
```

Among pre-op CT sessions, the one with `scan_date` **closest to `P_IP_P_Date`** (most
recent pre-op) is selected as `pre_op`. Earlier pre-op CTs move to `pre_op_other`.

### File stem and derivative paths

Each scan is assigned a canonical stem: `{spss_id}_{site}_s{idx:03d}_{modality}`  
e.g. `2637442_AUMC_s001_CT`

All derivative paths (label, dist map, metrics, CFD) are stored in the config using
this stem so downstream steps can find them without re-scanning directories.

### Ground truth extraction

All labels are extracted from the SPSS row at this step:

| Label | Source | Logic |
|---|---|---|
| `reintervention` | `ReintervAAArel_0m18m_yn` + AE action cols | 1 if summary flag or any `AEAction.*answer4` |
| `death` | `AE0_AE_AE_Death`, `rAAA_death_*`, `Reason` cols | any of these flagged |
| `endoleak_1a/1b/2/3/5` | `Endoleak_spec_*answerN` pattern | regex on column names |
| `migration` | `Endograft_Spc.*answer1` | |
| `infection` | `Endograft_Spc.*answer2` | |
| `sac_regression/stable/expansion` | `Bl_*AAAdiam` + `FU*AAAdiam` | Δ ≤ −5 mm / −5<Δ<5 / ≥ 5 mm |

**SAC diameter mapping strategy** (fallback order):
1. CT-CT: baseline `Bl.*CTA.*AAAdiam` + follow-up `FU.*CTA.*AAAdiam`
2. US-US: baseline `Bl.*DUS.*AAAdiam` + follow-up `FU.*DUS.*AAAdiam`
3. Cross-modality fallbacks (CT-US, US-CT, GEN-CT, GEN-US)
4. None — no diameter pair available

For complications, the script also finds the **earliest occurrence date** (from FU visit
dates or AE event dates) and stores it as `comp_timing` for survival analysis.

---

## Step 04 — Data Organising (`04_data_organizing.py`)

**Input**: `radar_multimodal_config_v1.json` + `imagesTr_all/` (from step 00)  
**Output**: `training_subset_v1/imagesTr/`, `cfdTr/`

Reads the config and copies each NIfTI from the flat `imagesTr_all/` dump into the
structured `training_subset_v1/` hierarchy under the canonical file stem.

```
imagesTr_all/02637442_AUMC_02637442_CT_2.nii.gz
    →  training_subset_v1/imagesTr/2637442_AUMC_s002_CT.nii.gz
```

Also copies renamed CFD `.vtp` meshes into `cfdTr/` if a CFD source directory is
provided. Skips files that already exist (idempotent).

---

## Step 05 — nnUNet Segmentation + AAA Metrics (`05_nnUNet_labels_metrics_generation.py`)

**Input**: `training_subset_v1/imagesTr/*.nii.gz` (CT only)  
**Output**:
- `labelsTr/{stem}.nii.gz` — segmentation mask (0=bg, 1=lumen, 2=ILT)
- `distMapsTr/{stem}.nii.gz` — lumen-to-wall distance map (mm)
- `metricsTr/{stem}.json` — 5 AAA geometry scalars

Non-CT modalities (MR, US, XR) are skipped — the nnUNet model was trained on CTA only.

### Segmentation

Runs nnUNetv2 inference (`nnUNetPredictor`) on each CT using a pre-trained fold
ensemble. Produces a 3-label mask aligned to the input NIfTI space.

### Distance map

```python
dist_map = distance_transform_edt(~lumen_mask, sampling=voxel_spacing)
dist_map_masked = dist_map * (lumen_mask | ilt_mask)
```

Euclidean distance from each AAA-region voxel to the nearest lumen surface, in mm.
Values outside the lumen+ILT mask are zeroed. This becomes channel 3 of the 4-channel
CTA input tensor during training.

### AAA geometry metrics

| Metric | Formula |
|---|---|
| `volume_lumen_ml` | `sum(lumen_mask) × voxel_volume / 1000` |
| `volume_ilt_ml` | `sum(ilt_mask) × voxel_volume / 1000` |
| `volume_total_ml` | `sum(lumen_mask \| ilt_mask) × voxel_volume / 1000` |
| `max_lumen_to_wall_dist_mm` | `max(dist_map[ilt_mask])` |
| `avg_lumen_to_wall_dist_mm` | `mean(dist_map[ilt_mask])` |

These 5 scalars are the AAA branch inputs to the model (bypass the L1 feature gate).

---

## Step 06 — Crop Full-body Scans (`06_crop_full_body_scans.py`)

**Input/Output**: `imagesTr/`, `labelsTr/`, `distMapsTr/` (in-place)

Some sites acquired full-body CTAs (head-to-toe). The MONAI pipeline resamples to
(1.5, 1.5, 3) mm and then pads/crops to 224×224×160 voxels. Target axial extent is
160 × 3 mm = **480 mm**. Without cropping, the aortic sac occupies only a small
fraction of the 160-slice window and the rest is zero-padded.

This step:
1. Finds the axial (S-I) extent of the nnUNet segmentation mask (any non-zero label)
2. Adds a padding fraction on each end (default 15% of the mask extent)
3. Crops `imagesTr`, `labelsTr`, and `distMapsTr` to the same axial range
4. Skips scans where the mask already fills ≥ 80% of the volume (`--skip-threshold`)

Only the axial axis is cropped; X/Y are left unchanged. The three paired files are
always cropped together to keep them spatially consistent.

---

## Step 07 — Train/Val/Test Split (`07_generate_data_split.py`)

**Input**: `radar_multimodal_config_v1.json` + existence check on `imagesTr/`  
**Output**: `training_subset_v1/dataset_splits.json`

Splits at **patient level** (not scan level) — all scans for a patient land in the
same set, preventing data leakage.

### Stratification

Double stratification: `{site}_{reintervention_label}`, giving 8 strata
(4 sites × 2 labels). Within each stratum, patients are shuffled with a fixed seed
and split proportionally.

Default ratios: **70% train / 10% val / 20% test**  
Default seed: `3299`

Only patients with a pre-op CT NIfTI actually present on disk are included.
Patients with missing pre-op CT are silently excluded.

Output format:
```json
{
  "train":    ["2637442_AUMC", "387_UMCG", ...],
  "val":      [...],
  "test":     [...],
  "metadata": {
    "seed": 3299,
    "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
    "total_patients": 847
  }
}
```
