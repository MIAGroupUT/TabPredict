"""
03_data_matching_v1.py — Match SPSS clinical data with session index → config JSON (v2)
=========================================================================================

Combines:
  - SPSS .sav clinical database  (ground truth + clinical features)
  - session_index_v2.json        (imaging sessions from data)

Output: radar_multimodal_config_v2.json

JSON structure (per patient)
----------------------------
{
  "{spss_id}_{site_abbr}": {
    "metadata": {
      "id":              "{spss_id}",          // SPSS-normalised (no leading zeros)
      "original_nr":     "{raw_patient_id}",   // as stored in SPSS PatientNr
      "site":            "008 - AmsterdamUMC",   // or "007 - MST" / "009 - UMCG" / RNST
      "site_short":      "AUMC",
      "folder_id":       "02637442"            // raw folder name in data
    },
    "scans": {
      "pre_op": {
        "session_id":        "02637442_CT_1",
        "modality":          "CT",
        "modality_index":    1,
        "file_stem":         "2637442_AUMC_s001_CT",
        "dicom_session_dir": "/.../02637442/02637442_CT_1",
        "nifti":             "/.../imagesTr/2637442_AUMC_s001_CT.nii.gz",
        "label":             "/.../labelsTr/2637442_AUMC_s001_CT.nii.gz",
        "dist_map":          "/.../distMapsTr/2637442_AUMC_s001_CT.nii.gz",
        "metrics":           "/.../metricsTr/2637442_AUMC_s001_CT.json",
        "cfd":               null
      },
      "pre_op_other": [
        {
          "session_id":        "02637442_US_1",   // pre-op US (index 1 = before surgery)
          "modality":          "US",
          "modality_index":    1,
          "file_stem":         "2637442_AUMC_s001_US",
          "nifti":             "/.../imagesTr/2637442_AUMC_s001_US.nii.gz"
          // no label / dist_map / metrics / cfd for non-CT modalities
        }
      ],
      "post_op": [
        {
          "session_id":        "HT624_CT_1",
          "modality":          "CT",
          "modality_index":    2,
          ...
        },
        {
          "session_id":        "HT624_US_2",
          "modality":          "US",
          "modality_index":    2,
          ...
        }
      ]
    },
    "pre_op_features":  { ... },   // all Bl_* SPSS columns (unchanged from v1)
    "post_op_features": { ... },   // all non-Bl_* SPSS columns (unchanged from v1)
    "ground_truth":     { ... }    // reintervention, death, complications (unchanged)
  }
}

Backward-compat note
---------------------
V1: entry["scans"]["pre_op"]["nifti"].
The training DataLoader should select the pre_op key exclusively for the v1-style model.
"""

import datetime
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


# =========================================================
# SITE CONFIGURATION  (mirrors v1 SITE_CONFIG)
# =========================================================
SITE_CONFIG = {
    "AUMC": {
        "spss_site_name": "008 - AmsterdamUMC",
        "spss_id_format": lambda x: str(int(x)),   # strip leading zeros
    },
    "UMCG": {
        "spss_site_name": "009 - UMCG",
        "spss_id_format": lambda x: str(int(x)),
    },
    "MST": {
        "spss_site_name": "007 - Medisch Spectrum Twente (MST)",
        "spss_id_format": lambda x: str(int(x)),   # strip leading zeros, same as AUMC
    },
    "RNST": {
        "spss_site_name": "001 lokatie Arnhem",
        "spss_id_format": lambda x: str(int(x)).zfill(4),  # RNST IDs are zero-padded to 4 digits in SPSS
    },
}

# Reverse lookup: SPSS site name → site_abbr
_SITE_NAME_TO_ABBR = {cfg["spss_site_name"]: abbr for abbr, cfg in SITE_CONFIG.items()}


# =========================================================
# SERIALISATION HELPERS  (identical to v1)
# =========================================================

def serialize_value(val):
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.isoformat()
    try:
        if pd.isna(val) or val is pd.NA:
            return None
    except Exception:
        pass
    if isinstance(val, (np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.float64, np.float32)):
        return float(val)
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return val


def is_flagged(val) -> bool:
    try:
        if pd.isna(val) or val is None:
            return False
        return float(val) == 1.0
    except Exception:
        return False


# =========================================================
# PATH BUILDERS
# =========================================================

def scan_paths(spss_id: str, site_abbr: str, modality: str, modality_index: int, derivatives_base: str) -> dict:
    """Return the standard derivative paths for any imaging scan.

    CT-only derivatives (label, dist_map, metrics, cfd) are omitted for non-CT modalities.
    """
    stem = f"{spss_id}_{site_abbr}_s{modality_index:03d}_{modality}"
    base = Path(derivatives_base) / "training_subset_v2"
    paths = {
        "file_stem": stem,
        "nifti":     str(base / "imagesTr" / f"{stem}.nii.gz"),
    }
    if modality == "CT":
        paths["label"]    = str(base / "labelsTr"   / f"{stem}.nii.gz")
        paths["dist_map"] = str(base / "distMapsTr" / f"{stem}.nii.gz")
        paths["metrics"]  = str(base / "metricsTr"  / f"{stem}.json")
        paths["cfd"]      = None
    return paths


# =========================================================
# GROUND TRUTH EXTRACTION  (identical to v1)
# =========================================================

COMP_PATTERNS = {
    "endoleak_1a": r"Endoleak_spec_.*answer0",
    "endoleak_1b": r"Endoleak_spec_.*answer(1|2)",
    "endoleak_2":  r"Endoleak_spec_.*answer(3|4|5|6)",
    "endoleak_3":  r"Endoleak_spec_.*answer(7|8)",
    "endoleak_5":  r"Endoleak_spec_.*answer11",
    "occlusion":   r"Occlusion_",
    "stenosis":    r"Stenosis_",
    "kinking":     r"Kinking_",
    "fracture":    r"Fract_",
    "migration":   r"Endograft_Spc.*answer1",
    "infection":   r"Endograft_Spc.*answer2",
}


def extract_ground_truth(row, df_columns: list) -> dict:
    """Extract all outcome labels from a SPSS patient row."""
    # Reintervention
    reint_summary = is_flagged(row.get("ReintervAAArel_0m18m_yn"))
    reint_ae_cols = [c for c in df_columns if re.search(r"AEAction.*answer4", c)]
    has_reint = int(reint_summary or any(is_flagged(row.get(c)) for c in reint_ae_cols))

    # Death
    d_surg  = is_flagged(row.get("Bl_Bl_rAAA_death_surg"))
    d_hosp  = is_flagged(row.get("Bl_Bl_rAAA_deathER"))
    d_ae    = is_flagged(row.get("AE0_AE_AE_Death"))
    reason_cols = [c for c in df_columns if re.search(r"^Reason(_\d+)?$", c)]
    d_rsn   = any(serialize_value(row.get(c)) == 0 for c in reason_cols)
    has_death = int(d_ae or d_surg or d_hosp or d_rsn)

    # Sac dynamics
    bl_ct_cols = [c for c in df_columns if re.search(r"Bl.*CTA.*AAAdiam", c, re.I)]
    bl_us_cols = [c for c in df_columns if re.search(r"Bl.*(?:US|DUS).*AAAdiam", c, re.I)]
    bl_ct = next((float(row.get(c)) for c in bl_ct_cols if pd.notna(row.get(c))), None)
    bl_us = next((float(row.get(c)) for c in bl_us_cols if pd.notna(row.get(c))), None)

    fu_cands_ct, fu_cands_us = [], []
    for c in df_columns:
        if "NE" in c or "AAAdiam" not in c or c.startswith("Bl_"):
            continue
        m = re.search(r"FU(\d+)", c, re.I) or re.search(r"(\d+)yr", c, re.I)
        if not m:
            continue
        vn = int(m.group(1))
        if re.search(r"CTA", c, re.I):
            fu_cands_ct.append((c, vn))
        elif re.search(r"DUS|_US_", c, re.I):
            fu_cands_us.append((c, vn))

    fu_cands_ct.sort(key=lambda x: x[1], reverse=True)
    fu_cands_us.sort(key=lambda x: x[1], reverse=True)
    fu_ct = next((float(row.get(c)) for c, _ in fu_cands_ct if pd.notna(row.get(c))), None)
    fu_us = next((float(row.get(c)) for c, _ in fu_cands_us if pd.notna(row.get(c))), None)

    if bl_ct is not None and fu_ct is not None:
        bl_val, fu_val = bl_ct, fu_ct
    elif bl_us is not None and fu_us is not None:
        bl_val, fu_val = bl_us, fu_us
    else:
        bl_gen = [c for c in df_columns if re.search(r"^Bl_AAAdiam$", c)]
        bl_val = next((float(row.get(c)) for c in bl_gen if pd.notna(row.get(c))), None) \
                 if (bl_ct is None and bl_us is None) else (bl_ct or bl_us)
        fu_val = fu_ct or fu_us

    diam_diff    = (fu_val - bl_val) if (bl_val is not None and fu_val is not None) else None
    sac_regression = int(diam_diff is not None and diam_diff <= -5.0)
    sac_stable     = int(diam_diff is not None and -5.0 < diam_diff < 5.0)
    sac_expansion  = int(diam_diff is not None and diam_diff >= 5.0)

    gt_comps = {}
    for label, pattern in COMP_PATTERNS.items():
        cols = [c for c in df_columns if re.search(pattern, c)]
        gt_comps[label] = int(any(is_flagged(row.get(c)) for c in cols))

    return {
        "reintervention":  has_reint,
        "death":           has_death,
        "death_surg":      int(d_surg),
        "death_hosp":      int(d_hosp),
        "sac_regression":  sac_regression,
        "sac_stable":      sac_stable,
        "sac_expansion":   sac_expansion,
        "diam_diff":       serialize_value(diam_diff),
        **gt_comps,
    }


# =========================================================
# MAIN BUILD FUNCTION
# =========================================================

def build_multimodal_dataset_v2(
    sav_path: str,
    session_index_json: str,
    derivatives_base: str,
    output_json: str,
) -> None:
    """
    Merge SPSS clinical data with the imaging session index to produce
    the v2 multimodal config JSON.

    Parameters
    ----------
    sav_path : str
        Path to the SPSS .sav clinical database.
    session_index_json : str
        Output of 01_build_session_index.py.
    derivatives_base : str
        Root under which training_subset_v2/ will live.
    output_json : str
        Destination for the completed radar_multimodal_config_v2.json.
    """
    print(f"Loading SPSS data: {sav_path}")
    df, meta = pyreadstat.read_sav(sav_path)
    df["patient_str"] = df["PatientNr"].astype(str).str.split(".").str[0]
    df["site_str"]    = df["Site"].astype(str).str.strip()

    print(f"Loading session index: {session_index_json}")
    with open(session_index_json) as f:
        session_index = json.load(f)

    # Feature column split (same rule as v1)
    pre_op_cols  = [c for c in df.columns if c.startswith("Bl_") or c in ["PatientNr", "CreatedInSite", "Site", "Height"]]
    post_op_cols = [c for c in df.columns if c not in pre_op_cols and c not in ["PatientNr", "patient_str", "site_str"]]

    final_data = {}
    stats = {"total": 0, "matched": 0, "no_preop_ct": 0, "spss_not_found": 0}

    for site_abbr, patients in session_index.items():
        cfg = SITE_CONFIG.get(site_abbr)
        if cfg is None:
            print(f"[WARN] No SITE_CONFIG for {site_abbr}, skipping")
            continue

        spss_site_name = cfg["spss_site_name"]
        id_fmt         = cfg["spss_id_format"]

        for folder_id, patient_info in patients.items():
            stats["total"] += 1
            sessions = patient_info["sessions"]

            # Normalise to SPSS ID
            try:
                spss_id = id_fmt(folder_id)
            except Exception:
                spss_id = folder_id

            # Match SPSS row
            mask = (df["patient_str"] == spss_id) & (df["site_str"] == spss_site_name)
            patient_rows = df[mask]
            if patient_rows.empty:
                print(f"[MISS] {folder_id} ({site_abbr}) not in SPSS")
                stats["spss_not_found"] += 1
                continue

            row = patient_rows.iloc[0]

            # Separate sessions by timing and modality.
            # pre_op    → the pre-op CT (primary model input, always index 1).
            # pre_op_other → other pre-op modalities (US_1, MR_1, …); index 1 but not CT.
            # post_op   → everything with is_pre_op=False.
            pre_op_ct_sessions    = [s for s in sessions if s["is_pre_op"] and s["modality"] == "CT"]
            pre_op_other_sessions = [s for s in sessions if s["is_pre_op"] and s["modality"] != "CT"]
            post_op_sessions      = [s for s in sessions if not s["is_pre_op"]]

            if not pre_op_ct_sessions:
                print(f"[WARN] {folder_id} ({site_abbr}) has no pre-op CT session — included with null pre_op")
                stats["no_preop_ct"] += 1
                pre_op_entry = None
            else:
                ps = pre_op_ct_sessions[0]
                pre_op_entry = {
                    "session_id":        ps["session_id"],
                    "modality":          ps["modality"],
                    "modality_index":    ps["modality_index"],
                    "dicom_session_dir": ps["session_dir"],
                    **scan_paths(spss_id, site_abbr, ps["modality"], ps["modality_index"], derivatives_base),
                }

            def _make_entry(s: dict) -> dict:
                return {
                    "session_id":        s["session_id"],
                    "modality":          s["modality"],
                    "modality_index":    s["modality_index"],
                    "dicom_session_dir": s["session_dir"],
                    **scan_paths(spss_id, site_abbr, s["modality"], s["modality_index"], derivatives_base),
                }

            pre_op_other_entries = [_make_entry(s) for s in sorted(
                pre_op_other_sessions, key=lambda s: (s["modality"], s["modality_index"])
            )]

            post_op_entries = [_make_entry(s) for s in post_op_sessions]
            # Sort post-op: CT first (by modality_index), then other modalities alphabetically
            post_op_entries.sort(key=lambda x: (x["modality"] != "CT", x["modality"], x["modality_index"]))

            unique_key = f"{spss_id}_{site_abbr}"
            final_data[unique_key] = {
                "metadata": {
                    "id":         spss_id,
                    "original_nr": serialize_value(row["PatientNr"]),
                    "site":        spss_site_name,
                    "site_short":  site_abbr,
                    "folder_id":   folder_id,
                },
                "scans": {
                    "pre_op":       pre_op_entry,
                    "pre_op_other": pre_op_other_entries,
                    "post_op":      post_op_entries,
                },
                "pre_op_features":  {k: serialize_value(row[k]) for k in pre_op_cols  if k in row},
                "post_op_features": {k: serialize_value(row[k]) for k in post_op_cols if k in row},
                "ground_truth":     extract_ground_truth(row, list(df.columns)),
            }
            stats["matched"] += 1

    # Save
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(final_data, f, indent=4)

    print(f"\n{'='*70}")
    print(f"V2 CONFIG BUILD COMPLETE")
    print(f"{'='*70}")
    print(f"  Patients in session index : {stats['total']}")
    print(f"  Matched to SPSS           : {stats['matched']}")
    print(f"  Not found in SPSS         : {stats['spss_not_found']}")
    print(f"  Missing pre-op CT         : {stats['no_preop_ct']}")
    print(f"  Output                    : {output_json}")

    # Quick GT summary
    if final_data:
        total = len(final_data)
        reint = sum(1 for p in final_data.values() if p["ground_truth"]["reintervention"])
        print(f"\n  Reintervention prevalence : {reint}/{total} ({100*reint/total:.1f}%)")

    _print_scan_coverage(final_data)


def _print_scan_coverage(final_data: dict) -> None:
    """Print a brief table of scan counts per patient, broken down by modality."""
    from collections import Counter
    print(f"\n{'Patient Key':<25} {'PreCT':<6} {'PreOth':<7} {'FU-CT':<7} {'FU-US':<7} {'FU-MR':<7} {'FU-XR':<7} {'Other'}")
    print("-" * 90)
    for key, p in sorted(final_data.items()):
        scans   = p["scans"]
        po      = scans.get("post_op", [])
        poo     = scans.get("pre_op_other", [])
        fu_mods = Counter(s["modality"] for s in po)
        pre_oth = Counter(s["modality"] for s in poo)
        preop_flag = "YES" if scans.get("pre_op") else "NO"
        pre_oth_str = ",".join(f"{m}:{n}" for m, n in sorted(pre_oth.items())) or "-"
        print(
            f"  {key:<23} {preop_flag:<6} {pre_oth_str:<7} {fu_mods.get('CT', 0):<7}"
            f" {fu_mods.get('US', 0):<7} {fu_mods.get('MR', 0):<7} {fu_mods.get('XR', 0):<7}"
            f" {sum(v for k, v in fu_mods.items() if k not in ('CT','US','MR','XR'))}"
        )


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    SAV_PATH = (
        "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/"
        "1. RADAR_datebase_ARTinEVAR_v4, incl Bl_AAAdiam voor oude Rijnstate patienten.sav"
    )
    SESSION_INDEX = (
        "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2/"
        "clinical_data/session_index_v2.json"
    )
    DERIVATIVES_BASE = "/deepstore/datasets/mia/AI4EVAR/derivatives"
    OUTPUT_JSON = (
        "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2/"
        "clinical_data/radar_multimodal_config_v2.json"
    )

    build_multimodal_dataset_v2(SAV_PATH, SESSION_INDEX, DERIVATIVES_BASE, OUTPUT_JSON)
