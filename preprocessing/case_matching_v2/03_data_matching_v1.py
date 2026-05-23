"""
03_data_matching_v1.py — Match SPSS clinical data with session index → config JSON
=========================================================================================

Combines:
  - SPSS .sav clinical database  (ground truth + clinical features)
  - session_index_v1.json        (imaging sessions from data)

Output: radar_multimodal_config_v1.json

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
from collections import defaultdict
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

def scan_paths(
    spss_id: str,
    site_abbr: str,
    modality: str,
    modality_index: int,
    derivatives_base: str,
    cfd_src: str | None = None,
) -> dict:
    """Return the standard derivative paths for any imaging scan.

    CT-only derivatives (label, dist_map, metrics, cfd) are omitted for non-CT modalities.
    cfd_src: full path to the renamed source .vtp; cfd destination is set only if it exists.
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
        paths["cfd"]      = str(base / "cfdTr" / f"{stem}.vtp") if (cfd_src and Path(cfd_src).exists()) else None
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
        "reintervention":    has_reint,
        "death":             has_death,
        "death_surg":        int(d_surg),
        "death_hosp":        int(d_hosp),
        "sac_measured":      int(diam_diff is not None),
        "sac_regression":    sac_regression,
        "sac_stable":        sac_stable,
        "sac_expansion":     sac_expansion,
        "diam_diff":         serialize_value(diam_diff),
        "has_baseline_diam": bl_val is not None,
        "has_fu_diam":       fu_val is not None,
        **gt_comps,
    }


# =========================================================
# MAIN BUILD FUNCTION
# =========================================================

def build_multimodal_dataset_v1(
    sav_path: str,
    session_index_json: str,
    derivatives_base: str,
    output_json: str,
    cfd_renamed_dir: str | None = None,
) -> None:
    """
    Merge SPSS clinical data with the imaging session index to produce
    the v1 multimodal config JSON.

    Parameters
    ----------
    sav_path : str
        Path to the SPSS .sav clinical database.
    session_index_json : str
        Output of 01_build_session_index.py.
    derivatives_base : str
        Root under which training_subset_v1/ will live.
    output_json : str
        Destination for the completed radar_multimodal_config_v1.json.
    cfd_renamed_dir : str | None
        Directory of renamed CFD .vtp files (output of 02_cfd_name_cleaning.py).
        For each CT scan, the corresponding {folder_id}_{site}_{session_id}.vtp is
        looked up here; cfd is set to the destination path only if the source exists.
    """
    print(f"Loading SPSS data: {sav_path}")
    df, meta = pyreadstat.read_sav(sav_path)
    df["patient_str"] = df["PatientNr"].astype(str).str.split(".").str[0]
    df["site_str"]    = df["Site"].astype(str).str.strip()

    print(f"Loading session index: {session_index_json}")
    with open(session_index_json) as f:
        session_index = json.load(f)

    cfd_base = Path(cfd_renamed_dir) if cfd_renamed_dir else None

    # Feature column split (same rule as v1)
    pre_op_cols  = [c for c in df.columns if c.startswith("Bl_") or c in ["PatientNr", "CreatedInSite", "Site", "Height"]]
    post_op_cols = [c for c in df.columns if c not in pre_op_cols and c not in ["PatientNr", "patient_str", "site_str"]]

    final_data = {}
    stats = {"total": 0, "matched": 0, "no_preop_ct": 0, "spss_not_found": 0}
    spss_not_found_list: list[tuple[str, str]] = []   # (folder_id, site_abbr)
    no_preop_ct_list:    list[tuple[str, str]] = []   # (spss_id,   site_abbr)

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
                spss_not_found_list.append((folder_id, site_abbr))
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
                no_preop_ct_list.append((spss_id, site_abbr))
                pre_op_entry = None
            else:
                ps = pre_op_ct_sessions[0]
                _cfd_src = str(cfd_base / f"{folder_id}_{site_abbr}_{ps['session_id']}.vtp") if cfd_base else None
                pre_op_entry = {
                    "session_id":        ps["session_id"],
                    "modality":          ps["modality"],
                    "modality_index":    ps["modality_index"],
                    "dicom_session_dir": ps["session_dir"],
                    **scan_paths(spss_id, site_abbr, ps["modality"], ps["modality_index"], derivatives_base, cfd_src=_cfd_src),
                }

            def _make_entry(s: dict) -> dict:
                _cfd = str(cfd_base / f"{folder_id}_{site_abbr}_{s['session_id']}.vtp") if cfd_base else None
                return {
                    "session_id":        s["session_id"],
                    "modality":          s["modality"],
                    "modality_index":    s["modality_index"],
                    "dicom_session_dir": s["session_dir"],
                    **scan_paths(spss_id, site_abbr, s["modality"], s["modality_index"], derivatives_base, cfd_src=_cfd),
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

    print(f"Config saved → {output_json}")
    _print_analysis_report(final_data, stats, spss_not_found_list, no_preop_ct_list, output_json)


def _print_analysis_report(
    final_data: dict,
    stats: dict,
    spss_not_found: list[tuple[str, str]],
    no_preop_ct: list[tuple[str, str]],
    output_json: str,
) -> None:
    """Print analysis report to console and save full detail to <output>_analysis.txt."""
    from collections import Counter

    report_lines: list[str] = []

    def rprint(s=""):
        print(s)
        report_lines.append(str(s))

    def fprint(s=""):
        report_lines.append(str(s))

    # Per-site grouping
    site_data: dict[str, list] = defaultdict(list)
    for p in final_data.values():
        site_data[p["metadata"]["site_short"]].append(p)
    sites = sorted(site_data.keys())

    total_n = len(final_data)
    MW, CW = 38, 16  # metric / column widths — MW matches v1

    def fmt_cell(count, denom, note=""):
        if denom == 0:
            return f"{count:<5} (   -  ){note}".ljust(CW)
        return f"{count:<5} ({100 * count / denom:5.1f}%){note}".ljust(CW)

    def col_header(label, n):
        return f"{label+' (N='+str(n)+')':{CW}}"

    header  = f"{'Metric':{MW}} | {col_header('Total', total_n)}"
    header += "".join(f" | {col_header(s, len(site_data[s]))}" for s in sites)
    sep_len = len(header)
    sep     = "-" * sep_len

    rprint("\n" + "=" * sep_len)
    rprint("FINAL MULTI-MODAL DATASET ANALYSIS (v1)")
    rprint("=" * sep_len)
    rprint(f"  Patients in session index : {stats['total']}")
    rprint(f"  Matched to SPSS           : {stats['matched']}")
    rprint(f"  Not found in SPSS         : {stats['spss_not_found']}")
    rprint(f"  Missing pre-op CT         : {stats['no_preop_ct']}")

    # ---- Scan coverage ----
    rprint("\nSCAN COVERAGE")
    rprint(sep)
    rprint(header)
    rprint(sep)

    cov_metrics = [
        ("Has pre-op CT",    lambda p: p["scans"]["pre_op"] is not None),
        ("Has CFD",          lambda p: p["scans"].get("pre_op") is not None
                                       and p["scans"]["pre_op"].get("cfd") is not None),
        ("Has pre-op other", lambda p: bool(p["scans"].get("pre_op_other"))),
        ("Has post-op CT",   lambda p: any(s["modality"] == "CT"  for s in p["scans"].get("post_op", []))),
        ("Has post-op US",   lambda p: any(s["modality"] == "US"  for s in p["scans"].get("post_op", []))),
        ("Has post-op MR",   lambda p: any(s["modality"] == "MR"  for s in p["scans"].get("post_op", []))),
        ("Has post-op XR",   lambda p: any(s["modality"] == "XR"  for s in p["scans"].get("post_op", []))),
    ]
    for label, fn in cov_metrics:
        row  = f"{label:{MW}} | {fmt_cell(sum(1 for p in final_data.values() if fn(p)), total_n)}"
        row += "".join(
            f" | {fmt_cell(sum(1 for p in site_data[s] if fn(p)), len(site_data[s]))}"
            for s in sites
        )
        rprint(row)
    rprint(sep)

    # ---- Ground truth ----
    GT_METRICS = [
        "endoleak_1a", "endoleak_1b", "endoleak_2", "endoleak_3", "endoleak_5",
        "occlusion", "stenosis", "kinking", "fracture", "migration", "infection",
        "reintervention", "death", "death_surg", "death_hosp",
        "sac_measured", "sac_regression", "sac_stable", "sac_expansion",
    ]
    SAC_RATIO_KEYS = {"sac_regression", "sac_stable", "sac_expansion"}

    rprint("\nGROUND TRUTH SUMMARY")
    rprint(sep)
    rprint(header)
    rprint(sep)

    for metric in GT_METRICS:
        is_sac_ratio = metric in SAC_RATIO_KEYS
        note = "*" if is_sac_ratio else ""
        if is_sac_ratio:
            t_denom = sum(1 for p in final_data.values() if p["ground_truth"].get("sac_measured") == 1)
        else:
            t_denom = total_n
        t_count = sum(1 for p in final_data.values() if p["ground_truth"].get(metric) == 1)
        row = f"{metric:{MW}} | {fmt_cell(t_count, t_denom, note)}"
        for s in sites:
            if is_sac_ratio:
                s_denom = sum(1 for p in site_data[s] if p["ground_truth"].get("sac_measured") == 1)
            else:
                s_denom = len(site_data[s])
            s_count = sum(1 for p in site_data[s] if p["ground_truth"].get(metric) == 1)
            row += f" | {fmt_cell(s_count, s_denom, note)}"
        rprint(row)

    # Data-quality rows below the GT block (same section, separated)
    rprint(sep)
    for dq_label, dq_fn in [
        ("no_baseline_diam",       lambda p: not p["ground_truth"].get("has_baseline_diam", True)),
        ("no_fu_diam (bl present)", lambda p: p["ground_truth"].get("has_baseline_diam") and
                                              not p["ground_truth"].get("has_fu_diam", True)),
    ]:
        t_count = sum(1 for p in final_data.values() if dq_fn(p))
        row = f"{dq_label:{MW}} | {fmt_cell(t_count, total_n)}"
        for s in sites:
            s_count = sum(1 for p in site_data[s] if dq_fn(p))
            row += f" | {fmt_cell(s_count, len(site_data[s]))}"
        rprint(row)

    rprint(sep)
    rprint("* sac_regression / sac_stable / sac_expansion are % of sac_measured, not total")
    rprint("=" * sep_len)

    # ---- Patient-ID detail: txt only ----
    fprint("\n" + "=" * sep_len)
    fprint("DATA QUALITY DETAIL — PATIENT IDs")
    fprint("=" * sep_len)

    # Not found in SPSS
    nf_by_site: dict[str, list] = defaultdict(list)
    for fid, site in spss_not_found:
        nf_by_site[site].append(fid)
    fprint(f"\nNot found in SPSS — {len(spss_not_found)} patients")
    for site, pids in sorted(nf_by_site.items()):
        fprint(f"  {site} ({len(pids)}): {', '.join(pids)}")
    if not spss_not_found:
        fprint("  (none)")

    # Missing pre-op CT
    nc_by_site: dict[str, list] = defaultdict(list)
    for sid, site in no_preop_ct:
        nc_by_site[site].append(sid)
    fprint(f"\nMissing pre-op CT — {len(no_preop_ct)} patients")
    for site, pids in sorted(nc_by_site.items()):
        fprint(f"  {site} ({len(pids)}): {', '.join(pids)}")
    if not no_preop_ct:
        fprint("  (none)")

    # Missing baseline diameter
    bl_missing = [(k, p["metadata"]["site_short"]) for k, p in final_data.items()
                  if not p["ground_truth"].get("has_baseline_diam", True)]
    bl_by_site: dict[str, list] = defaultdict(list)
    for key, site in bl_missing:
        bl_by_site[site].append(key)
    fprint(f"\nMissing Baseline Diameter — {len(bl_missing)} patients total")
    for site, pids in sorted(bl_by_site.items()):
        fprint(f"  {site} ({len(pids)}): {', '.join(pids)}")
    if not bl_missing:
        fprint("  (none)")

    # Missing FU diameter (baseline present)
    fu_missing = [(k, p["metadata"]["site_short"]) for k, p in final_data.items()
                  if p["ground_truth"].get("has_baseline_diam") and
                  not p["ground_truth"].get("has_fu_diam", True)]
    fu_by_site: dict[str, list] = defaultdict(list)
    for key, site in fu_missing:
        fu_by_site[site].append(key)
    fprint(f"\nMissing Follow-up Diameter (baseline present, no FU) — {len(fu_missing)} patients total")
    for site, pids in sorted(fu_by_site.items()):
        fprint(f"  {site} ({len(pids)}): {', '.join(pids)}")
    if not fu_missing:
        fprint("  (none)")

    # Post-op images but no pre-op CT
    postop_no_preop = [
        (k, p["metadata"]["site_short"]) for k, p in final_data.items()
        if p["scans"]["pre_op"] is None and p["scans"].get("post_op")
    ]
    po_by_site: dict[str, list] = defaultdict(list)
    for key, site in postop_no_preop:
        po_by_site[site].append(key)
    fprint(f"\nPost-op images present but NO pre-op CT — {len(postop_no_preop)} patients")
    for site, pids in sorted(po_by_site.items()):
        fprint(f"  {site} ({len(pids)}): {', '.join(pids)}")
    if not postop_no_preop:
        fprint("  (none)")

    fprint("=" * sep_len)

    # ---- Per-patient scan table: txt only ----
    fprint("\n" + "=" * sep_len)
    fprint("PER-PATIENT SCAN DETAIL")
    fprint("=" * sep_len)
    fprint(
        f"{'Patient Key':<25} {'PreCT':<6} {'CFD':<5} {'PreOth':<12}"
        f" {'FU-CT':<7} {'FU-US':<7} {'FU-MR':<7} {'FU-XR':<7} Other"
    )
    fprint("-" * 95)
    for key, p in sorted(final_data.items()):
        scans   = p["scans"]
        pre_op  = scans.get("pre_op")
        po      = scans.get("post_op", [])
        poo     = scans.get("pre_op_other", [])
        fu_mods = Counter(s["modality"] for s in po)
        pre_oth = Counter(s["modality"] for s in poo)
        pre_oth_str = ",".join(f"{m}:{n}" for m, n in sorted(pre_oth.items())) or "-"
        fprint(
            f"  {key:<23}"
            f" {'YES' if pre_op else 'NO':<6}"
            f" {'YES' if (pre_op and pre_op.get('cfd')) else 'NO':<5}"
            f" {pre_oth_str:<12}"
            f" {fu_mods.get('CT', 0):<7} {fu_mods.get('US', 0):<7}"
            f" {fu_mods.get('MR', 0):<7} {fu_mods.get('XR', 0):<7}"
            f" {sum(v for k, v in fu_mods.items() if k not in ('CT', 'US', 'MR', 'XR'))}"
        )
    fprint("=" * sep_len)

    # Save
    analysis_txt = output_json.rsplit(".json", 1)[0] + "_analysis.txt"
    with open(analysis_txt, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nAnalysis report saved → {analysis_txt}")


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
        "clinical_data/session_index_v1.json"
    )
    DERIVATIVES_BASE = "/deepstore/datasets/mia/AI4EVAR/derivatives"
    OUTPUT_JSON = (
        "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2/"
        "clinical_data/radar_multimodal_config_v1.json"
    )

    CFD_RENAMED = "/deepstore/datasets/mia/AI4EVAR/raw_data/CFDs/renamed_v1"

    build_multimodal_dataset_v1(SAV_PATH, SESSION_INDEX, DERIVATIVES_BASE, OUTPUT_JSON, cfd_renamed_dir=CFD_RENAMED)
