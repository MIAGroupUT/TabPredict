import json
import pandas as pd
import pyreadstat
import os
import re
import datetime
import numpy as np
from pathlib import Path


def serialize_value(val):
    """
    Convert raw SPSS / numpy / Python values into JSON-safe types.

    Purpose
    -------
    Ensures all values can be safely written into a JSON dataset.

    Handles
    -------
    - datetime → ISO string
    - NaN / NA → None
    - numpy numeric types → native Python int/float
    - bytes → UTF-8 decoded string
    - everything else → unchanged
    """
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.isoformat()
    if pd.isna(val) or val is pd.NA:
        return None
    if isinstance(val, (np.int64, np.int32, np.float64, np.float32)):
        return float(val) if isinstance(val, (float, np.float64, np.float32)) else int(val)
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='ignore')
    return val


def is_flagged(val):
    """
    Convert SPSS-style binary indicators into Python boolean flags.

    Purpose
    -------
    Many SPSS variables are stored as floats (0.0 / 1.0).
    This function standardizes them into boolean logic.

    Returns
    -------
    bool
        True if value == 1.0, otherwise False.
    """
    try:
        if pd.isna(val) or val is None:
            return False
        return float(val) == 1.0
    except:
        return False


# =========================================================
# SITE CONFIGURATION
# =========================================================
"""
Site-specific configuration for RADAR dataset mapping.

Each entry defines:
- file_pattern: how image filenames encode patient ID
- spss_site_name: exact SPSS site string for matching
- spss_site_abbr: short label used in output dataset
- spss_id_format: transformation rule for ID normalization
"""
SITE_CONFIG = {
    "RNST": {
        "file_pattern": r"(\b\d+)_RNST",
        "spss_site_name": "001 lokatie Arnhem",
        "spss_site_abbr": "RNST",
        "spss_id_format": lambda x: str(int(x)).zfill(4)
    },
    "MST": {
        "file_pattern": r"(\b\d+)_MST",
        "spss_site_name": "007 - Medisch Spectrum Twente (MST)",
        "spss_site_abbr": "MST",
        "spss_id_format": lambda x: str(int(x))
    },
    "AUMC": {
        "file_pattern": r"(\b\d+)_AUMC",
        "spss_site_name": "008 - AmsterdamUMC",
        "spss_site_abbr": "AUMC",
        "spss_id_format": lambda x: str(int(x))
    },
    "UMCG": {
        "file_pattern": r"(\b\d+)_UMCG",
        "spss_site_name": "009 - UMCG",
        "spss_site_abbr": "UMCG",
        "spss_id_format": lambda x: str(int(x))
    }
}


def find_matching_cfd(cfd_files, spss_id, site_name):
    """
    Find CFD (computational fluid dynamics) file matching a patient.

    Parameters
    ----------
    cfd_files : dict
        Mapping of CFD filenames → metadata (ID + site).
    spss_id : str
        Patient ID from SPSS.
    site_name : str
        Full SPSS site name.

    Returns
    -------
    str or None
        CFD filename if found, otherwise None.
    """
    for fn, p_list in cfd_files.items():
        if any(cp['spss_id'] == spss_id and cp['site_name'] == site_name for cp in p_list):
            return fn
    return None


def build_multimodal_dataset(sav_path, image_dir, cfd_dir, output_json):
    """
    Build a multimodal dataset combining:
    - SPSS clinical data
    - CT images (NIfTI)
    - CFD simulations (optional)

    Pipeline Overview
    -----------------
    1. Load SPSS clinical dataset
    2. Map image folders → patient IDs
    3. Map CFD folders → patient IDs
    4. Match patients across modalities (SPSS + image)
    5. Attach CFD if available (optional, NOT required)
    6. Extract:
        - pre-op features
        - post-op features
        - ground truth labels
    7. Export unified JSON dataset

    Parameters
    ----------
    sav_path : str
        Path to SPSS .sav clinical dataset.

    image_dir : str
        Root directory containing CT image files.

    cfd_dir : str
        Root directory containing CFD results.

    output_json : str
        Output path for final multimodal JSON dataset.

    Output Structure
    ----------------
    {
        patient_id_site: {
            metadata: {...},
            paths: {...},
            pre_op_features: {...},
            post_op_features: {...},
            ground_truth: {...}
        }
    }
    """

    print(f"--- Loading Clinical Data: {sav_path} ---")
    df, meta = pyreadstat.read_sav(sav_path)

    # Normalize identifiers for matching
    df['patient_str'] = df['PatientNr'].astype(str).str.split('.').str[0]
    df['site_str'] = df['Site'].astype(str).str.strip()

    def get_file_map(directory):
        """
        Build mapping from filename → possible (patient_id, site).

        Used for both:
        - CT images
        - CFD files

        Returns
        -------
        dict
            filename → list of candidate (spss_id, site_name)
        """
        fmap = {}
        if not os.path.exists(directory):
            return fmap

        for f in os.listdir(directory):
            if not ('.nii.gz' in f or '.vtp' in f):
                continue

            potential_matches = []

            for site_key, config in SITE_CONFIG.items():
                match = re.search(config['file_pattern'], f)
                if match:
                    raw_id = match.group(1)
                    spss_id = config['spss_id_format'](raw_id)

                    potential_matches.append({
                        "spss_id": spss_id,
                        "site_name": config['spss_site_name']
                    })

            if potential_matches:
                fmap[f] = potential_matches

        return fmap

    # Load modalities
    img_files = get_file_map(image_dir)
    cfd_files = get_file_map(cfd_dir)

    # Ground truth definitions
    # comp_patterns = {
    #     "endoleak": r"Endograft_Spc.*answer0",
    #     "migration": r"Endograft_Spc.*answer1",
    #     "component_separation": r"Endograft_Spc.*answer2",
    #     "kinking": r"Endograft_Spc.*answer3",
    #     "occlusion": r"Endograft_Spc.*answer4",
    #     "stenosis": r"Endograft_Spc.*answer5",
    #     "aneurysm_rupture": r"Endograft_Spc.*answer6",
    #     "infection": r"Endograft_Spc.*answer7"
    # }

    comp_patterns = {
        "endoleak": r"Endograft_Spc.*answer0",            # 0.0 = Endoleak
        "migration": r"Endograft_Spc.*answer1",           # 1.0 = Endograft migration
        "infection": r"Endograft_Spc.*answer2",           # 2.0 = Endograft infection
        "erosion": r"Endograft_Spc.*answer3",             # 3.0 = Device erosion
        "limb_obstruction": r"Endograft_Spc.*answer4",    # 4.0 = Endograft limb obstruction
        "claudication": r"Endograft_Spc.*answer5",        # 5.0 = Buttock/leg claudication
        "occlusion": r"^Occlusion(_\d+)*$",               # Dedicated Occlusion columns
        "rupture": r"Endograft_Spc.*answer6"              # This was "Bleeding" in the list, verify if Rupture is separate
    }

    final_data = {}
    stats = {k: 0 for k in list(comp_patterns.keys()) + ["total", "reint", "death", "death_surg", "death_hosp",\
             "sac regression (using max)", "stable sac (using max)", "sac expansion (using max)", "sac regression (using min)", "stable sac (using min)", "sac expansion (using min)"]}
    matched_df_list = []

    # Pre-op / Post-op feature split
    pre_op_cols = [
        c for c in df.columns
        if (c.startswith('Bl_') or c in ['Sexe', 'Age', 'Weight', 'Height'])
        and "_AE_SAE_" not in c
    ]

    post_op_cols = [
        c for c in df.columns
        if c not in pre_op_cols and c not in ['PatientNr', 'patient_str', 'site_str']
    ]

    # =====================================================
    # MAIN MATCHING LOOP
    # =====================================================
    for img_fname, possibilities in img_files.items():
        for p in possibilities:

            spss_id = p['spss_id']
            site_name = p['site_name']
            site_abbr = next(
                (cfg['spss_site_abbr'] for cfg in SITE_CONFIG.values()
                 if cfg['spss_site_name'] == site_name),
                "UNKNOWN"
            )

            mask = (df['patient_str'] == spss_id) & (df['site_str'] == site_name)
            patient_row_df = df[mask]

            if patient_row_df.empty:
                continue

            # Optional CFD (NOT required)
            cfd_fname = find_matching_cfd(cfd_files, spss_id, site_name)

            row = patient_row_df.iloc[0]
            matched_df_list.append(patient_row_df)

            # ---- Ground truth extraction ----
            reint_summary = is_flagged(row.get('ReintervAAArel_0m18m_yn'))
            reint_ae_cols = [c for c in df.columns if re.search(r'AEAction.*answer4', c)]
            reint_ae = any(is_flagged(row.get(c)) for c in reint_ae_cols)
            has_reint = int(reint_summary or reint_ae)

            # ---- Comprehensive Death Extraction ----
            # 1. Direct Tags from rAAA (Surgery/ER)
            d_surg_raw = is_flagged(row.get('Bl_Bl_rAAA_death_surg'))           # Death during surgery
            d_hosp_raw = is_flagged(row.get('Bl_Bl_rAAA_deathER'))              # Death during hospital
            
            # 2. General Follow-up Death Tags
            death_ae = is_flagged(row.get('AE0_AE_AE_Death'))        # Primary AE Death tag
            
            # 3. SAE "Reason Serious" Check (Mapping shows 0.0 = Death)
            # Scan all Reason tags (Reason, Reason_1, Reason_2, etc.)
            reason_cols = [c for c in df.columns if re.search(r'^Reason(_\d+)?$', c)]
            death_by_reason = any(serialize_value(row.get(c)) == 0 for c in reason_cols) # 0.0 is 'Death'

            has_death_surg = int(d_surg_raw)
            has_death_hosp = int(d_hosp_raw)
            has_death = int(death_ae or d_surg_raw or d_hosp_raw or death_by_reason)

            # diam_diff = row.get('AAArem_con_diamdiff_baseline465p')
            # reg = int(not pd.isna(diam_diff) and diam_diff <= -5.0)
            # stable = int(not pd.isna(diam_diff) and diam_diff >= 0.0)

            # ---- Manual Sac Dynamics Calculation ----
            # 1. Identify Baseline Diameter
            # We prioritize the three specific fields mentioned: Bl_AAAdiam, Bl_Bl_Im_CTA_AAAdiam, Bl_Bl_Im_US_AAAdiam
            # We also include a general regex for any other "Baseline" diameter tags found in the mapping
            baseline_patterns = [
                'Bl_AAAdiam', 
                'Bl_Bl_Im_CTA_AAAdiam', 
                'Bl_Bl_Im_US_AAAdiam',
                r'CTA_AAAdiam', # standard mapping tag
                r'US_AAAdiam'   # standard mapping tag[cite: 1, 3]
            ]
            
            baseline_vals = []
            for pat in baseline_patterns:
                # Find columns matching the specific name or pattern
                cols = [c for c in df.columns if re.search(pat, c, re.I)]
                for c in cols:
                    val = row.get(c)
                    if pd.notna(val):
                        baseline_vals.append(float(val))
            
            # Use the largest available baseline diameter as the reference[cite: 1, 3]
            baseline_ref = max(baseline_vals) if baseline_vals else None
            
            # 2. Identify Follow-up Diameters
            # We look for all tags starting with 'F' (Follow-up) followed by 'AAAdiam'
            fu_regex = r'^(F.*AAAdiam|.*AAAdiam_.*(1yr|6m|12m|yr))'
            fu_cols = [c for c in df.columns if re.search(fu_regex, c, re.I) and 'NE' not in c]
            fu_values = [float(row.get(c)) for c in fu_cols if pd.notna(row.get(c))]
            
            if baseline_ref is not None and fu_values:
                # To identify regression or expansion risk, we compare against the MAX follow-up size[cite: 1, 3]
                min_fu_diam = min(fu_values)    # Use minmum diameter detected during Follow-Up
                max_fu_diam = max(fu_values)    # Use maximum diameter detected during Follow-Up
                diam_diff_using_max = max_fu_diam - baseline_ref
                diam_diff_using_min = min_fu_diam - baseline_ref
                # print(f"Patient {spss_id} | Baseline: {baseline_ref} | Max FU: {max_fu_diam} | Min FU: {min_fu_diam} | Diff: {diam_diff}")

            # 3. Assign Clinical Labels
            # Regression: Decrease of 5mm or more
            # Stable/Expansion: No change or any increase
            reg_using_max = int(diam_diff_using_max is not None and diam_diff_using_max <= -5.0)
            reg_using_min = int(diam_diff_using_min is not None and diam_diff_using_min <= -5.0)
            stable_using_max = int(diam_diff_using_max is not None and diam_diff_using_max == 0.0)
            stable_using_min = int(diam_diff_using_min is not None and diam_diff_using_min == 0.0)
            expansion_using_max = int(diam_diff_using_max is not None and diam_diff_using_max > 0.0)
            expansion_using_min = int(diam_diff_using_min is not None and diam_diff_using_min > 0.0)

            gt_comps = {}
            for label, pattern in comp_patterns.items():
                target_cols = [c for c in df.columns if re.search(pattern, c)]
                found = int(any(is_flagged(row.get(c)) for c in target_cols))
                gt_comps[label] = found
                stats[label] += found

            stats["total"] += 1
            stats["reint"] += has_reint
            stats["death_surg"] += has_death_surg
            stats["death_hosp"] += has_death_hosp
            stats["death"] += has_death
            stats["sac regression (using max)"] += reg_using_max
            stats["stable sac (using max)"] += stable_using_max
            stats["sac expansion (using max)"] += expansion_using_max
            stats["sac regression (using min)"] += reg_using_min
            stats["stable sac (using min)"] += stable_using_min
            stats["sac expansion (using min)"] += expansion_using_min
            unique_key = f"{spss_id}_{site_abbr}"

            # ---- Final structured record ----
            final_data[unique_key] = {
                "metadata": {
                    "id": spss_id,
                    "original_nr": serialize_value(row['PatientNr']),
                    "site": site_name,
                    "site_short": site_abbr,
                    "img": img_fname,
                    "cfd": cfd_fname if cfd_fname else None
                },
                "paths": {
                    "pre_op_cta": str(Path(image_dir) / img_fname),
                    "pre_op_cfd": str(Path(cfd_dir) / cfd_fname) if cfd_fname else None
                },
                "pre_op_features": {
                    k: serialize_value(row[k]) for k in pre_op_cols if k in row
                },
                "post_op_features": {
                    k: serialize_value(row[k]) for k in post_op_cols if k in row
                },
                "ground_truth": {
                    "reintervention": has_reint,
                    "death": has_death,
                    "death_surg": has_death_surg,
                    "death_hosp": has_death_hosp,
                    "sac_regression_using_max": reg_using_max,
                    "stable_sac_using_max": stable_using_max,
                    "sac_expansion_using_max": expansion_using_max,
                    "sac_regression_using_min": reg_using_min,
                    "stable_sac_using_min": stable_using_min,
                    "sac_expansion_using_min": expansion_using_min,
                    "diam_diff_using_max": serialize_value(diam_diff_using_max),
                    "diam_diff_using_min": serialize_value(diam_diff_using_min),
                    **gt_comps
                }
            }

            break

    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(final_data, f, indent=4)

    # =====================================================
    # ANALYSIS REPORT
    # =====================================================
    if matched_df_list:
        m_df = pd.concat(matched_df_list)

        print("\n" + "=" * 95)
        print("FINAL MULTI-MODAL DATASET ANALYSIS")
        print("=" * 95)
        print(f"Matched Patients: {len(final_data)}")

        pre_sparsity = m_df[pre_op_cols].isna().mean()
        print("\nTOP 10 MOST COMPLETE PRE-OP FEATURES:")
        pre_top = (1 - pre_sparsity).sort_values(ascending=False).head(10)

        for field, score in pre_top.items():
            label = str(meta.column_names_to_labels.get(field, "No Label"))[:45]
            print(f" - {field:25} | {score:>4.0%} filled | {label}")

        print("\nGROUND TRUTH SUMMARY")
        print("-" * 95)
        for k, v in stats.items():
            if k == "total":
                continue
            print(f"{k:40} | {v:<10} | {(v/stats['total'])*100:.1f}%")

        print("=" * 95)


# =====================================================
# EXECUTION
# =====================================================
build_multimodal_dataset(
    "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/1. RADAR_datebase_ARTinEVAR_v4, incl Bl_AAAdiam voor oude Rijnstate patienten.sav",
    "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr",
    "/deepstore/datasets/mia/AI4EVAR/raw_data/CFDs/renamed",
    "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1/clinical_data/radar_multimodal_config_v1.json"
)