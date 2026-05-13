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
    comp_patterns = {
        # --- ENDOLEAKS (Granular) ---
        "endoleak_1a": r"Endoleak_spec_.*answer0",
        "endoleak_1b": r"Endoleak_spec_.*answer(1|2)",
        "endoleak_2":  r"Endoleak_spec_.*answer(3|4|5|6)",
        "endoleak_3":  r"Endoleak_spec_.*answer(7|8)",
        "endoleak_5":  r"Endoleak_spec_.*answer11",
        
        # --- PATENCY & MECHANICAL ---
        "occlusion": r"Occlusion_",
        "stenosis":  r"Stenosis_",
        "kinking":   r"Kinking_",
        "fracture":  r"Fract_",
        
        # --- ENDOGRAFT SPECIFIC ---
        "migration": r"Endograft_Spc.*answer1",
        "infection": r"Endograft_Spc.*answer2",
    }

    final_data = {}
    stats = {k: 0 for k in list(comp_patterns.keys()) + ["total", "reint", "death", "death_surg", "death_hosp",\
             "sac_measured", "sac_regression", "sac_stable", "sac_expansion"]}
    matched_df_list = []

    # Pre-op / Post-op feature split
    pre_op_cols = [
        c for c in df.columns
        if (c.startswith('Bl_')) or c in ['PatientNr', 'CreatedInSite', 'Site', 'Height']
        # if (c.startswith('Bl_') or c in ['Sexe', 'Age', 'Weight', 'Height'])
        # and "_AE_SAE_" not in c
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

            row = patient_row_df.iloc[0]
            # if int(row.get('P_IP_P_DT')) == 9: 
            #     # 9 is Vascutek Anaconda graft type which is no longer used
            #     # ONLY matters for Post-op model
            #     # For pre-op we use every case we can get
            #     continue

            # Optional CFD (NOT required)
            cfd_fname = find_matching_cfd(cfd_files, spss_id, site_name)

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

            # ---- Sac Dynamics Calculation (same-modality pairs preferred) ----
            # Per PI guidance: CT-CT > US-US > CT-US (cross-modality worst case).
            # Baseline: Bl.*CTA.*AAAdiam (CT) | Bl.*(?:US|DUS).*AAAdiam (US) | Bl_AAAdiam (generic)
            # Follow-up: CTA columns (CT) | DUS|_US_ columns (US)
            # FU candidates sorted by embedded visit number; latest non-null value used.
            # Regression: diff <= -5 mm | Stable: |diff| < 5 mm | Expansion: diff >= 5 mm

            bl_ct_cols = [c for c in df.columns if re.search(r'Bl.*CTA.*AAAdiam', c, re.I)]
            bl_ct = next((float(row.get(c)) for c in bl_ct_cols if pd.notna(row.get(c))), None)

            bl_us_cols = [c for c in df.columns if re.search(r'Bl.*(?:US|DUS).*AAAdiam', c, re.I)]
            bl_us = next((float(row.get(c)) for c in bl_us_cols if pd.notna(row.get(c))), None)

            fu_cands_ct, fu_cands_us = [], []
            for c in df.columns:
                if 'NE' in c or 'AAAdiam' not in c or c.startswith('Bl_'):
                    continue
                m = re.search(r'FU(\d+)', c, re.I) or re.search(r'(\d+)yr', c, re.I)
                if not m:
                    continue
                visit_num = int(m.group(1))
                if re.search(r'CTA', c, re.I):
                    fu_cands_ct.append((c, visit_num))
                elif re.search(r'DUS|_US_', c, re.I):
                    fu_cands_us.append((c, visit_num))

            fu_cands_ct.sort(key=lambda x: x[1], reverse=True)
            fu_cands_us.sort(key=lambda x: x[1], reverse=True)
            fu_ct = next((float(row.get(c)) for c, _ in fu_cands_ct if pd.notna(row.get(c))), None)
            fu_us = next((float(row.get(c)) for c, _ in fu_cands_us if pd.notna(row.get(c))), None)

            if bl_ct is not None and fu_ct is not None:       # CT-CT (best)
                bl_val, fu_val = bl_ct, fu_ct
            elif bl_us is not None and fu_us is not None:     # US-US (second)
                bl_val, fu_val = bl_us, fu_us
            else:                                              # CT-US fallback (worst case)
                if bl_ct is None and bl_us is None:
                    bl_gen_cols = [c for c in df.columns if re.search(r'^Bl_AAAdiam$', c)]
                    bl_val = next((float(row.get(c)) for c in bl_gen_cols if pd.notna(row.get(c))), None)
                else:
                    bl_val = bl_ct if bl_ct is not None else bl_us
                fu_val = fu_ct if fu_ct is not None else fu_us
            
            # NOTE: Missing diameter values are expected for some patients, especially if they were lost to follow-up or if measurements were not taken.
            # Not every patient has a diameter baseline reported. There are also patients which have a baseline diameter but no FU diameter is reported.
            # if bl_val is None:
            #     print(f"  WARNING: no baseline diam found for {spss_id}")
            # if fu_val is None:
            #     print(f"  INFO: no FU diam found for {spss_id} (bl={bl_val})")

            diam_diff = (fu_val - bl_val) if (bl_val is not None and fu_val is not None) else None

            sac_regression = int(diam_diff is not None and diam_diff <= -5.0)
            sac_stable     = int(diam_diff is not None and -5.0 < diam_diff < 5.0)
            sac_expansion  = int(diam_diff is not None and diam_diff >= 5.0)

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
            stats["sac_measured"]   += int(diam_diff is not None)
            stats["sac_regression"] += sac_regression
            stats["sac_stable"]     += sac_stable
            stats["sac_expansion"]  += sac_expansion    # diff ≥ +5
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
                    "sac_regression": sac_regression,
                    "sac_stable": sac_stable,
                    "sac_expansion": sac_expansion,
                    "diam_diff": serialize_value(diam_diff),
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
        sac_keys = {"sac_regression", "sac_stable", "sac_expansion"}
        sac_n = stats["sac_measured"] or 1
        for k, v in stats.items():
            if k in ("total", "sac_measured"):
                continue
            denom = sac_n if k in sac_keys else stats["total"]
            label = f" (of {stats['sac_measured']} with measurement)" if k in sac_keys else ""
            print(f"{k:40} | {v:<10} | {(v/denom)*100:.1f}%{label}")

        print(f"{'sac_measured':40} | {stats['sac_measured']:<10} | {(stats['sac_measured']/stats['total'])*100:.1f}% of total")
        print("=" * 95)

# =====================================================
# EXECUTION
# =====================================================
build_multimodal_dataset(
    "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/1. RADAR_datebase_ARTinEVAR_v4, incl Bl_AAAdiam voor oude Rijnstate patienten.sav",
    "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr_pre-op",
    "/deepstore/datasets/mia/AI4EVAR/raw_data/CFDs/renamed",
    "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1/clinical_data/radar_multimodal_config_v1.json"
)