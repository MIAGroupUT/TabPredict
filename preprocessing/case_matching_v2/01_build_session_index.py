"""
01_build_session_index.py — Build a per-patient session index from data
==========================================================================================

Walks data/ and produces a JSON index of every imaging session per patient,
with modality, scan index, pre-op flag, and directory paths.

Two XNAT session layouts are handled:

  Layered (AUMC, UMCG)
  ---------------------
  {patient_id}/
    {session_id}/          ← one dir per modality visit; AUMC name encodes modality,
      scans/...              UMCG names are purely numeric

  Flat (MST, RNST)
  -----------------
  {patient_id}/
    {patient_id}/          ← single session dir, same name as patient
      scans/{series}/...   ← different modalities are different series inside

For flat-session sites a synthetic session_id "{patient_id}_{modality}_1" is created
so the index is uniform across all sites.

Output: session_index_v1.json

Session index structure
-----------------------
{
  "AUMC": {
    "02637442": {
      "patient_dir": "/deepstore/.../02637442",
      "sessions": [
        {
          "session_id":    "02637442_CT_1",
          "modality":      "CT",
          "modality_index": 1,
          "is_pre_op":     true,
          "session_dir":   "/deepstore/.../02637442/02637442_CT_1"
        },
        ...
      ]
    }
  },
  "MST": {
    "0": {
      "patient_dir": "/deepstore/.../0",
      "sessions": [
        {
          "session_id":    "0_CT_1",      // synthetic
          "modality":      "CT",
          "modality_index": 1,
          "is_pre_op":     true,
          "session_dir":   "/deepstore/.../0/0"
        }
      ]
    }
  }
}

Pre-op detection rules
----------------------
AUMC:     session whose xnat_id prefix equals the patient folder name AND local index == 1.
MST/RNST: all sessions are pre-op (datasets contain pre-op data only).
UMCG:     scan index == 1 (any modality); index embedded in the numeric session ID.

Modality index assignment
-------------------------
For CT: pre-op always receives modality_index=1; post-op CTs follow in sorted order.
For all other modalities: sorted alphabetically by session_id, starting from 1.
UMCG: the raw numeric suffix is used as-is (encodes chronological visit order).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pydicom


# =========================================================
# SITE CONFIGURATION
# =========================================================
SITE_MAP = {
    "ARTinEVAR-AUMC_data": "AUMC",
    "ARTinEVAR-UMCG_data": "UMCG",
    "ARTinEVAR-MST_data":  "MST",   # pre-op only; flat-session layout
    "ARTinEVAR_data":      "RNST",  # pre-op only; flat-session layout
}

# Sites that store all modalities as series inside a single session dir
_FLAT_SESSION_SITES = {"MST", "RNST"}

_AUMC_SESSION_RE = re.compile(r'^.+_([A-Z]{2,3})_(\d+)$')

# DICOM Modality tag → pipeline label.
# CR and DX are both plain X-rays; some sites use _XR_ in session names, others _CR_.
_DICOM_MODALITY_MAP = {
    "CR": "XR",
    "DX": "XR",
    "PT": "PET",   # DICOM standard tag for PET; use PET in filenames/index
}


# =========================================================
# MODALITY DETECTION
# =========================================================

def _modality_from_dicom_dir(dicom_dir: Path) -> str:
    """Read normalised modality from a specific DICOM directory (no recursion)."""
    for f in dicom_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            raw = str(getattr(ds, "Modality", "UNKNOWN")).upper()
            return _DICOM_MODALITY_MAP.get(raw, raw)
        except Exception:
            continue
    return "UNKNOWN"


def read_dicom_modality(session_dir: Path) -> str:
    """Read normalised modality by searching recursively for DICOM dirs in session_dir."""
    try:
        dicom_dirs = [
            d for d in session_dir.rglob("files")
            if d.is_dir() and d.parent.name == "DICOM"
        ]
    except (PermissionError, OSError):
        return "UNKNOWN"

    for dicom_dir in dicom_dirs:
        mod = _modality_from_dicom_dir(dicom_dir)
        if mod != "UNKNOWN":
            return mod
    return "UNKNOWN"


def get_session_modality(session_id: str, site_abbr: str, session_dir: Path) -> str:
    """
    Determine modality for a layered-session (AUMC/UMCG) site.
    Not called for flat-session sites — those use _build_sessions_flat().
    """
    if site_abbr == "AUMC":
        m = _AUMC_SESSION_RE.match(session_id)
        if m:
            raw = m.group(1)
            return _DICOM_MODALITY_MAP.get(raw, raw)  # CR/DX → XR
    # UMCG (numeric session IDs) and fallback: read modality from DICOM tag
    return read_dicom_modality(session_dir)


# =========================================================
# PRE-OP DETECTION
# =========================================================

def is_preop_session(session_id: str, modality: str, patient_id: str, site_abbr: str) -> bool:
    """
    Return True if this session is pre-operative.
    Only called for layered-session sites (AUMC, UMCG).

    AUMC: xnat_id prefix == patient folder name AND local index == 1 (any modality).
    UMCG: scan index == 1 (any modality); index is embedded in the numeric session ID.
    MST/RNST: handled in _build_sessions_flat() — all are pre-op by definition.
    """
    if site_abbr == "AUMC":
        m = _AUMC_SESSION_RE.match(session_id)
        if m:
            suffix_len = len(m.group(1)) + len(m.group(2)) + 2
            prefix = session_id[:-suffix_len]
            local_index = m.group(2)
            return prefix == patient_id and local_index == "1"
        return False

    if site_abbr == "UMCG":
        # session_id = base_id + scan_index (all digits); index 1 is pre-op for any modality
        if session_id.startswith(patient_id) and session_id[len(patient_id):].isdigit():
            return session_id[len(patient_id):] == "1"

    return False


# =========================================================
# FLAT-SESSION BUILDER  (MST, RNST)
# =========================================================

def _build_sessions_flat(patient_dir: Path, patient_id: str) -> list[dict]:
    """
    For flat-session sites (MST, RNST): the patient folder contains a single session
    subdirectory (named after the patient ID) that holds all modalities as series.
    Returns one synthetic session entry per distinct modality found.

    Synthetic session_id follows the AUMC convention: {patient_id}_{modality}_1
    """
    session_dir = patient_dir / patient_id
    if not session_dir.is_dir():
        subdirs = sorted(d for d in patient_dir.iterdir() if d.is_dir())
        session_dir = subdirs[0] if subdirs else None
    if session_dir is None:
        return []

    seen: set[str] = set()
    sessions: list[dict] = []
    try:
        dicom_dirs = [
            d for d in session_dir.rglob("files")
            if d.is_dir() and d.parent.name == "DICOM"
        ]
    except (PermissionError, OSError):
        return []

    for dicom_dir in dicom_dirs:
        mod = _modality_from_dicom_dir(dicom_dir)
        if mod == "UNKNOWN" or mod in seen:
            continue
        seen.add(mod)
        sessions.append({
            "session_id":  f"{patient_id}_{mod}_1",
            "modality":    mod,
            "is_pre_op":   True,
            "session_dir": str(session_dir),
        })

    return sessions


# =========================================================
# MAIN INDEX BUILDER
# =========================================================

def build_session_index(DATA_ROOT: str) -> dict:
    """
    Walk data/ and return a nested dict with all per-patient session metadata.
    """
    root = Path(DATA_ROOT)
    index = {}

    for center_dir in sorted(root.iterdir()):
        if not center_dir.is_dir():
            continue

        site_abbr = SITE_MAP.get(center_dir.name)
        if site_abbr is None:
            continue

        study_dirs = [d for d in center_dir.iterdir() if d.is_dir()]
        if not study_dirs:
            continue
        study_dir = study_dirs[0]

        site_index = {}

        for patient_dir in sorted(study_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name

            # ---- Collect raw session list ----
            if site_abbr in _FLAT_SESSION_SITES:
                sessions_raw = _build_sessions_flat(patient_dir, patient_id)
            else:
                sessions_raw = []
                for session_dir in sorted(patient_dir.iterdir()):
                    if not session_dir.is_dir():
                        continue
                    session_id = session_dir.name
                    modality   = get_session_modality(session_id, site_abbr, session_dir)
                    preop      = is_preop_session(session_id, modality, patient_id, site_abbr)
                    sessions_raw.append({
                        "session_id":  session_id,
                        "modality":    modality,
                        "is_pre_op":   preop,
                        "session_dir": str(session_dir),
                    })

            # ---- Assign modality_index ----
            # UMCG: raw scan number stripped from session_id (e.g. "3843" → 3 for patient "384").
            # All others: sequential per-modality counter; CT pre-op always gets index 1.
            modality_groups: dict = defaultdict(list)
            for s in sessions_raw:
                modality_groups[s["modality"]].append(s)

            for mod, mod_sessions in modality_groups.items():
                if mod == "CT":
                    preop_list  = [s for s in mod_sessions if s["is_pre_op"]]
                    postop_list = sorted(
                        [s for s in mod_sessions if not s["is_pre_op"]],
                        key=lambda s: s["session_id"],
                    )
                    ordered = preop_list + postop_list
                else:
                    ordered = sorted(mod_sessions, key=lambda s: s["session_id"])

                for i, s in enumerate(ordered):
                    if site_abbr == "UMCG":
                        raw_idx = s["session_id"][len(patient_id):]
                        s["modality_index"] = int(raw_idx) if raw_idx.isdigit() else i + 1
                    else:
                        s["modality_index"] = i + 1

            site_index[patient_id] = {
                "patient_dir": str(patient_dir),
                "sessions":    sessions_raw,
            }

        index[site_abbr] = site_index
        print(f"[{site_abbr}] Indexed {len(site_index)} patients")

    return index


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    DATA_ROOT   = "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/data"
    OUTPUT_JSON = "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2/clinical_data/session_index_v1.json"

    index = build_session_index(DATA_ROOT)

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(index, f, indent=2)

    total_patients = sum(len(v) for v in index.values())
    total_sessions = sum(
        len(p["sessions"])
        for site in index.values()
        for p in site.values()
    )
    print(f"\nSaved session index → {OUTPUT_JSON}")
    print(f"  Patients: {total_patients}  |  Sessions: {total_sessions}")
