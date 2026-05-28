"""
01_build_session_index.py — Build a per-patient session index from data
==========================================================================================

Walks data/ and produces a JSON index of every imaging session per patient,
with modality, scan date, and directory paths.

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
          "session_id":     "02637442_CT_1",
          "modality":       "CT",
          "modality_index":  1,
          "scan_date":      "2018-04-23",   // ISO date from DICOM StudyDate; null if unreadable
          "session_dir":    "/deepstore/.../02637442/02637442_CT_1"
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
          "session_id":     "0_CT_1",
          "modality":       "CT",
          "modality_index":  1,
          "scan_date":      "2005-03-14",
          "session_dir":    "/deepstore/.../0/0"
        }
      ]
    }
  }
}

Scan date extraction
--------------------
StudyDate (0008,0020) is read from the first DICOM file in each session and stored as
"scan_date" (ISO YYYY-MM-DD, or null if unreadable).  Pre-op / post-op classification is
deferred to 03_data_matching_v1.py, which compares scan_date against P_IP_P_Date (implant-
procedure date) from SPSS.

Modality index assignment
-------------------------
Sessions are sorted chronologically by scan_date (nulls last, then by session_id) before
indices are assigned; index 1 = earliest scan for that modality.
UMCG: the raw numeric suffix in the session_id is used as-is (encodes visit order).
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
    "ARTinEVAR-MST_data":  "MST",   # flat-session layout
    "ARTinEVAR_data":      "RNST",  # flat-session layout
}

_FLAT_SESSION_SITES = {"MST", "RNST"}

_AUMC_SESSION_RE = re.compile(r'^.+_([A-Z]{2,3})_(\d+)$')

_DICOM_MODALITY_MAP = {
    "CR": "XR",
    "DX": "XR",
    "PT": "PET",
}

# Imaging modalities we want to index.  Anything not in this set is ignored —
# safer than a denylist since the DICOM standard has dozens of non-image types
# (SR, KO, PR, REG, RTSTRUCT, …) and new ones are added over time.
#   CT   — computed tomography (also the CT component of PET-CT)
#   MR   — magnetic resonance imaging
#   PET  — positron emission tomography (raw DICOM tag PT, mapped above)
#   XR   — plain X-ray (raw tags CR / DX, both mapped above)
#   US   — ultrasound
_IMAGE_MODALITIES = {"CT", "MR", "PET", "XR", "US"}


# =========================================================
# DICOM TAG READING
# =========================================================

def _read_first_dicom_tags(dicom_dir: Path) -> tuple[str, str | None]:
    """Return (modality, scan_date) from the first readable DICOM file in dicom_dir.

    scan_date is ISO YYYY-MM-DD from StudyDate (0008,0020), or None if unavailable.
    """
    for f in dicom_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            raw_mod = str(getattr(ds, "Modality", "UNKNOWN")).upper()
            modality = _DICOM_MODALITY_MAP.get(raw_mod, raw_mod)
            raw_date = str(getattr(ds, "StudyDate", "") or "").strip()
            scan_date = (
                f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                if len(raw_date) == 8 else None
            )
            return modality, scan_date
        except Exception:
            continue
    return "UNKNOWN", None


def read_dicom_tags(session_dir: Path) -> tuple[str, str | None]:
    """Search session_dir recursively for DICOM dirs; return (modality, scan_date).

    Non-image series (SR dose reports, KO, etc.) are skipped so that a CT session
    containing an embedded SR series is still labelled as CT.
    """
    try:
        dicom_dirs = sorted(
            (d for d in session_dir.rglob("files") if d.is_dir() and d.parent.name == "DICOM"),
            key=lambda p: str(p),
        )
    except (PermissionError, OSError):
        return "UNKNOWN", None

    for dicom_dir in dicom_dirs:
        mod, scan_date = _read_first_dicom_tags(dicom_dir)
        if mod in _IMAGE_MODALITIES:
            return mod, scan_date
    return "UNKNOWN", None


def get_session_modality(session_id: str, site_abbr: str, session_dir: Path) -> tuple[str, str | None]:
    """Return (modality, scan_date) for a layered-session site (AUMC/UMCG).

    DICOM tags are always the primary source — the Modality tag is authoritative and
    robust across sites.  Session name parsing is a fallback for AUMC only, used when
    DICOM files are unreadable (e.g. permission error or corrupt archive).
    """
    mod, scan_date = read_dicom_tags(session_dir)
    if mod in _IMAGE_MODALITIES:
        return mod, scan_date
    # Fallback: parse modality from AUMC session name if DICOM is unreadable
    if site_abbr == "AUMC":
        m = _AUMC_SESSION_RE.match(session_id)
        if m:
            raw = m.group(1)
            return _DICOM_MODALITY_MAP.get(raw, raw), None
    return "UNKNOWN", None


# =========================================================
# FLAT-SESSION BUILDER  (MST, RNST)
# =========================================================

def _build_sessions_flat(patient_dir: Path, patient_id: str) -> list[dict]:
    """
    For flat-session sites (MST, RNST): the patient folder contains a single session
    directory holding all modalities as series.
    Returns one entry per distinct modality, with scan_date from DICOM StudyDate.
    Synthetic session_id follows {patient_id}_{modality}_1.
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
        mod, scan_date = _read_first_dicom_tags(dicom_dir)
        if mod not in _IMAGE_MODALITIES or mod in seen:
            continue
        seen.add(mod)
        sessions.append({
            "session_id":  f"{patient_id}_{mod}_1",
            "modality":    mod,
            "scan_date":   scan_date,
            "session_dir": str(session_dir),
        })

    return sessions


# =========================================================
# MAIN INDEX BUILDER
# =========================================================

def build_session_index(DATA_ROOT: str) -> dict:
    """Walk data/ and return a nested dict with all per-patient session metadata."""
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
                    modality, scan_date = get_session_modality(session_id, site_abbr, session_dir)
                    if modality not in _IMAGE_MODALITIES:
                        continue
                    sessions_raw.append({
                        "session_id":  session_id,
                        "modality":    modality,
                        "scan_date":   scan_date,
                        "session_dir": str(session_dir),
                    })

            # ---- Assign modality_index chronologically by scan_date ----
            modality_groups: dict = defaultdict(list)
            for s in sessions_raw:
                modality_groups[s["modality"]].append(s)

            for mod, mod_sessions in modality_groups.items():
                if site_abbr == "UMCG":
                    # UMCG raw numeric suffix encodes visit order directly
                    for s in mod_sessions:
                        raw_idx = s["session_id"][len(patient_id):]
                        s["modality_index"] = int(raw_idx) if raw_idx.isdigit() else None
                    # Sequential fallback for any unresolved indices
                    max_idx = max(
                        (s["modality_index"] for s in mod_sessions if s["modality_index"] is not None),
                        default=0,
                    )
                    for i, s in enumerate(s for s in mod_sessions if s["modality_index"] is None):
                        s["modality_index"] = max_idx + i + 1
                else:
                    ordered = sorted(
                        mod_sessions,
                        key=lambda s: (s.get("scan_date") is None, s.get("scan_date") or "", s["session_id"]),
                    )
                    for i, s in enumerate(ordered):
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
    OUTPUT_JSON = (
        "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2/"
        "clinical_data/session_index_v1.json"
    )

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
