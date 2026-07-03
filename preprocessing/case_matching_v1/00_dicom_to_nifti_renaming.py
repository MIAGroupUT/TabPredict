"""
00_dicom_to_nifti_renaming.py — DICOM → NIfTI conversion for data (all sessions)
================================================================================

Converts every session found in data/ to NIfTI format.
All modalities are converted — CT, US, MR, XR, etc. — to support training
on the full imaging archive. nnUNet (script 05) only uses CT when running inference.

Two XNAT session layouts are handled:

  Layered (AUMC, UMCG)
  ---------------------
  {patient_id}/
    {session_id}/          ← one dir per modality visit; name encodes modality
      scans/{series}/resources/DICOM/

  Flat (MST, RNST)
  -----------------
  {patient_id}/
    {patient_id}/          ← single session dir, same name as patient
      scans/{series}/resources/DICOM/   ← different modalities are different series

Output flat directory
---------------------
imagesTr_all/
  {raw_patient_id}_{site_abbr}_{session_id}.nii.gz

For flat-session sites a synthetic session_id "{patient_id}_{modality}_1" is used
so the filename encodes the modality consistently across all sites.
"""

import re
from pathlib import Path

import pydicom
import SimpleITK as sitk


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

# AUMC/UMCG session name pattern: {xnat_id}_{MODALITY}_{local_index}
_AUMC_SESSION_RE = re.compile(r'^.+_([A-Z]{2,3})_\d+$')

# DICOM Modality tag → pipeline label.
# CR (computed radiography) and DX (digital radiography) are both plain X-rays.
# Some sites encode them as _XR_ in session names, others as _CR_ — both normalise to XR.
_DICOM_MODALITY_MAP = {
    "CR": "XR",
    "DX": "XR",
    "PT": "PET",   # DICOM standard tag for PET; use PET in filenames/index
}

# Imaging modalities we want to convert.  Anything not in this set is ignored —
# safer than a denylist since the DICOM standard has dozens of non-image types
# (SR, KO, PR, REG, RTSTRUCT, …) and new ones are added over time.
#   CT   — computed tomography (also the CT component of PET-CT)
#   MR   — magnetic resonance imaging
#   PET  — positron emission tomography (raw DICOM tag PT, mapped above)
#   XR   — plain X-ray (raw tags CR / DX, both mapped above)
#   US   — ultrasound
_IMAGE_MODALITIES = {"CT", "MR", "PET", "XR", "US"}


# =========================================================
# HELPERS
# =========================================================

def read_dicom_modality(dicom_dir: Path) -> str:
    """Return the normalised modality label from the first readable DICOM file."""
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


def is_axial(ds) -> bool:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return False
    row, col = iop[:3], iop[3:]
    return (
        abs(row[0] - 1) < 0.1 and abs(row[1]) < 0.1 and abs(row[2]) < 0.1
        and abs(col[1] - 1) < 0.1 and abs(col[0]) < 0.1 and abs(col[2]) < 0.1
    )


def score_dicom_series(dicom_dir: Path) -> float:
    """Score a DICOM series — higher means more likely to be the desired axial CTA.

    Resolution (slice thickness + slice count) is weighted heavily so that a
    0.625 mm thin-slice series always wins over a 3 mm or 5 mm reconstruction.
    """
    files = [f for f in dicom_dir.rglob("*") if f.is_file()]
    if not files:
        return -1.0

    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    except Exception:
        return -1.0

    score = 0.0

    if is_axial(ds):
        score += 5
    else:
        score -= 3

    desc = str(getattr(ds, "SeriesDescription", "")).lower()
    if "arter" in desc:  score += 4
    if "cta"   in desc:  score += 4
    if any(x in desc for x in ["contrast", "ce"]):  score += 2
    if "ax"    in desc:  score += 2
    if any(x in desc for x in ["cor", "sag"]):       score -= 5
    if any(x in desc for x in ["scout", "localizer", "topogram", "surview"]): score -= 10

    # Resolution: strongly prefer the thinnest available slice reconstruction.
    # Sub-mm (0.625/0.75 mm) > 1 mm > 2 mm >> 3–5 mm >> >5 mm.
    thickness = getattr(ds, "SliceThickness", None)
    if thickness is not None:
        try:
            t = float(thickness)
            if t <= 0.75:    score += 8
            elif t <= 1.25:  score += 6
            elif t <= 2.0:   score += 4
            elif t <= 5.0:   score += 0
            else:            score -= 6
        except Exception:
            pass

    # Slice count: thin-slice series have many more slices for the same anatomy.
    n = len(files)
    if n > 300:    score += 6
    elif n > 150:  score += 4
    elif n > 80:   score += 2
    elif n > 30:   score += 1
    elif n < 20:   score -= 5

    return score


def find_best_dicom_dir(session_dir: Path, modality: str) -> Path | None:
    """
    Walk session_dir/scans/{series}/resources/DICOM/ and return the best series.

    Only directories named "DICOM" that sit directly under a "resources" folder are
    considered — this excludes SNAPSHOT/secondary resource folders that may also
    contain a subdirectory called "DICOM".

    CT: uses CTA-specific scoring (arterial phase, axial, thin-slice).  The highest-
        resolution (thinnest-slice) series wins.
    All other modalities: picks the DICOM dir with the most files (largest volume/cine),
    discarding obvious scouts (< 5 files).
    Returns None if no matching DICOM series is found.
    """
    candidates = []
    try:
        dicom_dirs = sorted(
            (d for d in session_dir.rglob("files") if d.is_dir() and d.parent.name == "DICOM"),
            key=lambda p: str(p),
        )
    except (PermissionError, OSError) as e:
        print(f"    [!]  Cannot walk {session_dir}: {e}")
        return None

    for dicom_dir in dicom_dirs:
        series_modality = read_dicom_modality(dicom_dir)
        if series_modality not in _IMAGE_MODALITIES or series_modality != modality:
            continue

        if modality == "CT":
            s = score_dicom_series(dicom_dir)
        else:
            files = [f for f in dicom_dir.rglob("*") if f.is_file()]
            s = len(files) if len(files) >= 5 else -1  # skip tiny scouts

        candidates.append((s, dicom_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_dir = candidates[0]
    series_name = best_dir.parent.parent.parent.name
    print(f"    Best series: {series_name} (score={best_score:.0f})")
    return best_dir


def _enumerate_modalities(session_dir: Path) -> list[str]:
    """Return sorted list of distinct image modalities found in a session directory.

    Non-image series (SR, KO, etc.) are excluded.
    """
    seen: set[str] = set()
    try:
        for dicom_dir in session_dir.rglob("files"):
            if not (dicom_dir.is_dir() and dicom_dir.parent.name == "DICOM"):
                continue
            mod = read_dicom_modality(dicom_dir)
            if mod in _IMAGE_MODALITIES:
                seen.add(mod)
    except (PermissionError, OSError):
        pass
    return sorted(seen)


def convert_to_nifti(dicom_dir: Path) -> sitk.Image | None:
    """Read a DICOM series and reorient to canonical RAS.

    Without this, the saved NIfTI keeps whatever direction cosines the native
    DICOM series had (varies by scanner, patient positioning, slice order) —
    not guaranteed consistent across patients/sites. DICOMOrient is a lossless
    axis permutation/flip (no resampling/interpolation), and RAS matches the
    orientation the MONAI training pipeline reorients to at load time
    (src/data/loaders.py: Orientationd(axcodes="RAS")) — so raw NIfTIs used
    directly by downstream steps (nnUNet, metrics, CFD) are already consistent
    before that transform ever runs.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        return None
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0]))
    img = reader.Execute()
    return sitk.DICOMOrient(img, "RAS")


def session_modality(session_id: str, site_abbr: str, session_dir: Path) -> str:
    """Determine the imaging modality for a layered-session (AUMC/UMCG) site.

    DICOM Modality tag is always the primary source — authoritative and robust across
    sites.  Session name parsing is a fallback for AUMC only, used when DICOM files
    are unreadable.
    Not called for flat-session sites (MST/RNST) — those use _enumerate_modalities().
    """
    dicom_dirs = sorted(
        (d for d in session_dir.rglob("files") if d.is_dir() and d.parent.name == "DICOM"),
        key=lambda p: str(p),
    )
    for dicom_dir in dicom_dirs:
        mod = read_dicom_modality(dicom_dir)
        if mod in _IMAGE_MODALITIES:
            return mod
    # Fallback: parse from AUMC session name if DICOM is unreadable
    if site_abbr == "AUMC":
        m = _AUMC_SESSION_RE.match(session_id)
        if m:
            raw = m.group(1)
            return _DICOM_MODALITY_MAP.get(raw, raw)
    return "UNKNOWN"


# =========================================================
# MAIN CONVERSION LOOP
# =========================================================

def convert_all_sessions(DATA_ROOT: str, output_dir: str) -> None:
    """
    Walk data/ and convert every imaging session to NIfTI.

    Flat-session sites (MST, RNST): the single session dir is enumerated for all
    modalities and a synthetic session_id "{patient_id}_{modality}_1" is used.
    Layered-session sites (AUMC, UMCG): each session subdir encodes one modality.

    Output filename: {raw_patient_id}_{site_abbr}_{session_id}.nii.gz
    """
    root = Path(DATA_ROOT)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for center_dir in root.iterdir():
        if not center_dir.is_dir():
            continue

        site_abbr = SITE_MAP.get(center_dir.name)
        if site_abbr is None:
            print(f"[SKIP] Unknown centre folder: {center_dir.name}")
            continue

        # One level deeper is the XNat study folder (e.g. "The ART in EVAR study-AUMC")
        study_dirs = [d for d in center_dir.iterdir() if d.is_dir()]
        if not study_dirs:
            continue
        study_dir = study_dirs[0]  # always one study per centre

        print(f"\n=== Site: {site_abbr} ===")

        for patient_dir in sorted(study_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            raw_patient_id = patient_dir.name
            print(f"\n  Patient: {raw_patient_id}")

            if site_abbr in _FLAT_SESSION_SITES:
                # ---- Flat layout: one session dir holds all modalities as series ----
                flat_session_dir = patient_dir / raw_patient_id
                if not flat_session_dir.is_dir():
                    subdirs = sorted(d for d in patient_dir.iterdir() if d.is_dir())
                    flat_session_dir = subdirs[0] if subdirs else None
                if flat_session_dir is None:
                    print(f"    [!]  No session dir found under {patient_dir}")
                    continue

                for modality in _enumerate_modalities(flat_session_dir):
                    synthetic_session_id = f"{raw_patient_id}_{modality}_1"
                    out_name = f"{raw_patient_id}_{site_abbr}_{synthetic_session_id}.nii.gz"
                    out_path = out / out_name

                    if out_path.exists():
                        print(f"    [{modality}] {synthetic_session_id} — already converted, skip")
                        continue

                    print(f"    [{modality}] {synthetic_session_id} — converting …")
                    best_dicom = find_best_dicom_dir(flat_session_dir, modality)
                    if best_dicom is None:
                        print(f"    [!]  No valid {modality} DICOM found in {flat_session_dir}")
                        continue

                    try:
                        img = convert_to_nifti(best_dicom)
                        if img is None:
                            print(f"    [!]  DICOM read failed: {best_dicom}")
                            continue
                        sitk.WriteImage(img, str(out_path))
                        print(f"    [OK] Saved → {out_name}")
                    except Exception as e:
                        print(f"    [!]  Conversion error for {synthetic_session_id}: {e}")

            else:
                # ---- Layered layout: each session subdir is one modality visit ----
                for session_dir in sorted(patient_dir.iterdir()):
                    if not session_dir.is_dir():
                        continue

                    session_id = session_dir.name
                    modality   = session_modality(session_id, site_abbr, session_dir)

                    if modality == "UNKNOWN":
                        print(f"    [??] {session_id} — unknown modality {modality}, skipped")
                        continue

                    out_name = f"{raw_patient_id}_{site_abbr}_{session_id}.nii.gz"
                    out_path = out / out_name

                    if out_path.exists():
                        print(f"    [{modality}] {session_id} — already converted, skip")
                        continue

                    print(f"    [{modality}] {session_id} — converting …")
                    best_dicom = find_best_dicom_dir(session_dir, modality)
                    if best_dicom is None:
                        print(f"    [!]  No valid {modality} DICOM found in {session_dir}")
                        continue

                    try:
                        img = convert_to_nifti(best_dicom)
                        if img is None:
                            print(f"    [!]  DICOM read failed: {best_dicom}")
                            continue
                        sitk.WriteImage(img, str(out_path))
                        print(f"    [OK] Saved → {out_name}")
                    except Exception as e:
                        print(f"    [!]  Conversion error for {session_id}: {e}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    DATA_ROOT  = "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/data"
    OUTPUT_DIR = "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr_all"

    convert_all_sessions(DATA_ROOT, OUTPUT_DIR)
