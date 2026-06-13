"""
Config for the project.

This file defines import paths, device setup, and shared constants.
"""
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths — computed once, relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# Data — keep all raw/processed data under datasets/
DATASETS     = REPO_ROOT / "datasets"

# Keep per-cohort h5ad files in this directory
# e.g.  datasets/processed_h5ad/GSE189125_pre_ici.h5ad
H5AD_DIR     = DATASETS / "processed_h5ad"

# Store pathway prior
PRIOR_NPZ    = DATASETS / "resources" / "pathway_prior.npz"
SIM_PRIOR_NPZ = DATASETS / "simulation_rs" / "sim_pathway_prior.npz"
GMT_FILE     = DATASETS / "resources" / "h.all.v2026.1.Hs.symbols.gmt"


def cohort_h5ad(cohort_id: str) -> Path:
    return H5AD_DIR / f"{cohort_id}.h5ad"

# Results directory
import os
RESULTS_IRAEGIS    = Path(os.environ.get("IRAEGIS_RESULTS_DIR", REPO_ROOT / "results" / "iraegis"))
RESULTS_BASELINES = Path(os.environ.get("IRAEGIS_BASELINES_DIR", REPO_ROOT / "results" / "baselines"))

# ---------------------------------------------------------------------------
# Device & reproducibility
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Seed
RANDOM_STATE = 0

# ---------------------------------------------------------------------------
# Cohort definitions
# ---------------------------------------------------------------------------
COHORTS_REAL = [
    "GSE216329_integrated_pre_ici",
    "GSE249898_integrated_pre_ici",
    "GSE285888_pre_ici",
    "GSE189125_pre_ici",
]
COHORTS_SIM = [
    "RS_cohort1",
    "RS_cohort2",
    "RS_cohort3",
    "DS_cohort1",
    "DS_cohort2",
    "DS_cohort3",
]
COHORT_SHORT = {
    "GSE216329_integrated_pre_ici": "GSE216329",
    "GSE249898_integrated_pre_ici": "GSE249898",
    "GSE285888_pre_ici":            "GSE285888",
    "GSE189125_pre_ici":            "GSE189125",
    "RS_cohort1": "RS_1", "RS_cohort2": "RS_2", "RS_cohort3": "RS_3",
    "DS_cohort1": "DS_1", "DS_cohort2": "DS_2", "DS_cohort3": "DS_3",
}

# ---------------------------------------------------------------------------
# Standardized CV folds — all models use these for fair comparison
# ---------------------------------------------------------------------------
CV_MODE = "loocv"
CV_N_SPLITS  = 3
CV_N_REPEATS = 3

# Fallback to k-fold when LOOCV becomes computationally infeasible.
# Triggered only when CV_MODE == "loocv" and n_patients >= CV_AUTO_FALLBACK_N.
CV_AUTO_FALLBACK_N      = 100
CV_AUTO_FALLBACK_SPLITS = 5

# ---------------------------------------------------------------------------
# Data QC thresholds — applied uniformly to all models before training
# ---------------------------------------------------------------------------
QC_MIN_GENES_PER_CELL = 200      # drop cells expressing fewer genes
QC_MIN_CELLS_PER_PAT_CT = 30     # min cells per patient per CT
QC_MIN_PATIENT_FRACTION = 0.0    # disabled — informative CTs can be sparse

# ---------------------------------------------------------------------------
# Gene selection — pathway genes + top HVG_BACKFILL non-pathway HVGs
# ---------------------------------------------------------------------------
HVG_BACKFILL = 2000


def get_cv_folds(patients, labels):
    """Return list of (train_idx, test_idx) tuples for standardized CV.

    Given a sorted patient list and binary labels, every model calling this
    function with the same cohort data will get identical folds.

    Parameters
    ----------
    patients : list[str]
        Patient IDs — will be sorted internally for determinism.
    labels : array-like
        Binary (0/1) label per patient, aligned with `patients`.

    Returns
    -------
    list[tuple[ndarray, ndarray]]
        Each element is (train_indices, test_indices) into the *sorted*
        patient array.  Also returns the sorted patient list.
    """
    import numpy as np

    order = np.argsort(patients)
    sorted_patients = [patients[i] for i in order]
    sorted_labels = np.asarray(labels)[order]

    eff = effective_cv_mode(len(sorted_patients))
    if eff == "loocv":
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        folds = list(loo.split(np.arange(len(sorted_patients))))
    elif eff == "kfold":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(
            n_splits=CV_AUTO_FALLBACK_SPLITS, shuffle=True,
            random_state=RANDOM_STATE)
        folds = list(skf.split(np.arange(len(sorted_patients)), sorted_labels))
    else:
        from sklearn.model_selection import RepeatedStratifiedKFold
        rcv = RepeatedStratifiedKFold(
            n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS,
            random_state=RANDOM_STATE)
        folds = list(rcv.split(np.arange(len(sorted_patients)), sorted_labels))

    return sorted_patients, sorted_labels, folds


def effective_cv_mode(n_patients: int) -> str:
    """Resolve CV mode at call time.

    Returns "loocv", "kfold" (auto-fallback), or "repeated_stratified".
    """
    if CV_MODE == "loocv" and n_patients >= CV_AUTO_FALLBACK_N:
        return "kfold"
    return CV_MODE
