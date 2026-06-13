#!/usr/bin/env python3

# Reference: https://github.com/GENyO-BioInformatics/singleDeep

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.config import (DEVICE, RANDOM_STATE, RESULTS_BASELINES,
                          get_cv_folds, effective_cv_mode,
                          CV_AUTO_FALLBACK_SPLITS)
from utils.profiler import start as prof_start, stop as prof_stop

# Hyper-parameters
LR          = 0.001
MAX_EPOCHS  = 1000   
MIN_EPOCHS  = 50     
WINDOW_SIZE = 5      
ES_EPSILON  = 0.01
STEP_SIZE   = 30     
GAMMA       = 1.0
BATCH_PROP  = 0.1


# Use the cohort's grouped cell types
N_GENES_TOP = 2000
MIN_CELLS   = 10

# Model — 6-layer FC (singleDeep architecture)
class SingleDeepNet(nn.Module):
    """
    Fixed-width architecture matching GENyO-BioInformatics/singleDeep:
        nGenes → 500 → 250 → 125 → 50 → outNeurons
    ReLU activations, no BatchNorm.
    """
    def __init__(self, n_genes: int, Hs1: int = 500, Hs2: int = 250,
                 Hs3: int = 125, Hs4: int = 50, out_neurons: int = 2):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_genes, Hs1), nn.ReLU(),
            nn.Linear(Hs1, Hs2),     nn.ReLU(),
            nn.Linear(Hs2, Hs3),     nn.ReLU(),
            nn.Linear(Hs3, Hs4),     nn.ReLU(),
            nn.Linear(Hs4, out_neurons),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)   # (B, 2) logits


# Early stopping
def _early_stop(loss_history: list[float], min_epochs: int = None) -> bool:
    SLIDING_SIZE = 2
    total_size = 2 * WINDOW_SIZE - SLIDING_SIZE 
    if min_epochs is None:
        min_epochs = MIN_EPOCHS
    if len(loss_history) <= min_epochs:
        return False
    if len(loss_history) < total_size:
        return False
    recent = loss_history[-total_size:]
    window1 = sum(recent[:WINDOW_SIZE]) / WINDOW_SIZE
    window2 = sum(recent[-WINDOW_SIZE:]) / WINDOW_SIZE
    return (window2 - window1) > ES_EPSILON

# Data loading
def load_cohort(cohort: str, n_genes: int = N_GENES_TOP, h5ad_path=None,
                prior_path=None):
    from utils.celltype_groups import infer_celltype_groups, EXCLUDE_LABEL
    from utils.data_helpers import load_cells_cohort
    X, pat_ids, ct_raw, _, patients, labels = load_cells_cohort(
        cohort, n_genes, h5ad_path, prior_path)

    _, ct_map = infer_celltype_groups(
        pd.DataFrame({"final_celltype": ct_raw, "patient_id": pat_ids}),
        min_patients=3, min_cells_per_patient=10,
        split_groups=["T_cells", "Monocytes", "Dendritic"])
    ct_mapped = np.array([ct_map.get(c, EXCLUDE_LABEL) for c in ct_raw])

    keep      = (ct_mapped != EXCLUDE_LABEL) & (ct_mapped != "Other")
    X         = X[keep]
    pat_ids   = pat_ids[keep]
    ct_mapped = ct_mapped[keep]

    pat_to_lab = {p: l for p, l in zip(patients, labels)}
    cell_labels = np.array([pat_to_lab[p] for p in pat_ids], dtype=np.float32)

    ct_names_all = sorted(set(ct_mapped.tolist()))
    print(f"Cohort {cohort}: {len(patients)} patients, {X.shape[1]} genes, "
          f"{len(ct_names_all)} cell-type clusters (per-CT training, matches "
          f"GENyO-BioInformatics/singleDeep canonical workflow)")
    print(f"  irAE Yes={labels.sum():.0f}  No={len(labels)-labels.sum():.0f}")
    return X, cell_labels, pat_ids, ct_mapped, labels, patients, ct_names_all

# Train model
def train_ct_model(X_tr: np.ndarray, y_tr: np.ndarray,
                   pat_tr: np.ndarray | None = None) -> nn.Module | None:

    if len(X_tr) < MIN_CELLS or len(np.unique(y_tr)) < 2:
        return None

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos

    # Inverse-frequency weights, normalized to sum to 1
    raw_w  = np.array([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)])
    norm_w = raw_w / raw_w.sum()
    cw     = torch.tensor(norm_w, dtype=torch.float32, device=DEVICE)

    rng_np = np.random.default_rng(RANDOM_STATE)
    n_total = len(X_tr)

    if pat_tr is not None:
        unique_pats = np.unique(pat_tr)
        if len(unique_pats) >= 4:
            pat_to_label = {p: int(y_tr[pat_tr == p][0]) for p in unique_pats}
            pat_labels_arr = np.array([pat_to_label[p] for p in unique_pats])
            n_val_pats = max(1, int(np.round(0.20 * len(unique_pats))))
            
            yes_pats = unique_pats[pat_labels_arr == 1]
            no_pats  = unique_pats[pat_labels_arr == 0]
            n_val_yes = max(1, int(np.round(n_val_pats * len(yes_pats) / len(unique_pats))))
            n_val_no  = max(1, n_val_pats - n_val_yes)
            n_val_yes = min(n_val_yes, len(yes_pats) - 1)
            n_val_no  = min(n_val_no,  len(no_pats)  - 1)
            if n_val_yes >= 1 and n_val_no >= 1:
                val_pats = np.concatenate([
                    rng_np.choice(yes_pats, n_val_yes, replace=False),
                    rng_np.choice(no_pats,  n_val_no,  replace=False)])
                val_pats_set = set(val_pats.tolist())
                val_mask = np.array([p in val_pats_set for p in pat_tr])
            else:
                val_mask = np.zeros(n_total, dtype=bool)
        else:
            val_mask = np.zeros(n_total, dtype=bool)
    else:
        n_val = max(2, int(0.15 * n_total))
        val_idx = rng_np.choice(n_total, size=n_val, replace=False)
        val_mask = np.zeros(n_total, dtype=bool); val_mask[val_idx] = True

    use_val = (val_mask.sum() > 0 and
               len(np.unique(y_tr[val_mask])) >= 2 and
               len(np.unique(y_tr[~val_mask])) >= 2)
    if not use_val:
        val_mask = np.zeros(n_total, dtype=bool)

    X_train = X_tr[~val_mask]; y_train = y_tr[~val_mask]
    X_val   = X_tr[val_mask];  y_val   = y_tr[val_mask]
    n_tr    = len(X_train)

    model = SingleDeepNet(X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.SGD(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit  = nn.CrossEntropyLoss(weight=cw)

    X_train_t = torch.from_numpy(X_train).to(DEVICE)
    y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(DEVICE)
    X_val_t   = torch.from_numpy(X_val).to(DEVICE) if use_val else None
    y_val_t   = torch.from_numpy(y_val.astype(np.int64)).to(DEVICE) if use_val else None

    batch_size = max(2, int(np.ceil(BATCH_PROP * n_tr)))
    rng_torch = torch.Generator(device='cpu').manual_seed(RANDOM_STATE)

    loss_history = []   # validation loss per epoch (training loss as fallback)
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(MAX_EPOCHS):
        #Train one epoch
        model.train()
        perm = torch.randperm(n_tr, generator=rng_torch)
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train_t[idx])
            loss = crit(logits, y_train_t[idx])
            if torch.isnan(loss):
                return None
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

        #Compute validation loss
        model.eval()
        with torch.no_grad():
            if X_val_t is not None:
                val_logits = model(X_val_t)
                cur_loss = crit(val_logits, y_val_t).item()
            else:
                tr_logits = model(X_train_t)
                cur_loss = crit(tr_logits, y_train_t).item()
        loss_history.append(cur_loss)

        if cur_loss < best_val_loss:
            best_val_loss = cur_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if _early_stop(loss_history):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_ct(model: nn.Module, X_te: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t   = torch.from_numpy(X_te).to(DEVICE)
        logits = model(x_t)                          
        probs  = torch.softmax(logits, dim=1)[:, 1]  
    out = probs.cpu().numpy()
    out = np.nan_to_num(out, nan=0.5)
    return out


# Train fold: per-CT models + weighted majority vote
def _inner_cv_mcc(X_ct: np.ndarray, y_ct: np.ndarray,
                  ct_pats: np.ndarray, n_inner: int = 3) -> float:
    unique_pats = np.unique(ct_pats)
    pat_labels = np.array([int(y_ct[ct_pats == p][0]) for p in unique_pats])

    if len(unique_pats) < 2 or len(np.unique(pat_labels)) < 2:
        return 0.0

    min_class_count = int(min(np.bincount(pat_labels.astype(int))[:2]))
    n_splits = min(n_inner, len(unique_pats), min_class_count)
    if n_splits < 2:
        return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)

    oof_preds  = np.full(len(unique_pats), 0.5)
    pat_to_idx = {p: i for i, p in enumerate(unique_pats)}

    for tr_pi, va_pi in skf.split(unique_pats, pat_labels):
        tr_pats_set = set(unique_pats[tr_pi])
        va_pats_set = set(unique_pats[va_pi])

        cell_tr = np.array([p in tr_pats_set for p in ct_pats])
        cell_va = np.array([p in va_pats_set for p in ct_pats])

        if cell_tr.sum() < MIN_CELLS or len(np.unique(y_ct[cell_tr])) < 2:
            continue

        m = train_ct_model(X_ct[cell_tr], y_ct[cell_tr], pat_tr=ct_pats[cell_tr])
        if m is None:
            continue

        va_probs = predict_ct(m, X_ct[cell_va])
        va_ct_pats = ct_pats[cell_va]
        # Hard cell-majority vote per validation patient
        for p in va_pats_set:
            p_cells = (va_ct_pats == p)
            if p_cells.any():
                cell_preds = (va_probs[p_cells] >= 0.5).astype(int)
                oof_preds[pat_to_idx[p]] = float(
                    np.bincount(cell_preds, minlength=2).argmax())
    oof_binary = oof_preds.astype(int)
    if len(set(pat_labels)) < 2:
        return 0.0
    try:
        return matthews_corrcoef(pat_labels, oof_binary)
    except Exception:
        return 0.0


INNER_CV_FOLDS = 3

def train_fold(
    X_tr: np.ndarray, y_tr_cell: np.ndarray,
    ct_tr: np.ndarray, pat_tr: np.ndarray,
    X_te: np.ndarray, ct_te: np.ndarray, pat_te: np.ndarray,
    patients_te: list[str], ct_names: list[str],
    pat_labels_tr: np.ndarray,
) -> np.ndarray:

    ct_models:      dict[str, nn.Module] = {}
    ct_weight_map:  dict[str, float]     = {}

    for ct in ct_names:
        ct_mask   = (ct_tr == ct)
        X_ct      = X_tr[ct_mask]
        y_ct      = y_tr_cell[ct_mask]

        if len(X_ct) < MIN_CELLS or len(np.unique(y_ct)) < 2:
            continue

        ct_pats = np.array(pat_tr)[ct_mask]
        mcc_val = _inner_cv_mcc(X_ct, y_ct, ct_pats, n_inner=INNER_CV_FOLDS)
        ct_weight_map[ct] = max(mcc_val, 0.0)

        m_full = train_ct_model(X_ct, y_ct, pat_tr=ct_pats)
        if m_full is not None:
            ct_models[ct] = m_full

    if not ct_models:
        return np.full(len(patients_te), 0.5)

    # Discrete majority voting weighted by round(MCC * 100)
    probs = []
    for p in patients_te:
        p_mask = (pat_te == p)
        vote_counts = {0: 0, 1: 0}
        any_votes = False
        for ct, m in ct_models.items():
            ct_p_mask = p_mask & (ct_te == ct)
            if not ct_p_mask.any():
                continue
            cell_probs = predict_ct(m, X_te[ct_p_mask])
            
            cell_preds = (cell_probs >= 0.5).astype(int)
            ct_pred = int(np.bincount(cell_preds, minlength=2).argmax())
            n_votes = max(0, round(ct_weight_map.get(ct, 0.0) * 100))
            vote_counts[ct_pred] += n_votes
            any_votes = any_votes or (n_votes > 0)

        if not any_votes:
            probs.append(0.5); continue

        total = vote_counts[0] + vote_counts[1]
        probs.append(vote_counts[1] / total if total > 0 else 0.5)

    return np.array(probs)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--n-genes", type=int, default=N_GENES_TOP)
    p.add_argument("--h5ad", default=None,
                   help="Override h5ad path (default: datasets/processed_h5ad/<cohort_id>.h5ad)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--prior", default=None,
                   help="Path to pathway prior NPZ. Subset to prior-active genes "
                        "for fair comparison with irAEGIS (overrides --n-genes).")
    return p.parse_args()

def main():
    args = parse_args()
    prof_start("singledeep", args.cohort)
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    out_dir = Path(args.out_dir) if args.out_dir else (
        RESULTS_BASELINES / args.cohort / "singledeep")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, cell_labs, pat_ids, ct_mapped, labels, patients, ct_names = load_cohort(
        args.cohort, args.n_genes, args.h5ad, prior_path=args.prior)

    sorted_patients, sorted_labels, folds = get_cv_folds(patients, labels)
    patients = sorted_patients
    labels = sorted_labels

    print(f"\n[Patient-level evaluation (MCC-weighted vote)]")
    eff_mode = effective_cv_mode(len(labels))
    is_loocv = eff_mode == "loocv"
    pat_fold_aucs, pat_fold_aps = [], []
    pat_oof_probs = np.full(len(labels), np.nan, dtype=np.float64)
    pat_oof_counts = np.zeros(len(labels), dtype=np.int32)

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        patients_tr = [patients[i] for i in tr_idx]
        patients_te = [patients[i] for i in te_idx]

        tr_cell = np.isin(pat_ids, patients_tr)
        te_cell = np.isin(pat_ids, patients_te)

        probs = train_fold(
            X[tr_cell], cell_labs[tr_cell], ct_mapped[tr_cell], pat_ids[tr_cell],
            X[te_cell], ct_mapped[te_cell], pat_ids[te_cell],
            patients_te, ct_names, labels[tr_idx])

        for vi, pi in enumerate(te_idx):
            if np.isnan(pat_oof_probs[pi]):
                pat_oof_probs[pi] = 0.0
            pat_oof_probs[pi] += probs[vi]
            pat_oof_counts[pi] += 1

        if not is_loocv:
            if len(np.unique(labels[te_idx])) < 2:
                pat_fold_aucs.append(float("nan"))
                pat_fold_aps.append(float("nan"))
            else:
                try:
                    pat_fold_aucs.append(roc_auc_score(labels[te_idx], probs))
                    pat_fold_aps.append(average_precision_score(labels[te_idx], probs))
                except Exception:
                    pat_fold_aucs.append(float("nan"))
                    pat_fold_aps.append(float("nan"))

    valid_oof = pat_oof_counts > 0
    pat_oof_probs[valid_oof] /= pat_oof_counts[valid_oof]
    prevalence = float(labels.sum() / len(labels))

    if is_loocv:
        try:
            pat_mean = float(roc_auc_score(labels[valid_oof], pat_oof_probs[valid_oof]))
        except Exception:
            pat_mean = float("nan")
        pat_std = 0.0
        try:
            pat_mean_ap = float(average_precision_score(labels[valid_oof], pat_oof_probs[valid_oof]))
        except Exception:
            pat_mean_ap = float("nan")
    else:
        valid_pat = [a for a in pat_fold_aucs if not np.isnan(a)]
        valid_pat_aps = [a for a in pat_fold_aps if not np.isnan(a)]
        pat_mean = float(np.mean(valid_pat)) if valid_pat else float("nan")
        pat_std = float(np.std(valid_pat)) if valid_pat else float("nan")
        pat_mean_ap = float(np.mean(valid_pat_aps)) if valid_pat_aps else float("nan")

    pat_oof_preds = (pat_oof_probs[valid_oof] >= 0.5).astype(int)
    try:
        pat_bal_acc = float(balanced_accuracy_score(labels[valid_oof], pat_oof_preds))
    except Exception:
        pat_bal_acc = float("nan")
    try:
        pat_mcc = float(matthews_corrcoef(labels[valid_oof], pat_oof_preds))
    except Exception:
        pat_mcc = float("nan")

    cv_label = {"loocv": "LOOCV", "kfold": f"{CV_AUTO_FALLBACK_SPLITS}-fold CV"}.get(eff_mode, "3×3 CV")
    print(f"  {cv_label} patient AUC: {pat_mean:.4f}  BalAcc: {pat_bal_acc:.4f}  "
          f"MCC: {pat_mcc:.4f}  AUPRC: {pat_mean_ap:.4f} (base={prevalence:.3f})")

    summary = {
        "__patient_level__": {
            "fold_aucs": pat_fold_aucs if not is_loocv else [pat_mean],
            "mean_fold_auc": pat_mean,
            "std_fold_auc": pat_std,
            "fold_aps": pat_fold_aps if not is_loocv else [pat_mean_ap],
            "mean_fold_ap": pat_mean_ap,
            "balanced_accuracy": pat_bal_acc,
            "mcc": pat_mcc,
            "prevalence": prevalence,
            "cv_mode": eff_mode,
            "n_patients_yes": int(labels.sum()),
            "n_patients_no": int(len(labels) - labels.sum()),
        }
    }

    np.save(out_dir / "patient_oof_probs.npy", pat_oof_probs.astype(np.float32))
    np.save(out_dir / "patient_oof_labels.npy", labels.astype(np.int8))

    with open(out_dir / "irae_per_ct_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Patient-level AUC: {pat_mean:.4f} ± {pat_std:.4f}")
    print(f"Saved to {out_dir}")
    prof_stop()


if __name__ == "__main__":
    main()
