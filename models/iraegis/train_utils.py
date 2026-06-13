"""
irAEGIS training utils.
Contains the training loops and CV evaluation for the irAEGIS model.
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import utils.config as _cfg
from utils.config import DEVICE

def _rs():
    """Return the current RANDOM_STATE (may be overridden by --seed)."""
    return _cfg.RANDOM_STATE
from models.iraegis.model_utils import PathwayAE, LinearIraeClassifier

# Hyperparameters
AE_LATENT_DIM    = 32
AE_DROPOUT       = 0.1
AE_BATCH_SIZE    = 2048
AE_N_EPOCHS      = 80
AE_LR            = 1e-3
AE_WEIGHT_DECAY  = 1e-4
AE_MASK_FRAC     = 0.3    # denoising AE: fraction of input genes masked per cell
AE_CT_AUX_WEIGHT = 0.3    # auxiliary cell-type classification loss weight on h
AE_DECORR_WEIGHT = 0.1    # pathway decorrelation loss weight on h


def train_ae(ae:         PathwayAE,
             X:          np.ndarray,
             n_epochs:   int   = AE_N_EPOCHS,
             batch_size: int   = AE_BATCH_SIZE,
             lr:         float = AE_LR,
             wd:         float = AE_WEIGHT_DECAY,
             val_frac:   float = 0.1,
             verbose:    bool  = True,
             ct_ids:     "np.ndarray | None" = None,
             ct_aux_weight: "float | None" = None,
             decorr_weight: "float | None" = None) -> list:
    
    # Train autoencoder
    ct_aux_w = AE_CT_AUX_WEIGHT if ct_aux_weight is None else float(ct_aux_weight)
    decorr_w = AE_DECORR_WEIGHT if decorr_weight is None else float(decorr_weight)
    ae.to(DEVICE)
    rng = np.random.default_rng(_rs())
    n   = len(X)
    idx = rng.permutation(n)
    n_val     = max(1, int(n * val_frac))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_gpu = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    use_ct_aux = ct_ids is not None
    if use_ct_aux:
        n_ct = int(ct_ids.max()) + 1
        ae.attach_ct_head(n_ct)
        ae.ct_head.to(DEVICE)
        ct_gpu = torch.tensor(ct_ids, dtype=torch.long, device=DEVICE)
        params = list(ae.parameters())
    else:
        params = list(ae.parameters())

    opt  = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history   = []
    best_val  = float("inf")
    best_st   = None

    for epoch in range(n_epochs):
        ae.train()
        perm   = rng.permutation(len(train_idx))
        t_idx  = train_idx[perm]
        ep_loss, nb = 0.0, 0

        for i in range(0, len(t_idx), batch_size):
            b_idx = t_idx[i:i + batch_size]
            x_b   = X_gpu[b_idx]
            # Denoising: randomly zero out AE_MASK_FRAC of genes in the INPUT,
            # but reconstruct the full clean input.
            keep_mask = (torch.rand_like(x_b) > AE_MASK_FRAC).float()
            x_in      = x_b * keep_mask / max(1.0 - AE_MASK_FRAC, 1e-6)
            ct_b = ct_gpu[b_idx] if ct_ids is not None else None
            h_b, _, x_r = ae(x_in, ct_ids=ct_b)
            loss  = F.mse_loss(x_r, x_b)
            if use_ct_aux and ct_aux_w > 0:
                ct_logits = ae.ct_head(h_b)
                loss = loss + ct_aux_w * F.cross_entropy(ct_logits, ct_gpu[b_idx])
            if decorr_w > 0:
                h_c = h_b - h_b.mean(0)
                cov = (h_c.T @ h_c) / max(h_b.shape[0] - 1, 1)
                diag = cov.diagonal()
                std = (diag + 1e-8).sqrt()
                corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
                corr = corr - torch.diag(corr.diagonal())
                loss = loss + decorr_w * (corr ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1

        sched.step()

        # Validation
        ae.eval()
        with torch.no_grad():
            vi = rng.choice(val_idx, min(4096, len(val_idx)), replace=False)
            xv = X_gpu[vi]
            ct_v = ct_gpu[vi] if ct_ids is not None else None
            _, _, xrv = ae(xv, ct_ids=ct_v)
            vl = F.mse_loss(xrv, xv).item()

        if vl < best_val:
            best_val = vl
            best_st  = {k: v.clone() for k, v in ae.state_dict().items()}

        rec = {"epoch": epoch + 1, "train_recon": ep_loss / nb, "val_recon": vl}
        history.append(rec)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  AE  epoch {epoch+1:3d}  train={ep_loss/nb:.4f}  "
                  f"val={vl:.4f}  best={best_val:.4f}")

    if best_st:
        ae.load_state_dict(best_st)
    return history


# Pre-compute embeddings h and z and save to disk. 
def precompute_embeddings(ae:         PathwayAE,
                          X:          np.ndarray,
                          obs:        pd.DataFrame,
                          out_dir:    Path,
                          batch_size: int = 16384,    # 4× the AE-train batch no gradient memory needed
                          verbose:    bool = True,
                          ct_ids:     "np.ndarray | None" = None,
                          suffix:     str = "") -> tuple:
   
    # Run all cells through frozen encoder.  Save h, z, and cell metadata.
    ae.eval()
    h_list, z_list = [], []

    ct_gpu = torch.tensor(ct_ids, dtype=torch.long, device=DEVICE) if ct_ids is not None else None

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            ct_b = ct_gpu[i:i + batch_size] if ct_gpu is not None else None
            hb, zb = ae.encode(xb, ct_ids=ct_b)
            h_list.append(hb.cpu().numpy())
            z_list.append(zb.cpu().numpy())

    h = np.vstack(h_list).astype(np.float32)
    z = np.vstack(z_list).astype(np.float32)

    np.save(out_dir / f"h_cells{suffix}.npy", h)
    np.save(out_dir / f"z_cells{suffix}.npy", z)
    if not suffix:
        barcode_col = obs["cell_barcode"] if "cell_barcode" in obs.columns else obs.index
        meta_df = obs[["patient_id", "final_celltype", "irAE_status"]].copy()
        meta_df.insert(0, "cell_barcode", barcode_col.values)
        meta_df.to_csv(out_dir / "cell_meta.csv", index=False)

    if verbose:
        print(f"  Saved h{suffix} {h.shape}, z{suffix} {z.shape} to {Path(out_dir).name}/")

    return h, z

# Per-CT irAE classifier
from utils.config import (CV_N_SPLITS, CV_N_REPEATS, CV_MODE,
                          CV_AUTO_FALLBACK_SPLITS,
                          get_cv_folds)

MIN_CELLS_PER_CT = 50
IRAE_LR_EPOCHS   = 50
IRAE_LR_LR       = 5e-3
IRAE_LR_WD       = 1e-2
IRAE_L1_LAMBDA   = 0.05

def _cv_mode_label(folds_or_n):
    # Returns kfold if the run fell back to k-fold (large cohort) else LOOCV
    n = folds_or_n if isinstance(folds_or_n, int) else len(folds_or_n)
    return "kfold" if n == CV_AUTO_FALLBACK_SPLITS else "loocv"

IRAE_N_SPLITS = CV_N_SPLITS
IRAE_N_REPEATS = CV_N_REPEATS
MCC_THRESHOLD = 0.5

def _build_patient_mean_h(h, pat_ids, ct_cell_idx, patient_list):
    # Compute patient-level mean h for cells in ct_cell_idx
    n_pw = h.shape[1]
    ct_pats = pat_ids[ct_cell_idx]
    pat_h = np.zeros((len(patient_list), n_pw), dtype=np.float32)
    for pi, p in enumerate(patient_list):
        p_mask = ct_pats == p
        if p_mask.any():
            pat_h[pi] = h[ct_cell_idx[p_mask]].mean(axis=0)
    return pat_h


def train_cell_irae_per_ct(
        h: np.ndarray,
        pat_ids: np.ndarray,
        ct_ids: np.ndarray,
        pat_labels: dict,
        ct_groups: list[str],
        n_pathways: int,
        n_epochs: int = IRAE_LR_EPOCHS,
        n_splits: int = IRAE_N_SPLITS,
        n_repeats: int = IRAE_N_REPEATS,
        verbose: bool = True) -> tuple:
   
    # Per-CT irAE classification
    unique_pats = sorted(pat_labels.keys())
    pat_arr = np.array([pat_labels[p] for p in unique_pats])
    n_pats = len(unique_pats)

    _, _, shared_folds = get_cv_folds(unique_pats, pat_arr)

    per_ct_models = {}
    per_ct_summaries = {}
    oof_probs = np.full(len(h), np.nan, dtype=np.float32)

    for ct_idx, ct_name in enumerate(ct_groups):
        ct_mask = ct_ids == ct_idx
        ct_cell_idx = np.where(ct_mask)[0]

        if len(ct_cell_idx) < MIN_CELLS_PER_CT:
            if verbose:
                print(f"\n  [{ct_name}] SKIP — only {len(ct_cell_idx)} cells")
            continue

        ct_pats_yes = set(p for i in ct_cell_idx
                          for p in [pat_ids[i]] if pat_labels[p] == 1)
        ct_pats_no = set(p for i in ct_cell_idx
                         for p in [pat_ids[i]] if pat_labels[p] == 0)
        if len(ct_pats_yes) < 2 or len(ct_pats_no) < 2:
            if verbose:
                print(f"\n  [{ct_name}] SKIP — too few patients "
                      f"(Yes={len(ct_pats_yes)}, No={len(ct_pats_no)})")
            continue

        pat_h = _build_patient_mean_h(h, pat_ids, ct_cell_idx, unique_pats)

        if verbose:
            print(f"\n  [{ct_name}] {len(ct_cell_idx)} cells, "
                  f"{len(ct_pats_yes)} Yes / {len(ct_pats_no)} No patients")

        # Evaluation
        is_loocv = CV_MODE == "loocv"
        oof_pat_probs = np.full(n_pats, np.nan, dtype=np.float32)
        oof_pat_counts = np.zeros(n_pats, dtype=np.int32)
        fold_aucs, fold_aps = [], []

        for fold_i, (tr_idx, va_idx) in enumerate(shared_folds):
            pat_h_tr = pat_h[tr_idx]
            y_tr = pat_arr[tr_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(pat_h_tr)
            X_va = scaler.transform(pat_h[va_idx])

            lr = LogisticRegression(
                max_iter=2000, solver="lbfgs", class_weight="balanced",
                random_state=_rs(), C=0.1)
            lr.fit(X_tr, y_tr)

            try:
                va_probs = lr.predict_proba(X_va)[:, 1]
            except Exception:
                va_probs = np.full(len(va_idx), 0.5)

            for vi, pi in enumerate(va_idx):
                if np.isnan(oof_pat_probs[pi]):
                    oof_pat_probs[pi] = 0.0
                oof_pat_probs[pi] += va_probs[vi]
                oof_pat_counts[pi] += 1

            if not is_loocv:
                va_labels = pat_arr[va_idx]
                if len(set(va_labels.astype(int).tolist())) < 2:
                    fold_aucs.append(float("nan"))
                    fold_aps.append(float("nan"))
                else:
                    try:
                        fold_aucs.append(float(roc_auc_score(va_labels, va_probs)))
                        fold_aps.append(float(average_precision_score(va_labels, va_probs)))
                    except Exception:
                        fold_aucs.append(float("nan"))
                        fold_aps.append(float("nan"))

        valid_oof = oof_pat_counts > 0
        oof_pat_probs[valid_oof] /= oof_pat_counts[valid_oof]

        prevalence = float(pat_arr.sum() / len(pat_arr))

        if is_loocv:
            try:
                mean_fold_auc = float(roc_auc_score(pat_arr[valid_oof], oof_pat_probs[valid_oof]))
            except Exception:
                mean_fold_auc = float("nan")
            std_fold_auc = 0.0
            try:
                mean_fold_ap = float(average_precision_score(pat_arr[valid_oof], oof_pat_probs[valid_oof]))
            except Exception:
                mean_fold_ap = float("nan")
            oof_preds = (oof_pat_probs[valid_oof] >= MCC_THRESHOLD).astype(int)
            try:
                bal_acc = float(balanced_accuracy_score(pat_arr[valid_oof], oof_preds))
            except Exception:
                bal_acc = float("nan")
            try:
                mcc = float(matthews_corrcoef(pat_arr[valid_oof], oof_preds))
            except Exception:
                mcc = float("nan")
        else:
            valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
            valid_aps = [a for a in fold_aps if not np.isnan(a)]
            mean_fold_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
            std_fold_auc = float(np.std(valid_aucs)) if valid_aucs else float("nan")
            mean_fold_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")
            oof_preds = (oof_pat_probs[valid_oof] >= MCC_THRESHOLD).astype(int)
            try:
                bal_acc = float(balanced_accuracy_score(pat_arr[valid_oof], oof_preds))
            except Exception:
                bal_acc = float("nan")
            try:
                mcc = float(matthews_corrcoef(pat_arr[valid_oof], oof_preds))
            except Exception:
                mcc = float("nan")

        if verbose:
            cv_label = "LOOCV" if is_loocv else f"{n_splits}×{n_repeats} CV"
            print(f"    {cv_label} AUC: {mean_fold_auc:.4f}  "
                  f"BalAcc: {bal_acc:.4f}  MCC: {mcc:.4f}  "
                  f"AUPRC: {mean_fold_ap:.4f} (base={prevalence:.3f})")

        per_ct_summaries[ct_name] = {
            "fold_aucs": fold_aucs if not is_loocv else [mean_fold_auc],
            "mean_fold_auc": mean_fold_auc,
            "std_fold_auc": std_fold_auc,
            "fold_aps": fold_aps if not is_loocv else [mean_fold_ap],
            "mean_fold_ap": mean_fold_ap,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
            "prevalence": prevalence,
            "cv_mode": _cv_mode_label(shared_folds),
            "n_folds": len(shared_folds),
            "n_cells": len(ct_cell_idx),
            "n_patients_yes": len(ct_pats_yes),
            "n_patients_no": len(ct_pats_no),
        }

        # Final model on all patients for explainability
        pat_h_gpu = torch.tensor(pat_h, dtype=torch.float32, device=DEVICE)
        pat_y_gpu = torch.tensor(pat_arr, dtype=torch.float32, device=DEVICE)
        n_yes = int(pat_arr.sum())
        n_no = n_pats - n_yes
        pw = torch.tensor([n_no / max(n_yes, 1)], dtype=torch.float32, device=DEVICE)

        clf_final = LinearIraeClassifier(n_pathways).to(DEVICE)
        opt = torch.optim.Adam(clf_final.parameters(), lr=IRAE_LR_LR,
                               weight_decay=IRAE_LR_WD)
        rng = np.random.default_rng(_rs() + ct_idx * 100)
        for epoch in range(n_epochs):
            clf_final.train()
            perm = rng.permutation(n_pats)
            logits = clf_final(pat_h_gpu[perm]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, pat_y_gpu[perm], pos_weight=pw)
            opt.zero_grad(); loss.backward()
            opt.step()

        per_ct_models[ct_name] = clf_final

        for pi, p in enumerate(unique_pats):
            if not valid_oof[pi]:
                continue
            cell_mask = ct_mask & np.array([pid == p for pid in pat_ids])
            oof_probs[cell_mask] = oof_pat_probs[pi]

    return per_ct_models, per_ct_summaries, oof_probs


# Patient-level classifier
def _compute_metrics(pat_arr, oof_vals, valid_mask, threshold=0.5):
    """Compute AUC, AUPRC, BalAcc, MCC from OOF predictions."""
    y = pat_arr[valid_mask]
    v = oof_vals[valid_mask]
    try:
        auc = float(roc_auc_score(y, v))
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y, v))
    except Exception:
        ap = float("nan")
    preds = (v >= threshold).astype(int)
    try:
        ba = float(balanced_accuracy_score(y, preds))
    except Exception:
        ba = float("nan")
    try:
        mc = float(matthews_corrcoef(y, preds))
    except Exception:
        mc = float("nan")
    return auc, ap, ba, mc


def train_h_concat_gated_concat_en(
        h: np.ndarray,
        pat_ids: np.ndarray,
        ct_ids: np.ndarray,
        pat_labels: dict,
        ct_groups: list[str],
        active_cts: list[str] | None = None,
        auc_gate: float = 0.50,
        C_inner: float = 0.1,
        C_outer: float = 1.0,
        use_logit: bool = True,
        verbose: bool = True) -> dict:
    
    # Patient-level irAEGIS classifier
    unique_pats = sorted(pat_labels.keys())
    pat_arr = np.array([pat_labels[p] for p in unique_pats], dtype=np.int64)
    n_pats = len(unique_pats)
    if active_cts is None:
        active_cts = sorted(ct_groups)
    K = len(active_cts)
    P = h.shape[1]

    # Aggregate to (patient, CT) via top 25% by norm mean
    name_to_idx = {n: i for i, n in enumerate(ct_groups)}
    pat_idx_arr = np.array([list(unique_pats).index(p) for p in pat_ids])
    pat_h_full = np.zeros((n_pats, len(ct_groups), P), dtype=np.float32)
    for j in range(len(ct_groups)):
        for i, p in enumerate(unique_pats):
            mask = (pat_ids == p) & (ct_ids == j)
            if not mask.any():
                continue
            cells = h[mask]
            if cells.shape[0] >= 4:
                norms = np.linalg.norm(cells, axis=1)
                cutoff = np.percentile(norms, 75)
                top = cells[norms >= cutoff]
                pat_h_full[i, j] = top.mean(axis=0) if len(top) else cells.mean(axis=0)
            else:
                pat_h_full[i, j] = cells.mean(axis=0)

    active_ct_idx = np.array([name_to_idx[c] for c in active_cts])
    pat_h = pat_h_full[:, active_ct_idx, :]

    if verbose:
        print(f"  [gated CT stacking] {n_pats} patients × {K} CTs × {P} pw, "
              f"gate={auc_gate}, C_inner={C_inner}, C_outer={C_outer}, top-25 agg")

    # Inner-LOOCV AUC per CT on a training subset
    def inner_aucs(tr_idx):
        aucs = np.zeros(K)
        tr_labels = pat_arr[tr_idx]
        if (tr_labels == 1).sum() < 2 or (tr_labels == 0).sum() < 2:
            return aucs
        for j in range(K):
            Xj = pat_h[tr_idx, j]
            preds = np.zeros(len(tr_idx))
            for itr, iva in LeaveOneOut().split(Xj):
                sc = StandardScaler().fit(Xj[itr])
                try:
                    lr = LogisticRegression(solver="liblinear", max_iter=2000,
                                              C=C_inner, class_weight="balanced",
                                              random_state=_rs())
                    lr.fit(sc.transform(Xj[itr]), tr_labels[itr])
                    preds[iva[0]] = lr.predict_proba(sc.transform(Xj[iva]))[0, 1]
                except Exception:
                    preds[iva[0]] = 0.5
            try:
                aucs[j] = roc_auc_score(tr_labels, preds)
            except Exception:
                aucs[j] = 0.5
        return aucs

    # Outer LOOCV across patients
    prevalence = float(pat_arr.mean())
    oof_probs = np.full(n_pats, np.nan, dtype=np.float64)
    fold_selected_cts = []

    for P_idx in range(n_pats):
        tr_idx = np.array([i for i in range(n_pats) if i != P_idx])
        per_ct = inner_aucs(tr_idx)
        selected = [j for j, a in enumerate(per_ct) if a >= auc_gate]
        if not selected:
            selected = [int(np.argmax(per_ct))]
        fold_selected_cts.append([active_cts[j] for j in selected])

        feats = np.full((n_pats, len(selected)), 0.5, dtype=np.float64)
        for col, j in enumerate(selected):
            X_tr_full = pat_h[tr_idx, j]; y_tr_full = pat_arr[tr_idx]
            if (y_tr_full == 1).sum() < 2 or (y_tr_full == 0).sum() < 2:
                continue
            sc = StandardScaler().fit(X_tr_full)
            try:
                lr = LogisticRegression(max_iter=2000, solver="lbfgs",
                                          class_weight="balanced", random_state=_rs(),
                                          C=C_inner)
                lr.fit(sc.transform(X_tr_full), y_tr_full)
                feats[P_idx, col] = lr.predict_proba(sc.transform(pat_h[[P_idx], j]))[0, 1]
            except Exception:
                feats[P_idx, col] = 0.5
            # Stacked inner-LOOCV predictions for training patients
            for i in tr_idx:
                tr_minus_i = np.array([m for m in tr_idx if m != i])
                X_tr2 = pat_h[tr_minus_i, j]; y_tr2 = pat_arr[tr_minus_i]
                if (y_tr2 == 1).sum() < 2 or (y_tr2 == 0).sum() < 2:
                    continue
                try:
                    sc2 = StandardScaler().fit(X_tr2)
                    lr2 = LogisticRegression(max_iter=2000, solver="lbfgs",
                                               class_weight="balanced",
                                               random_state=_rs(), C=C_inner)
                    lr2.fit(sc2.transform(X_tr2), y_tr2)
                    feats[i, col] = lr2.predict_proba(sc2.transform(pat_h[[i], j]))[0, 1]
                except Exception:
                    feats[i, col] = 0.5

        f = feats.copy()
        if use_logit:
            f = np.log(np.clip(f, 1e-6, 1 - 1e-6) / np.clip(1 - f, 1e-6, 1 - 1e-6))
        outer = LogisticRegression(C=C_outer, max_iter=2000,
                                     class_weight="balanced", random_state=_rs())
        outer.fit(f[tr_idx], pat_arr[tr_idx])
        oof_probs[P_idx] = outer.predict_proba(f[[P_idx]])[0, 1]

    valid = ~np.isnan(oof_probs)
    mean_auc, mean_ap, bal_acc, mcc = _compute_metrics(pat_arr, oof_probs, valid)
    avg_selected = (sum(len(s) for s in fold_selected_cts)
                    / max(len(fold_selected_cts), 1))

    if verbose:
        print(f"    LOOCV: AUC={mean_auc:.4f}  BalAcc={bal_acc:.4f}  "
              f"MCC={mcc:.4f}  AUPRC={mean_ap:.4f} (base={prevalence:.3f})")
        print(f"    Avg # CTs gated/fold: {avg_selected:.2f} / {K}")

    return {
        "active_cts": active_cts,
        "patient_order": unique_pats,
        "oof_probs": oof_probs.tolist(),
        "cv_summary": {
            "method": "gated CT stacking + top-25",
            "mean_auc": mean_auc, "std_auc": 0.0,
            "mean_fold_ap": mean_ap, "balanced_accuracy": bal_acc,
            "mcc": mcc, "prevalence": prevalence, "cv_mode": "loocv",
            "n_patients": n_pats, "n_active_cts": K,
            "auc_gate": auc_gate, "avg_selected_cts": avg_selected,
            "C_inner": C_inner, "C_outer": C_outer,
        },
    }

# Patient W_eff extraction 
def extract_patient_w_eff(h, pat_ids, ct_ids, pat_labels_dict, ct_groups,
                           gate=0.50, C_inner=0.1, C_outer=1.0):
    
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "scripts"))
    from run_iraegis_inference import aggregate_top25_mean, per_ct_inner_loocv_aucs

    unique_pats = sorted(pat_labels_dict.keys())
    pat_labels  = np.array([pat_labels_dict[p] for p in unique_pats])
    n_pat       = len(unique_pats)
    n_ct        = len(ct_groups)

    # Aggregate cells
    pat_h = aggregate_top25_mean(h, pat_ids, ct_ids, unique_pats, n_ct)

    # Inner-LOOCV AUC per CT
    ct_aucs = per_ct_inner_loocv_aucs(pat_h, pat_labels, np.arange(n_pat), C=C_inner)
    selected = [j for j, a in enumerate(ct_aucs) if a >= gate]
    if not selected:
        selected = [int(np.argmax(ct_aucs))]

    # Per-CT classifier fit on ALL patients
    per_ct_betas, per_ct_b0, per_ct_mu, per_ct_sd = [], [], [], []
    for j in selected:
        Xj = pat_h[:, j]
        sc = StandardScaler().fit(Xj)
        lr = LogisticRegression(C=C_inner, max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", random_state=_rs())
        lr.fit(sc.transform(Xj), pat_labels)
        per_ct_betas.append(lr.coef_[0].astype(np.float64))
        per_ct_b0.append(float(lr.intercept_[0]))
        per_ct_mu.append(sc.mean_.astype(np.float64))
        per_ct_sd.append(sc.scale_.astype(np.float64))

    feats = np.zeros((n_pat, len(selected)))
    for col, j in enumerate(selected):
        for i in range(n_pat):
            tr = np.array([m for m in range(n_pat) if m != i])
            try:
                sc2 = StandardScaler().fit(pat_h[tr, j])
                lr2 = LogisticRegression(C=C_inner, max_iter=2000, solver="lbfgs",
                                          class_weight="balanced", random_state=_rs())
                lr2.fit(sc2.transform(pat_h[tr, j]), pat_labels[tr])
                feats[i, col] = lr2.predict_proba(sc2.transform(pat_h[[i], j]))[0, 1]
            except Exception:
                feats[i, col] = 0.5

    f_logit = np.log(np.clip(feats, 1e-6, 1 - 1e-6) /
                     np.clip(1 - feats, 1e-6, 1 - 1e-6))
    outer = LogisticRegression(C=C_outer, max_iter=2000, class_weight="balanced",
                                random_state=_rs())
    outer.fit(f_logit, pat_labels)
    w_meta = outer.coef_[0].astype(np.float64)
    b_meta = float(outer.intercept_[0])

    payload = {}
    for col, j in enumerate(selected):
        beta = per_ct_betas[col]
        sd   = per_ct_sd[col]
        mu   = per_ct_mu[col]
        W_eff_k = w_meta[col] * beta / np.maximum(sd, 1e-12)
        payload[f"W_eff_{col}"]    = W_eff_k.astype(np.float64)
        payload[f"mu_train_{col}"] = mu.astype(np.float64)

    payload["selected_cts"]  = np.array([ct_groups[j] for j in selected], dtype=object)
    payload["ct_aucs"]       = ct_aucs.astype(np.float64)
    payload["ct_groups_all"] = np.array(ct_groups, dtype=object)
    payload["auc_gate"]      = np.float64(gate)
    payload["w_LR"]          = w_meta.astype(np.float32)
    payload["b_LR"]          = np.float64(b_meta)
    return payload
