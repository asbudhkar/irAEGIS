#!/usr/bin/env python3
# Hierarchical MIL baseline: Reference: https://github.com/minhchaudo/hier-mil

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Default per-cohort h5ad: datasets/processed_h5ad/<cohort_id>.h5ad
from utils.config import (DEVICE, RANDOM_STATE, RESULTS_BASELINES,
                          get_cv_folds, CV_MODE, effective_cv_mode,
                          CV_AUTO_FALLBACK_SPLITS)
from utils.profiler import start as prof_start, stop as prof_stop

# Default hyper-parameters
HIDDEN_DIM   = 64
DROPOUT      = 0.3
LR           = 2e-3
WEIGHT_DECAY = 1e-3
N_EPOCHS     = 200
PATIENCE     = 30
N_GENES_TOP  = 2000

# Optuna hyperparameter tuning
USE_OPTUNA      = True
OPTUNA_N_TRIALS = 10
OPTUNA_INNER_K  = 3

def _size_aware_optuna_budget(n_train: int) -> tuple[int, int]:
    """Return (n_trials, inner_k) appropriate for cohort size."""
    if n_train < 30:
        return 5, 3    
    if n_train < 50:
        return 7, 3      
    return OPTUNA_N_TRIALS, OPTUNA_INNER_K

# Model
try:
    from torch_geometric.utils import softmax as pyg_softmax
    from torch_geometric.nn import global_add_pool, global_mean_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

class HierMIL(nn.Module):
    
    def __init__(self, n_in: int, n_hid: int = 32, n_hid2: int = 0,
                 n_layers_lin: int = 1, n_layers_lin2: int = 0,
                 dropout: float = 0.0, attn1: bool = True, attn2: bool = True,
                 use_softmax: bool = True, n_out: int = 1):
        super().__init__()
        self.lin = nn.Sequential(
            *self._lin_layers(n_layers_lin, n_in, n_hid, n_hid, dropout))
        curr_in = n_in if len(self.lin) == 0 else n_hid
        self.w_c = nn.Sequential(nn.Linear(curr_in, 1), nn.Dropout(dropout))
        self.n_in1 = curr_in
        self.lin2 = nn.Sequential(
            *self._lin_layers(n_layers_lin2, curr_in, n_hid2, n_hid2, dropout))
        curr_in = curr_in if len(self.lin2) == 0 else n_hid2
        self.w_ct = nn.Sequential(nn.Linear(curr_in, 1), nn.Dropout(dropout))
        self.lin_out = nn.Linear(curr_in, n_out)
        self.attn1 = attn1
        self.attn2 = attn2
        self.use_softmax = use_softmax

    @staticmethod
    def _lin_layers(n_layers, n_in, n_hid, n_out, dropout):
        layers = []
        for i in range(n_layers):
            ci = n_in if i == 0 else n_hid
            co = n_out if i == n_layers - 1 else n_hid
            layers += [nn.Linear(ci, co), nn.ReLU(), nn.Dropout(dropout)]
        return layers

    def forward(self, X, batch, ct_size, n_ct):
        X = self.lin(X)
        if self.attn1:
            if self.use_softmax:
                w_c = pyg_softmax(self.w_c(X).squeeze(), batch)
            else:
                w_c = torch.sigmoid(self.w_c(X).squeeze())
            if self.attn2:
                X = global_add_pool(X * w_c.unsqueeze(-1), batch,
                                    size=ct_size).reshape(-1, n_ct, self.n_in1)
            else:
                X = global_add_pool(X * w_c.unsqueeze(-1), batch)
        else:
            if self.attn2:
                X = global_mean_pool(X, batch, size=ct_size).reshape(-1, n_ct, self.n_in1)
            else:
                X = global_mean_pool(X, batch)
        X = self.lin2(X)
        if self.attn2:
            if self.use_softmax:
                w_ct = torch.softmax(self.w_ct(X), dim=1)
            else:
                w_ct = torch.sigmoid(self.w_ct(X))
            X = (X * w_ct).sum(dim=1)
        X = self.lin_out(X)
        return X

# Data loading
def load_cohort(cohort: str, n_genes: int = N_GENES_TOP, h5ad_path=None,
                prior_path=None):
    import pandas as pd
    from utils.celltype_groups import EXCLUDE_LABEL
    from utils.data_helpers import load_cells_cohort
    X, pat_ids, ct_raw, _, patients, labels = load_cells_cohort(
        cohort, n_genes, h5ad_path, prior_path)

    from utils.celltype_groups import infer_celltype_groups
    _, ct_map = infer_celltype_groups(
        pd.DataFrame({"final_celltype": ct_raw, "patient_id": pat_ids}),
        min_patients=3, min_cells_per_patient=10,
        split_groups=["T_cells", "Monocytes", "Dendritic"])
    ct_mapped = np.array([ct_map.get(c, EXCLUDE_LABEL) for c in ct_raw])

    keep      = (ct_mapped != EXCLUDE_LABEL) & (ct_mapped != "Other")
    X         = X[keep]
    pat_ids   = pat_ids[keep]
    ct_mapped = ct_mapped[keep]

    ct_names_all = sorted(set(ct_mapped.tolist()))
    rng = np.random.default_rng(RANDOM_STATE)

    bags = {}
    pat_idx_map = {p: np.where(pat_ids == p)[0] for p in patients}
    for p in patients:
        p_idx = pat_idx_map[p]
        bags[p] = {}
        for ct in ct_names_all:
            m = ct_mapped[p_idx] == ct
            if not m.any():
                continue
            bags[p][ct] = X[p_idx[m]]

    valid_pats = [p for p in patients if bags[p]]
    _pat_to_idx = {p: i for i, p in enumerate(patients)}
    valid_labs = labels[[_pat_to_idx[p] for p in valid_pats]]

    print(f"Cohort {cohort}: {len(valid_pats)} patients, {X.shape[1]} genes, {len(ct_names_all)} CTs")
    print(f"  irAE Yes={valid_labs.sum():.0f}  No={len(valid_labs)-valid_labs.sum():.0f}")
    return bags, valid_labs, valid_pats, X.shape[1]


# Graph-batch construction

def _build_graph_batch(bags: dict, patients: list[str], all_ct: list[str]):
    n_ct = len(all_ct)
    ct_dict = {ct: i for i, ct in enumerate(all_ct)}
    Xs, batches = [], []
    for pi, p in enumerate(patients):
        for ct, arr in bags[p].items():
            if ct not in ct_dict or arr.shape[0] == 0:
                continue
            Xs.append(arr)
            batches.append(np.full(arr.shape[0], pi * n_ct + ct_dict[ct],
                                   dtype=np.int64))
    if not Xs:
        return None, None, 0, n_ct
    X_all = np.concatenate(Xs, axis=0).astype(np.float32)
    batch = np.concatenate(batches, axis=0)
    return X_all, batch, len(patients) * n_ct, n_ct


# Train one fold
def train_fold(bags_tr: dict, y_tr: np.ndarray,
               bags_te: dict, patients_te: list[str],
               n_genes: int, hp: dict | None = None,
               all_ct: list[str] | None = None) -> np.ndarray:

    if not _HAS_PYG:
        raise ImportError("torch_geometric is required for HierMIL graph batching. "
                          "Install with: pip install torch_geometric")
    if hp is None:
        hp = dict(hidden=HIDDEN_DIM, dropout=DROPOUT, lr=LR,
                  weight_decay=WEIGHT_DECAY, n_epochs=N_EPOCHS,
                  patience=PATIENCE, n_layers_lin=1)

    if all_ct is None:
        all_ct = sorted(set(ct for d in (bags_tr, bags_te) for p in d for ct in d[p]))

    patients_tr = list(bags_tr.keys())

    
    model = HierMIL(n_in=n_genes, n_hid=hp["hidden"], dropout=hp["dropout"],
                    n_layers_lin=hp.get("n_layers_lin", 1)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                             weight_decay=hp["weight_decay"])
    crit  = nn.BCEWithLogitsLoss()

    X_tr_np, batch_tr_np, ct_size_tr, n_ct = _build_graph_batch(
        bags_tr, patients_tr, all_ct)
    if X_tr_np is None:
        return np.full(len(patients_te), 0.5)
    X_tr   = torch.from_numpy(X_tr_np).to(DEVICE)
    batch_tr = torch.from_numpy(batch_tr_np).to(DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)

    # Train
    for epoch in range(hp["n_epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(X_tr, batch_tr, ct_size_tr, n_ct).squeeze(-1)
        loss = crit(logits, y_tr_t)
        loss.backward()
        opt.step()

    model.eval()
    X_te_np, batch_te_np, ct_size_te, _ = _build_graph_batch(
        bags_te, patients_te, all_ct)
    if X_te_np is None:
        return np.full(len(patients_te), 0.5)
    X_te = torch.from_numpy(X_te_np).to(DEVICE)
    batch_te = torch.from_numpy(batch_te_np).to(DEVICE)
    with torch.no_grad():
        logits_te = model(X_te, batch_te, ct_size_te, n_ct).squeeze(-1)
        probs = torch.sigmoid(logits_te).cpu().numpy()
    
    if probs.shape[0] != len(patients_te):
        full = np.full(len(patients_te), 0.5)
        full[:probs.shape[0]] = probs
        probs = full
    return probs

# Optuna hyperparameter tuning
def _tune_hyperparams(bags: dict, y: np.ndarray, patients: list[str],
                      n_genes: int, all_ct: list[str],
                      n_trials: int = OPTUNA_N_TRIALS,
                      inner_k: int = OPTUNA_INNER_K,
                      seed_offset: int = 0) -> dict | None:
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.model_selection import StratifiedKFold
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    skf = StratifiedKFold(n_splits=inner_k, shuffle=True,
                          random_state=RANDOM_STATE + seed_offset)
    inner_folds = list(skf.split(np.arange(len(y)), y))

    def objective(trial):
        hp = {
            "n_epochs":     trial.suggest_categorical("n_epochs", [100, 500, 1000]),
            "dropout":      trial.suggest_categorical("dropout", [0.0, 0.3, 0.5, 0.7]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2]),
            "n_layers_lin": trial.suggest_categorical("n_layers_lin", [1, 2]),
            "hidden":       trial.suggest_categorical("hidden", [32, 64, 128]),
            "lr":           trial.suggest_categorical("lr", [1e-3, 5e-3]),
            "patience":     PATIENCE,
        }
        oof = np.full(len(y), np.nan)
        for tr, va in inner_folds:
            bags_tr = {patients[i]: bags[patients[i]] for i in tr}
            bags_va = {patients[i]: bags[patients[i]] for i in va}
            patients_va = [patients[i] for i in va]
            prob = train_fold(bags_tr, y[tr], bags_va, patients_va,
                              n_genes, hp=hp, all_ct=all_ct)
            for vi, pi in enumerate(va):
                oof[pi] = prob[vi]
        valid = ~np.isnan(oof)
        return float(roc_auc_score(y[valid], oof[valid]))

    sampler = TPESampler(seed=RANDOM_STATE + seed_offset)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["patience"] = PATIENCE
    return best, study.best_value


# CV
def run_cv(bags: dict, y: np.ndarray, patients: list[str], n_genes: int,
           folds=None) -> dict:

    if folds is None:
        patients, y, folds = get_cv_folds(patients, y)
    n = len(y)
    eff_mode = effective_cv_mode(len(y))
    is_loocv = eff_mode == "loocv"

    all_ct = sorted({ct for p in patients for ct in bags[p]})

    fold_aucs, fold_aps = [], []
    prevalence = float(y.sum() / len(y))
    oof_probs = np.full(n, np.nan, dtype=np.float64)
    oof_counts = np.zeros(n, dtype=np.int32)

    for fold_i, (tr, te) in enumerate(folds):
        bags_tr     = {patients[i]: bags[patients[i]] for i in tr}
        bags_te     = {patients[i]: bags[patients[i]] for i in te}
        patients_te = [patients[i] for i in te]
        patients_tr = [patients[i] for i in tr]
        y_tr = y[tr]

        if not USE_OPTUNA:
            raise RuntimeError(
                "USE_OPTUNA=False is incompatible with faithful hier-mil; "
                "the original repo always tunes per outer fold.")
        if len(set(y_tr)) < 2:
            raise RuntimeError(
                f"Fold {fold_i} train split has only {len(set(y_tr))} class "
                f"— Optuna cannot tune on a single-class set.")
        n_trials_eff, inner_k_eff = _size_aware_optuna_budget(len(patients_tr))
        print(f"  [HierMIL fold {fold_i+1}/{len(folds)}] "
              f"Optuna ({n_trials_eff} trials × {inner_k_eff}-fold inner, "
              f"N_train={len(patients_tr)}) ...")
        best_hp, best_inner_auc = _tune_hyperparams(
            bags_tr, y_tr, patients_tr, n_genes, all_ct,
            n_trials=n_trials_eff, inner_k=inner_k_eff,
            seed_offset=fold_i)
        print(f"    best inner-AUC={best_inner_auc:.4f}  hp={best_hp}")

        prob = train_fold(bags_tr, y_tr, bags_te, patients_te, n_genes,
                          hp=best_hp, all_ct=all_ct)

        for vi, pi in enumerate(te):
            if np.isnan(oof_probs[pi]):
                oof_probs[pi] = 0.0
            oof_probs[pi] += prob[vi]
            oof_counts[pi] += 1

        if not is_loocv:
            if len(np.unique(y[te])) < 2:
                fold_aucs.append(float("nan"))
                fold_aps.append(float("nan"))
            else:
                try:
                    fold_aucs.append(roc_auc_score(y[te], prob))
                    fold_aps.append(average_precision_score(y[te], prob))
                except Exception:
                    fold_aucs.append(float("nan"))
                    fold_aps.append(float("nan"))

    valid_oof = oof_counts > 0
    oof_probs[valid_oof] /= oof_counts[valid_oof]

    if is_loocv:
        try:
            mean_auc = float(roc_auc_score(y[valid_oof], oof_probs[valid_oof]))
        except Exception:
            mean_auc = float("nan")
        std_auc = 0.0
        try:
            mean_ap = float(average_precision_score(y[valid_oof], oof_probs[valid_oof]))
        except Exception:
            mean_ap = float("nan")
    else:
        valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
        valid_aps = [a for a in fold_aps if not np.isnan(a)]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        std_auc = float(np.std(valid_aucs)) if valid_aucs else float("nan")
        mean_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")

    oof_preds = (oof_probs[valid_oof] >= 0.5).astype(int)
    try:
        bal_acc = float(balanced_accuracy_score(y[valid_oof], oof_preds))
    except Exception:
        bal_acc = float("nan")
    try:
        mcc = float(matthews_corrcoef(y[valid_oof], oof_preds))
    except Exception:
        mcc = float("nan")

    cv_label = {"loocv": "LOOCV", "kfold": f"{CV_AUTO_FALLBACK_SPLITS}-fold CV"}.get(eff_mode, "3×3 CV")
    print(f"  {cv_label} AUC={mean_auc:.4f}  BalAcc={bal_acc:.4f}  "
          f"MCC={mcc:.4f}  AUPRC={mean_ap:.4f} (base={prevalence:.3f})")

    return {
        "mean_fold_auc": mean_auc, "std_auc": std_auc,
        "fold_aucs": fold_aucs if not is_loocv else [mean_auc],
        "mean_fold_ap": mean_ap,
        "fold_aps": fold_aps if not is_loocv else [mean_ap],
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "prevalence": prevalence,
        "cv_mode": eff_mode,
        "n_patients": n, "n_pos": int(y.sum()),
        "oof_probs": oof_probs, "labels": y,
    }

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
    prof_start("hiermil", args.cohort)
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    out_dir = Path(args.out_dir) if args.out_dir else (
        RESULTS_BASELINES / args.cohort / "hierarchical_mil")
    out_dir.mkdir(parents=True, exist_ok=True)

    bags, labels, patients, n_genes = load_cohort(
        args.cohort, args.n_genes, args.h5ad, prior_path=args.prior)

    sorted_patients, sorted_labels, folds = get_cv_folds(patients, labels)
    patients = sorted_patients
    labels = sorted_labels

    print(f"\n[Patient-level evaluation (all CTs)]")
    patient_results = run_cv(bags, labels, patients, n_genes, folds=folds)

    summary = {
        "__patient_level__": {
            "fold_aucs": patient_results["fold_aucs"],
            "mean_fold_auc": patient_results["mean_fold_auc"],
            "std_fold_auc": patient_results["std_auc"],
            "fold_aps": patient_results["fold_aps"],
            "mean_fold_ap": patient_results["mean_fold_ap"],
            "balanced_accuracy": patient_results.get("balanced_accuracy"),
            "mcc": patient_results.get("mcc"),
            "prevalence": patient_results["prevalence"],
            "cv_mode": patient_results.get("cv_mode", CV_MODE),
            "n_patients": patient_results["n_patients"],
            "n_patients_yes": patient_results["n_pos"],
            "n_patients_no": patient_results["n_patients"] - patient_results["n_pos"],
        }
    }

    np.save(out_dir / "patient_oof_probs.npy",
            patient_results["oof_probs"].astype(np.float32))
    np.save(out_dir / "patient_oof_labels.npy",
            patient_results["labels"].astype(np.int8))

    with open(out_dir / "irae_per_ct_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Patient-level AUC: {patient_results['mean_fold_auc']:.4f} "
          f"± {patient_results['std_auc']:.4f}")
    print(f"Saved to {out_dir}")
    prof_stop()


if __name__ == "__main__":
    main()
