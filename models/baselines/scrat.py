#!/usr/bin/env python3

#ScRAT baseline: Reference https://github.com/yuzhenmao/ScRAT
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.config import (DEVICE, RANDOM_STATE, RESULTS_BASELINES, get_cv_folds,
                          effective_cv_mode, CV_AUTO_FALLBACK_SPLITS)
from utils.profiler import start as prof_start, stop as prof_stop

# Hyper-parameters
EMB_DIM     = 128
N_HEADS     = 8
N_LAYERS    = 1
DROPOUT     = 0.3
LR          = 0.01
WEIGHT_DECAY = 1e-4
N_EPOCHS    = 100
PATIENCE    = 2
N_GENES_TOP = 2000
SAMPLE_CELLS         = 500
TRAIN_NUM_SAMPLE = 20 
TEST_NUM_SAMPLE  = 50 
BATCH_SIZE       = 256  
MIN_SIZE         = 10000  

# PCA
USE_PCA          = True
PCA_COMPONENTS   = 50

AUGMENT_NUM    = 300    
MIXUP_ALPHA    = 0.5    
INTER_ONLY     = True   


class ScRAT(nn.Module):
    def __init__(self, n_genes: int, emb_dim: int = 128, n_heads: int = 8,
                 n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_net = nn.Sequential(
            nn.Linear(n_genes, emb_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=2 * emb_dim,
            dropout=dropout, batch_first=True,
            norm_first=False,
        )
       
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.output_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x, mask=None):
        z = self.input_net(x)
        z = self.transformer(z, src_key_padding_mask=mask)
        
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)             
            z = z * valid                                     
            denom = valid.sum(dim=1).clamp_min(1.0)           
            z = z.sum(dim=1) / denom                          
        else:
            z = z.mean(dim=1)
        return torch.sigmoid(self.output_net(z).squeeze(-1))

# Mixup augmentation
def _mixup_synthetic_patients(X: np.ndarray, pat_ids: np.ndarray,
                              celltypes: np.ndarray, patients_tr: list[str],
                              labels_tr: np.ndarray, augment_num: int,
                              alpha: float, min_size: int, inter_only: bool,
                              rng: np.random.Generator):
    if augment_num <= 0 or len(patients_tr) < 2:
        return None

    pat_to_lab = dict(zip(patients_tr, labels_tr))
    yes_pats = [p for p in patients_tr if pat_to_lab[p] >= 0.5]
    no_pats  = [p for p in patients_tr if pat_to_lab[p] <  0.5]
    if inter_only and (len(yes_pats) == 0 or len(no_pats) == 0):
        return None

    pat_idx = {p: np.where(pat_ids == p)[0] for p in set(patients_tr)}

    syn_X_list, syn_pid_list, syn_ct_list = [], [], []
    syn_pats, syn_labs = [], []
    NOISE_SIGMA = (1e-5) ** 0.5

    for j in range(augment_num):
        if inter_only:
            p1 = yes_pats[rng.integers(len(yes_pats))]
            p2 = no_pats [rng.integers(len(no_pats))]
        else:
            p1, p2 = rng.choice(patients_tr, size=2, replace=False)
        lam = float(rng.beta(alpha, alpha))
        idx_1 = pat_idx[p1]; idx_2 = pat_idx[p2]
        ct_1  = celltypes[idx_1]; ct_2 = celltypes[idx_2]
        cts_union = sorted(set(ct_1.tolist()) | set(ct_2.tolist()))

        syn_pat_id = f"mixup_{j:04d}"
        per_ct_mix = []
        for ct in cts_union:
            sub_1 = idx_1[ct_1 == ct]
            sub_2 = idx_2[ct_2 == ct]
            n_target = max(int(min_size * (
                lam * len(sub_1) / max(len(idx_1), 1) +
                (1 - lam) * len(sub_2) / max(len(idx_2), 1))), 1)
            
            unique_to_one = (len(sub_1) == 0) ^ (len(sub_2) == 0)
            if len(sub_1) > 0:
                s1 = sub_1[rng.integers(len(sub_1), size=n_target)]
                x1 = X[s1]
            else:
                x1 = np.zeros((n_target, X.shape[1]), dtype=X.dtype)
            if len(sub_2) > 0:
                s2 = sub_2[rng.integers(len(sub_2), size=n_target)]
                x2 = X[s2]
            else:
                x2 = np.zeros((n_target, X.shape[1]), dtype=X.dtype)
            x_mix = lam * x1 + (1 - lam) * x2
            if unique_to_one:
                x_mix = x_mix + rng.normal(0.0, NOISE_SIGMA, x_mix.shape).astype(np.float32)
            per_ct_mix.append((x_mix, np.full(n_target, ct, dtype=object)))

        syn_cells = np.concatenate([m[0] for m in per_ct_mix], axis=0)
        syn_cts   = np.concatenate([m[1] for m in per_ct_mix], axis=0)
        syn_X_list.append(syn_cells)
        syn_pid_list.append(np.full(syn_cells.shape[0], syn_pat_id, dtype=object))
        syn_ct_list.append(syn_cts)
        syn_pats.append(syn_pat_id)
        syn_labs.append(lam * float(pat_to_lab[p1]) + (1 - lam) * float(pat_to_lab[p2]))

    return (np.concatenate(syn_X_list, axis=0),
            np.concatenate(syn_pid_list, axis=0),
            np.concatenate(syn_ct_list, axis=0),
            syn_pats,
            np.array(syn_labs, dtype=np.float32))


# Sampling: create patient bags



def load_cohort(cohort: str, n_genes: int = N_GENES_TOP, h5ad_path=None,
                prior_path=None):
    import pandas as pd
    from utils.data_helpers import load_cells_cohort
    from utils.celltype_groups import infer_celltype_groups, EXCLUDE_LABEL
    X, pat_ids, celltypes, _, patients, labels = load_cells_cohort(
        cohort, n_genes, h5ad_path, prior_path)

    _, ct_map = infer_celltype_groups(
        pd.DataFrame({"final_celltype": celltypes, "patient_id": pat_ids}),
        min_patients=3, min_cells_per_patient=10,
        split_groups=["T_cells", "Monocytes", "Dendritic"])
    celltypes = np.array([ct_map.get(c, EXCLUDE_LABEL) for c in celltypes])
    keep = (celltypes != EXCLUDE_LABEL) & (celltypes != "Other")
    X, pat_ids, celltypes = X[keep], pat_ids[keep], celltypes[keep]

    pat_to_lab = {p: l for p, l in zip(patients, labels)}
    cell_labels = np.array([pat_to_lab[p] for p in pat_ids], dtype=np.float32)
    print(f"Cohort {cohort}: {len(patients)} patients, {X.shape[1]} genes, {len(X)} cells")
    print(f"  irAE Yes={labels.sum():.0f}  No={len(labels)-labels.sum():.0f}")
    return X, cell_labels, pat_ids, labels, patients, celltypes

def train_fold(X: np.ndarray, pat_ids: np.ndarray,
               labels: np.ndarray, patients_tr: list[str],
               patients_te: list[str], patients_all: list[str],
               labels_all: np.ndarray,
               n_genes: int,
               sample_cells: int = SAMPLE_CELLS,
               train_num_sample: int = TRAIN_NUM_SAMPLE,
               test_num_sample: int = TEST_NUM_SAMPLE,
               celltypes: np.ndarray = None,
               seed: int = 0) -> np.ndarray:
    
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import torch.nn.functional as F

    rng = np.random.default_rng(RANDOM_STATE + seed)
    pat_to_lab = dict(zip(patients_all, labels_all))
    all_pats = set(patients_tr) | set(patients_te)

    X_for_bags = X
    n_genes_eff = X.shape[1]
    if USE_PCA:
        tr_cell_mask = np.isin(pat_ids, list(patients_tr))
        n_comp = min(PCA_COMPONENTS, X.shape[1], int(tr_cell_mask.sum()) - 1)
        if n_comp >= 2:
            sc = StandardScaler(with_mean=True, with_std=True)
            X_tr_scaled = sc.fit_transform(X[tr_cell_mask])
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE + seed)
            pca.fit(X_tr_scaled)
            
            X_for_bags = pca.transform(sc.transform(X)).astype(np.float32)
            n_genes_eff = n_comp

    pat_idx_map = {p: np.where(pat_ids == p)[0] for p in all_pats}

    def _build_bags(X_src, pat_idx_src, pat_lab_src, pats, num_sample):
        
        bags, bag_labels, masks = [], [], []
        for p in pats:
            p_idx = pat_idx_src[p]
            y = pat_lab_src[p]
            n_cells = len(p_idx)
            if n_cells == 0:
                continue
            for _ in range(num_sample):
                bag = np.zeros((sample_cells, X_src.shape[1]), dtype=np.float32)
                mask = np.ones(sample_cells, dtype=bool)
                if n_cells >= sample_cells:
                    sel = rng.choice(p_idx, sample_cells, replace=False)
                    bag[:] = X_src[sel]
                    mask[:] = False
                else:
                    bag[:n_cells] = X_src[p_idx]
                    mask[:n_cells] = False
                bags.append(bag)
                bag_labels.append(y)
                masks.append(mask)
        if not bags:
            return None, None, None
        return np.stack(bags), np.array(bag_labels, dtype=np.float32), np.stack(masks)

    y_tr_arr = np.array([pat_to_lab[p] for p in patients_tr])
    k = 0
    while True:
        p_train, p_val, y_spl_tr, y_spl_va = train_test_split(
            list(patients_tr), y_tr_arr, test_size=0.33,
            random_state=RANDOM_STATE + seed + k)
        if len(set(y_spl_tr)) == 2 and len(set(y_spl_va)) == 2:
            break
        k += 1
        if k > 100:
            p_train, p_val = list(patients_tr), list(patients_tr)
            break

    train_X = X_for_bags
    train_pat_idx = pat_idx_map.copy()
    train_pat_lab = {p: float(pat_to_lab[p]) for p in p_train}
    pre_n_pats = len(p_train)

    n_train_real = len(p_train)
    eff_augment_num = AUGMENT_NUM

    if eff_augment_num > 0 and celltypes is not None:
        result = _mixup_synthetic_patients(
            X_for_bags, pat_ids, celltypes,
            patients_tr=p_train,
            labels_tr=np.array([pat_to_lab[p] for p in p_train]),
            augment_num=eff_augment_num, alpha=MIXUP_ALPHA,
            min_size=MIN_SIZE, inter_only=INTER_ONLY, rng=rng)
        if result is not None:
            syn_X, syn_pids, syn_cts, syn_pats, syn_labs = result
            
            offset = len(X_for_bags)
            train_X = np.concatenate([X_for_bags, syn_X], axis=0)
            for sp in syn_pats:
                m = (syn_pids == sp)
                train_pat_idx[sp] = offset + np.where(m)[0]
            for sp, sl in zip(syn_pats, syn_labs):
                train_pat_lab[sp] = float(sl)
            
            if INTER_ONLY:
                p_train_aug = list(syn_pats)
            else:
                p_train_aug = list(p_train) + list(syn_pats)
        else:
            p_train_aug = list(p_train)
    else:
        p_train_aug = list(p_train)

    tr_bags, tr_labels, tr_masks = _build_bags(
        train_X, train_pat_idx, train_pat_lab, p_train_aug, train_num_sample)
    if tr_bags is None:
        return np.full(len(patients_te), 0.5)
    val_bags, val_labels, val_masks = _build_bags(
        X_for_bags, pat_idx_map, {p: float(pat_to_lab[p]) for p in p_val},
        p_val, test_num_sample)
    if val_bags is None:
        val_bags, val_labels, val_masks = tr_bags, tr_labels, tr_masks

    if eff_augment_num > 0 and celltypes is not None:
        n_real_used = sum(1 for p in p_train_aug if p in set(p_train))
        n_syn_used  = len(p_train_aug) - n_real_used
        mode = "synthetic-only (INTER_ONLY)" if INTER_ONLY else "real + synthetic"
        print(f"      mixup [{mode}]: {n_real_used} real + {n_syn_used} synthetic "
              f"→ {len(tr_bags)} training bags  (alpha={MIXUP_ALPHA}, "
              f"min_size={MIN_SIZE}, augment_num={eff_augment_num})")

    model = ScRAT(n_genes_eff, EMB_DIM, N_HEADS, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    best_val_loss, best_state = float("inf"), None
    trigger_times = 0
    val_loss_history = []
    n_train = len(tr_bags)
    bs = min(BATCH_SIZE, n_train)

    for epoch in range(N_EPOCHS):
        model.train()
        perm = rng.permutation(n_train)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(tr_bags[idx]).to(DEVICE)
            yb = torch.from_numpy(tr_labels[idx]).to(DEVICE)
            mb = torch.from_numpy(tr_masks[idx]).to(DEVICE)

            pred = model(xb, mb)
            loss = F.binary_cross_entropy(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            n_val = len(val_bags)
            vbs = min(BATCH_SIZE, n_val)
            for i in range(0, n_val, vbs):
                xb = torch.from_numpy(val_bags[i:i + vbs]).to(DEVICE)
                yb = torch.from_numpy(val_labels[i:i + vbs]).to(DEVICE)
                mb = torch.from_numpy(val_masks[i:i + vbs]).to(DEVICE)
                pred = model(xb, mb)
                val_losses.append(F.binary_cross_entropy(pred, yb).item())

        val_loss = float(np.mean(val_losses))
        val_loss_history.append(val_loss)

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Early stop
        if (epoch > (N_EPOCHS - 50) and len(val_loss_history) >= 2
                and val_loss > val_loss_history[-2]):
            trigger_times += 1
            if trigger_times >= PATIENCE:
                break
        else:
            trigger_times = 0

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test predictions
    model.eval()
    probs = []
    with torch.no_grad():
        for p in patients_te:
            p_idx = pat_idx_map[p]
            n_cells = len(p_idx)
            if n_cells == 0:
                probs.append(0.5)
                continue
            p_probs = []
            for _ in range(test_num_sample):
                bag = np.zeros((1, sample_cells, n_genes_eff), dtype=np.float32)
                mask = np.ones((1, sample_cells), dtype=bool)
                if n_cells >= sample_cells:
                    sel = rng.choice(p_idx, sample_cells, replace=False)
                    bag[0] = X_for_bags[sel]
                    mask[0] = False
                else:
                    bag[0, :n_cells] = X_for_bags[p_idx]
                    mask[0, :n_cells] = False
                xb = torch.from_numpy(bag).to(DEVICE)
                mb = torch.from_numpy(mask).to(DEVICE)
                p_probs.append(model(xb, mb).item())
            probs.append(float(np.mean(p_probs)))
    return np.array(probs)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--n-genes", type=int, default=N_GENES_TOP)
    p.add_argument("--h5ad", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--prior", default=None)
    p.add_argument("--no-pca", action="store_true",
                   help="Skip PCA preprocessing (raw gene features). "
                        "Default ON matches yuzhenmao/ScRAT run.sh; OFF "
                        "tests whether PCA filters out pathway-aligned "
                        "signal in irAE prediction.")
    p.add_argument("--train-num-sample", type=int, default=None,
                   help="Override TRAIN_NUM_SAMPLE (paper: 20). Lowering "
                        "reduces memory linearly for the bag tensor.")
    p.add_argument("--min-size", type=int, default=None,
                   help="Override MIN_SIZE for mixup (paper: 10000). "
                        "Lowering reduces synthetic-cell memory linearly.")
    return p.parse_args()


def main():
    args = parse_args()
    prof_start("scrat", args.cohort)
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    if args.no_pca:
        global USE_PCA
        USE_PCA = False
        print("  [ScRAT] PCA preprocessing DISABLED (--no-pca)")

    if args.train_num_sample is not None:
        global TRAIN_NUM_SAMPLE
        TRAIN_NUM_SAMPLE = args.train_num_sample
        print(f"  [ScRAT] TRAIN_NUM_SAMPLE override: {TRAIN_NUM_SAMPLE} (paper default 20)")
    if args.min_size is not None:
        global MIN_SIZE
        MIN_SIZE = args.min_size
        print(f"  [ScRAT] MIN_SIZE override: {MIN_SIZE} (paper default 10000)")

    out_dir = Path(args.out_dir) if args.out_dir else (
        RESULTS_BASELINES / args.cohort / ("scrat_nopca" if args.no_pca else "scrat"))
    out_dir.mkdir(parents=True, exist_ok=True)

    X, cell_labs, pat_ids, labels, patients, celltypes = load_cohort(
        args.cohort, args.n_genes, args.h5ad, prior_path=args.prior)

    sorted_patients, sorted_labels, folds = get_cv_folds(patients, labels)
    patients = sorted_patients
    labels = sorted_labels

    print(f"\n[Patient-level evaluation (all cells)]")
    eff_mode = effective_cv_mode(len(labels))
    is_loocv = eff_mode == "loocv"
    pat_fold_aucs, pat_fold_aps = [], []
    pat_oof_probs = np.full(len(labels), np.nan, dtype=np.float64)
    pat_oof_counts = np.zeros(len(labels), dtype=np.int32)

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        patients_tr = [patients[i] for i in tr_idx]
        patients_te = [patients[i] for i in te_idx]

        probs = train_fold(X, pat_ids, labels, patients_tr,
                           patients_te, patients, labels,
                           X.shape[1], celltypes=celltypes, seed=fold_i)

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
