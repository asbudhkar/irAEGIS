#!/usr/bin/env python3
"""Competing baselines re-run under the leakage-free per-fold protocol (R3.6).

Reviewer 3.6 asks that every method be evaluated with "exactly the same data
partitions, preprocessing procedures, feature-selection rules, and leakage-free
validation protocol". irAEGIS now meets this; the baselines did not.

The leak in the original baseline runs was NOT label leakage. It was gene
selection: highly-variable genes were chosen once by variance over ALL patients,
outside the cross-validation loop (see scripts/tune_cell_lr_post_hvg.py, where
the selection sits in run_cohort rather than inside the fold). Per-patient
scaling was already fold-correct. This harness moves HVG selection inside the
fold, so the held-out patient contributes to nothing:

    for each held-out patient:
        select genes from TRAINING cells only, by default using irAEGIS's exact
        rule (every pathway-active gene plus the top 2000 non-pathway HVGs),
        so feature selection is identical across all methods as R3.6 requires
        aggregate to patient level on that gene set
        fit the classifier on TRAINING patients only
        score the held-out patient

Six methods are covered - the ones implemented internally, all of which follow
"aggregate to patient level, then classify":

    cell_lr              pseudobulk -> scale -> logistic regression
    cell_mlp             pseudobulk -> small MLP
    rf_pseudobulk        pseudobulk -> random forest
    xgboost_pseudobulk   pseudobulk -> gradient boosting
    pseudobulk_en        per-CT pseudobulk -> per-CT PCA(2) -> concat -> LR
    pseudobulk_en_gated  as above, restricted to the top-K cell types by
                         training-fold AUC

The three external codebases (ScRAT, singleDeep, hierarchical MIL) have their own
preprocessing pipelines and are not covered here; see docs/revision_parked_items.md.

Model settings are taken from the original tuning scripts so the only thing that
changes is WHEN gene selection happens. cell_mlp uses the ORIGINAL SimpleMLP from
models/baselines/cell_mlp.py (Linear-LayerNorm-GELU x2, HIDDEN=32, LR=1e-3,
WD=1e-2, 15 epochs, batch 2048) rather than a scikit-learn stand-in, so no method
is altered by this harness.

Outputs to results/iraegis/<cohort>/baselines_leakage_free/

Usage:
    python scripts/run_baselines_leakage_free.py --cohort GSE189125_pre_ici
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.config import RESULTS_IRAEGIS, RANDOM_STATE
from models.iraegis.data_utils import load_cohort_data
from utils import profiler
from models.iraegis.fold_selection import select_ct_groups, select_hvg_genes
# Faithful copy of scPIP's models/baselines/cell_mlp.py SimpleMLP and its tuned
# hyperparameters. Copied rather than imported: irAEGIS has its own `models` and
# `utils` packages which shadow scPIP's, so importing that module here resolves
# the wrong `utils.config`. Architecture and settings are byte-for-byte the
# published ones (HIDDEN=32, LR=1e-3, WD=1e-2, 15 epochs, batch 2048), so the
# method is unchanged - only the gene set differs.
MLP_HIDDEN, MLP_LR, MLP_WD, MLP_EPOCHS, MLP_BATCH = 32, 1e-3, 1e-2, 15, 2048


def _build_simple_mlp(n_in, hidden=MLP_HIDDEN):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden // 2),
        nn.LayerNorm(hidden // 2),
        nn.GELU(),
        nn.Linear(hidden // 2, 1),
    )

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
HVG_K = 2000                 # POST_HVG_K in the original cell_lr script
N_PCA = 2                    # pseudobulk_en used PCA(2)
GATE_TOPK = 5
N_BOOT = 1000
_CHUNK = 2000
GENE_SPACE = "matched"          # set from --gene-space; default matches irAEGIS
SHORT = {"cell_lr": "lr", "cell_mlp": "mlp", "rf_pseudobulk": "rf",
         "xgboost_pseudobulk": "xgb", "pseudobulk_en": "en",
         "pseudobulk_en_gated": "en_gated"}


def hvg_by_variance(X, train_mask, k=HVG_K):
    """Top-k genes by variance over TRAINING cells only, computed in chunks so
    the full training submatrix is never materialised."""
    rows = np.where(train_mask)[0]
    n = len(rows)
    var = np.empty(X.shape[1], dtype=np.float64)
    for s in range(0, X.shape[1], _CHUNK):
        sub = X[np.ix_(rows, np.arange(s, min(s + _CHUNK, X.shape[1])))].astype(np.float64)
        m = sub.sum(0) / n
        var[s:s + sub.shape[1]] = (sub * sub).sum(0) / n - m * m
    return np.sort(np.argsort(var)[-k:])


def pseudobulk(X, pat_ids, patients):
    out = np.zeros((len(patients), X.shape[1]), dtype=np.float32)
    for i, p in enumerate(patients):
        m = pat_ids == p
        if m.any():
            out[i] = X[m].mean(0)
    return out


def per_ct_pseudobulk(X, pat_ids, ct_ids, patients, n_ct):
    return {j: pseudobulk(X[ct_ids == j], pat_ids[ct_ids == j], patients)
            for j in range(n_ct)}


def _models():
    R = RANDOM_STATE
    return {
        "cell_lr": lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=R),

        "rf_pseudobulk": lambda: RandomForestClassifier(n_estimators=500, max_depth=3,
                                                        min_samples_leaf=3,
                                                        class_weight="balanced",
                                                        random_state=R, n_jobs=1),
        "xgboost_pseudobulk": None,      # constructed lazily, see below
    }


def _mlp_fit_score(Xtr, ytr, Xte):
    """Original SimpleMLP trained on patient pseudobulk, published settings."""
    import torch, torch.nn.functional as F
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sc = StandardScaler().fit(Xtr)
    xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    xe = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                      dtype=torch.float32, device=dev)
    torch.manual_seed(RANDOM_STATE)
    clf = _build_simple_mlp(xt.shape[1]).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(MLP_EPOCHS):
        clf.train()
        perm = rng.permutation(len(xt))
        for i in range(0, len(xt), MLP_BATCH):
            b = perm[i:i + MLP_BATCH]
            loss = F.binary_cross_entropy_with_logits(
                clf(xt[b]).squeeze(-1), yt[b], pos_weight=pw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0); opt.step()
    clf.eval()
    with torch.no_grad():
        return float(torch.sigmoid(clf(xe).squeeze(-1)).cpu().numpy()[0])


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.1,
                         random_state=RANDOM_STATE, n_jobs=1,
                         eval_metric="logloss", verbosity=0)


def _fit_score(model, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    try:
        model.fit(sc.transform(Xtr), ytr)
        return float(model.predict_proba(sc.transform(Xte))[0, 1])
    except Exception:
        return 0.5


def _en_features(per_ct, cts, tr, te, n_pca):
    """per-CT PCA fitted on TRAINING patients only, then concatenated."""
    a, b = [], []
    for j in cts:
        M = per_ct[j]
        sc = StandardScaler().fit(M[tr])
        A, B = sc.transform(M[tr]), sc.transform(M[[te]])
        k = max(1, min(n_pca, len(tr) - 1, A.shape[1]))
        p = PCA(n_components=k, random_state=RANDOM_STATE).fit(A)
        a.append(p.transform(A)); b.append(p.transform(B))
    return np.concatenate(a, 1), np.concatenate(b, 1)


def _boot(y, p, metric, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        v.append(metric(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run(cohort):
    print(f"\n{'='*70}\n  Leakage-free baselines: {cohort}\n{'='*70}", flush=True)
    X, obs, _g, _cg, _ci, pat, lab, prior = load_cohort_data(
        cohort, prior_genes_only=False,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    patients = sorted(lab.keys()); y = np.array([lab[p] for p in patients], dtype=np.int64)
    print(f"  {X.shape[0]:,} cells x {X.shape[1]:,} genes, {len(patients)} patients "
          f"({y.sum()} positive); gene space={GENE_SPACE}", flush=True)

    METHODS = ["cell_lr", "cell_mlp", "rf_pseudobulk", "xgboost_pseudobulk",
               "pseudobulk_en", "pseudobulk_en_gated"]
    oof = {m: np.full(len(patients), np.nan) for m in METHODS}
    profiler.start(f"internal_baselines_leakage_free_{GENE_SPACE}", cohort)
    recs, t0 = [], time.time()
    elapsed = {m: 0.0 for m in METHODS}   # per-method wall clock (R1.7)

    for i, held in enumerate(patients):
        train_cells = pat != held
        if GENE_SPACE == "matched":
            # exactly irAEGIS's rule: every pathway-active gene plus the top
            # HVG_BACKFILL non-pathway genes ranked on training cells only
            genes = select_hvg_genes(X, prior["mask"], train_cells)
        else:
            genes = hvg_by_variance(X, train_cells)      # TRAINING cells only
        Xf = X[:, genes]
        groups, ct_all, keep = select_ct_groups(obs, train_cells, SPLIT_CT_GROUPS)
        hi = patients.index(held)
        tr = np.array([k for k in range(len(patients)) if k != hi])

        pb = pseudobulk(Xf, pat, patients)
        mk = _models()
        for m in ["cell_lr", "rf_pseudobulk"]:
            _t = time.time()
            oof[m][hi] = _fit_score(mk[m](), pb[tr], y[tr], pb[[hi]])
            elapsed[m] += time.time() - _t
        _t = time.time()
        oof["cell_mlp"][hi] = _mlp_fit_score(pb[tr], y[tr], pb[[hi]])
        elapsed["cell_mlp"] += time.time() - _t
        _t = time.time()
        oof["xgboost_pseudobulk"][hi] = _fit_score(_xgb(), pb[tr], y[tr], pb[[hi]])
        elapsed["xgboost_pseudobulk"] += time.time() - _t

        per_ct = per_ct_pseudobulk(Xf[keep], pat[keep], ct_all[keep], patients, len(groups))
        _t = time.time()
        A, B = _en_features(per_ct, range(len(groups)), tr, hi, N_PCA)
        oof["pseudobulk_en"][hi] = _fit_score(
            LogisticRegression(C=1.0, penalty="l2", max_iter=2000,
                               class_weight="balanced", random_state=RANDOM_STATE),
            A, y[tr], B)
        elapsed["pseudobulk_en"] += time.time() - _t

        _t = time.time()
        # gated variant: rank cell types by inner-LOOCV AUC on TRAINING patients
        scores = []
        for j in range(len(groups)):
            M = per_ct[j]; pr = np.zeros(len(tr))
            for a_, b_ in LeaveOneOut().split(tr):
                t2 = tr[a_]
                if len(set(y[t2])) < 2: pr[b_[0]] = .5; continue
                pr[b_[0]] = _fit_score(
                    LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE),
                    M[t2], y[t2], M[tr[b_]])
            try: scores.append(roc_auc_score(y[tr], pr))
            except Exception: scores.append(0.5)
        top = list(np.argsort(scores)[::-1][:GATE_TOPK])
        A, B = _en_features(per_ct, top, tr, hi, N_PCA)
        oof["pseudobulk_en_gated"][hi] = _fit_score(
            LogisticRegression(C=1.0, penalty="l2", max_iter=2000,
                               class_weight="balanced", random_state=RANDOM_STATE),
            A, y[tr], B)
        elapsed["pseudobulk_en_gated"] += time.time() - _t

        recs.append({"patient": held, "label": int(y[hi]), "n_genes": len(genes),
                     "n_ct": len(groups), **{m: float(oof[m][hi]) for m in METHODS}})
        out = RESULTS_IRAEGIS / cohort / f"baselines_leakage_free_{GENE_SPACE}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_csv(out / "per_fold.csv", index=False)
        print(f"  [{i+1}/{len(patients)}] {held}: " +
              " ".join(f"{SHORT[m]}={oof[m][hi]:.2f}" for m in METHODS) +
              f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
        del Xf, per_ct

    profiler.stop()
    summary = {"cohort": cohort, "gene_space": GENE_SPACE,
               "hvg_k": HVG_K, "n_pca": N_PCA,
               "protocol": "leakage-free per-fold: HVG selected on training cells "
                           "only; cell-type grouping, scaling, PCA and classifiers "
                           "fit on training patients only",
               "methods": {}}
    print(f"\n  {'method':<22} {'AUC':>8} {'95% CI':>18} {'AUPRC':>8}", flush=True)
    print("  " + "-" * 60, flush=True)
    for m in METHODS:
        k = ~np.isnan(oof[m])
        if k.sum() < 3: continue
        a = float(roc_auc_score(y[k], oof[m][k])); lo, hi_ = _boot(y[k], oof[m][k], roc_auc_score)
        q = float(average_precision_score(y[k], oof[m][k]))
        summary["methods"][m] = {"auc": a, "auc_ci95": [lo, hi_], "auprc": q,
                                 "fit_seconds_total": round(elapsed[m], 1),
                                 "fit_seconds_per_fold": round(elapsed[m] / max(len(recs), 1), 2)}
        print(f"  {m:<22} {a:>8.4f} {f'[{lo:.3f}, {hi_:.3f}]':>18} {q:>8.4f} "
              f"{elapsed[m]/max(len(recs),1):>9.2f}", flush=True)
    summary["total_seconds"] = time.time() - t0
    (RESULTS_IRAEGIS / cohort / f"baselines_leakage_free_{GENE_SPACE}" / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"  -> {RESULTS_IRAEGIS / cohort / f'baselines_leakage_free_{GENE_SPACE}'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--gene-space", choices=["matched", "hvg2000"], default="matched",
                    help="matched (default) = the published rule these baselines "
                         "already used - pathway-active genes + top-2000 "
                         "non-pathway HVG - but ranked on training cells only. "
                         "hvg2000 = top-2000 by variance with no pathway prior, "
                         "a stricter no-prior variant.")
    a = ap.parse_args()
    global GENE_SPACE; GENE_SPACE = a.gene_space
    run(a.cohort)


if __name__ == "__main__":
    main()
