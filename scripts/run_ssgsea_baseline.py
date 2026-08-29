#!/usr/bin/env python3
"""ssGSEA Hallmark baseline — established pathway scoring, no learned representation.

Reviewer 1 asks for comparison against "established pathway scoring approaches
and simpler pathway-based predictive models". This is that comparison:

    expression -> patient x cell-type pseudobulk -> ssGSEA Hallmark scores
               -> cell-type gate + patient-level logistic regression

ssGSEA rather than GSVA, deliberately. GSVA estimates gene-wise expression
distributions *across* samples, so a held-out patient's expression shifts every
other patient's scores in a gene-dependent way that does not cancel downstream.
ssGSEA ranks genes *within* each sample, so the enrichment step is sample-local.

The normalised scores are not, however, literally sample-independent: gseapy
returns NES, and the Barbie-2009 convention divides by a dataset-wide range.
Verified empirically on this implementation — adding samples rescales scores by
a *single global constant* (0.8167 in the test), identical across every sample
and every pathway to within 5e-8. Because the downstream standardises features
per cell type within each fold, that constant cancels exactly: standardised
features were identical to within 8e-7 whether a profile was scored alone or
alongside others. The held-out patient therefore cannot influence the fitted
model, even though the raw NES values are not invariant.

To reproduce: score one pseudobulk profile alone, then again with others, and
compare the 50 pathway scores before and after StandardScaler.

Everything downstream is identical to irAEGIS — the same inner-LOOCV cell-type
gate and the same stacked patient-level classifier — so any gap is attributable
to the representation, not the classifier.

Complements run_hallmark_baseline.py, which uses the naive mean-expression
score. Reporting both separates "fixed pathway scores don't work here" from
"that particular fixed aggregation is unstable".

Outputs to results/iraegis/<cohort>/ssgsea_baseline/
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import RESULTS_IRAEGIS, COHORTS_REAL
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import train_h_concat_gated_concat_en
from models.iraegis.fold_selection import select_ct_groups

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
GMT = REPO / "datasets" / "resources" / "h.all.v2026.1.Hs.symbols.gmt"
N_BOOTSTRAP = 1000


def load_gmt(path: Path) -> dict:
    sets = {}
    for line in path.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def pseudobulk(X, pat_ids, ct_ids, patients, n_ct):
    """Mean expression per (patient, cell type). Returns (profiles, keys)."""
    cols, keys = [], []
    for pi, p in enumerate(patients):
        for j in range(n_ct):
            m = (pat_ids == p) & (ct_ids == j)
            if m.any():
                cols.append(X[m].mean(axis=0)); keys.append((pi, j))
    return np.asarray(cols, dtype=np.float32), keys


def ssgsea_scores(profiles, genes, gene_sets, threads=4):
    """ssGSEA NES per (profile x pathway). Ranks within each sample."""
    import gseapy
    df = pd.DataFrame(profiles.T, index=list(genes),
                      columns=[f"S{i}" for i in range(profiles.shape[0])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = gseapy.ssgsea(data=df, gene_sets=gene_sets, outdir=None,
                          no_plot=True, min_size=5, threads=threads,
                          permutation_num=0, verbose=False)
    wide = r.res2d.pivot(index="Name", columns="Term", values="NES").astype(float)
    wide = wide.reindex([f"S{i}" for i in range(profiles.shape[0])])
    return wide.fillna(0.0).to_numpy(np.float32), list(wide.columns)


def _boot(y, p, metric, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [metric(y[i], p[i]) for i in
         (np.concatenate([rng.choice(pos, len(pos), True),
                          rng.choice(neg, len(neg), True)]) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run_cohort(cohort: str) -> dict:
    print(f"\n{'='*70}\n  ssGSEA Hallmark baseline: {cohort}\n{'='*70}")
    gene_sets = load_gmt(GMT)
    print(f"  {len(gene_sets)} Hallmark gene sets from {GMT.name}")

    X, obs, _gn, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=False,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    genes = list(prior["gene_names"])
    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    print(f"  {X.shape[0]:,} cells x {X.shape[1]:,} genes, {len(patients)} patients")

    out = RESULTS_IRAEGIS / cohort / "ssgsea_baseline"; out.mkdir(parents=True, exist_ok=True)
    oof, rows, t0 = np.full(len(patients), np.nan), [], time.time()

    for i, held in enumerate(patients):
        # cell-type grouping from training-fold patients only
        groups_f, ct_all, keep = select_ct_groups(obs, pat_ids != held, SPLIT_CT_GROUPS)
        Xf, ctf, patf = X[keep], ct_all[keep], pat_ids[keep]

        prof, keys = pseudobulk(Xf, patf, ctf, patients, len(groups_f))
        S, pw = ssgsea_scores(prof, genes, gene_sets)

        # one "cell" per (patient, CT) so the standard aggregation is a no-op
        h = S
        h_pat = np.array([patients[k[0]] for k in keys])
        h_ct = np.array([k[1] for k in keys], dtype=np.int64)

        hi = patients.index(held)
        res = train_h_concat_gated_concat_en(
            h, h_pat, h_ct, pat_labels, groups_f, verbose=False, only_patient_idx=hi)
        oof[hi] = res["oof_probs"][hi]
        rows.append({"patient": held, "label": int(y[hi]), "oof_prob": float(oof[hi]),
                     "n_ct": len(groups_f), "n_profiles": len(keys),
                     "n_pathways": S.shape[1]})
        pd.DataFrame(rows).to_csv(out / "per_fold.csv", index=False)
        print(f"  [{i+1}/{len(patients)}] {held} (label {y[hi]}): {oof[hi]:.4f}   "
              f"{len(groups_f)} CTs, {len(keys)} profiles, {S.shape[1]} pathways")

    auc, ap = float(roc_auc_score(y, oof)), float(average_precision_score(y, oof))
    alo, ahi = _boot(y, oof, roc_auc_score); plo, phi = _boot(y, oof, average_precision_score)
    summary = {"cohort": cohort,
               "model": "ssGSEA Hallmark scores on patient x cell-type pseudobulk; "
                        "no autoencoder, no latent representation",
               "why_ssgsea_not_gsva": "ssGSEA ranks genes within each sample, so it is "
                        "leakage-free; GSVA uses cross-sample gene distributions",
               "n_patients": len(patients), "n_positive": int(y.sum()),
               "auc": auc, "auc_ci95": [alo, ahi], "auprc": ap, "auprc_ci95": [plo, phi],
               "n_bootstrap": N_BOOTSTRAP, "total_seconds": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  AUC   = {auc:.4f}  95% CI [{alo:.3f}, {ahi:.3f}]")
    print(f"  AUPRC = {ap:.4f}  95% CI [{plo:.3f}, {phi:.3f}]\n  -> {out}")
    return summary


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cohort"); ap_.add_argument("--all", action="store_true")
    a = ap_.parse_args()
    cohorts = [a.cohort] if a.cohort else (list(COHORTS_REAL) if a.all
               else ap_.error("pass --cohort or --all"))
    res = [run_cohort(c) for c in cohorts]
    if len(res) > 1:
        print(f"\n{'='*70}\n  SUMMARY — ssGSEA Hallmark baseline\n{'='*70}")
        for s in res:
            ci = f"[{s['auc_ci95'][0]:.3f}, {s['auc_ci95'][1]:.3f}]"
            print(f"  {s['cohort']:<34} {s['auc']:>7.4f} {ci:>16}")


if __name__ == "__main__":
    main()
