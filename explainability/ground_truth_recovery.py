#!/usr/bin/env python3

# Ground-truth recovery on simulated cohorts: Pathway recall@5 and Gene precision@30

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IRAEGIS_DIR = REPO / "results" / "iraegis"

SIM_DIR = {
    "RS": REPO / "datasets" / "simulation_rs",
    "DS": REPO / "datasets" / "simulation_ds",
}
COHORT_GT = {f"{prefix}_cohort{n}": SIM_DIR[prefix]
             for prefix in ("RS", "DS") for n in (1, 2, 3)}


def load_gt(cohort: str) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    # Return GT pathway rows for a cohort + {pathway → gene_set} for that cohort.
    sim_dir = COHORT_GT[cohort]
    gt = pd.read_csv(sim_dir / "ground_truth_pathways.csv")
    gt = gt[gt["cohort"] == cohort].copy()

    prior = np.load(sim_dir / "sim_pathway_prior.npz", allow_pickle=True)
    pw_names = list(prior["pathway_names"])
    gene_names = list(prior["gene_names"])
    mask = prior["mask"]  # (n_genes, n_pathways)
    pw_to_genes = {p: set(np.array(gene_names)[mask[:, j].astype(bool)])
                   for j, p in enumerate(pw_names)}
    return gt, pw_to_genes


def recall_at_k(pw_df: pd.DataFrame, gt: pd.DataFrame, k: int) -> float:
    # Mean over signal CTs of (#GT pathways ∩ top-k attributed) / (#GT pathways).

    per_ct = []
    for ct, sub in gt.groupby("celltype"):
        gt_set = set(sub["pathway"])
        ct_rows = pw_df[pw_df["celltype"] == ct].copy()
        if ct_rows.empty:
            per_ct.append(0.0)
            continue
        ct_rows["_rank_val"] = ct_rows["h_diff"].abs()
        topk = ct_rows.nlargest(k, "_rank_val")["pathway"]
        per_ct.append(len(gt_set & set(topk)) / max(len(gt_set), 1))
    return float(np.mean(per_ct)) if per_ct else float("nan")


def precision_at_30(gene_df: pd.DataFrame, gt: pd.DataFrame,
                    pw_to_genes: dict[str, set[str]]) -> float:
    # Mean over signal CTs of (#top-30 genes in GT pathways) / 30.
    per_ct = []
    for ct, sub in gt.groupby("celltype"):
        gt_genes: set[str] = set()
        for p in sub["pathway"]:
            gt_genes |= pw_to_genes.get(p, set())
        top30 = gene_df[gene_df["celltype"] == ct].nsmallest(30, "rank")["gene"]
        per_ct.append(sum(g in gt_genes for g in top30) / 30.0)
    return float(np.mean(per_ct)) if per_ct else float("nan")


def score_cohort(cohort: str) -> dict:
    expl_dir = IRAEGIS_DIR / cohort / "cell_explainability"
    pw_path = expl_dir / "cell_pathway_attribution.csv"
    gn_path = expl_dir / "cell_gene_attribution.csv"
    if not pw_path.exists() or not gn_path.exists():
        print(f"  [{cohort}] SKIP — missing attribution file(s)")
        return None
    pw_df = pd.read_csv(pw_path)
    gene_df = pd.read_csv(gn_path)
    gt, pw_to_genes = load_gt(cohort)
    if gt.empty:
        print(f"  [{cohort}] SKIP — no GT rows")
        return None
    return {
        "cohort": cohort,
        "pathway_recall_at5":  recall_at_k(pw_df, gt, 5),
        "gene_precision_at30": precision_at_30(gene_df, gt, pw_to_genes),
        "n_signal_cts": int(gt["celltype"].nunique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+",
                    default=list(COHORT_GT),
                    help="Simulated cohorts to score (default: all 6).")
    ap.add_argument("--out", default=str(REPO / "results" / "ground_truth_recovery.csv"))
    args = ap.parse_args()

    rows = []
    for c in args.cohorts:
        if c not in COHORT_GT:
            print(f"  [{c}] SKIP — not a simulated cohort")
            continue
        row = score_cohort(c)
        if row:
            rows.append(row)
            print(f"  {c:<12}  recall@5={row['pathway_recall_at5']:.4f}  "
                  f"precision@30={row['gene_precision_at30']:.4f}  "
                  f"(n_signal_cts={row['n_signal_cts']})")
    if not rows:
        sys.exit("No results.")
    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    try: shown = out_path.relative_to(REPO)
    except ValueError: shown = out_path
    print(f"\nSaved → {shown}")


if __name__ == "__main__":
    main()
