#!/usr/bin/env python3
# Class contrastivity 

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def class_contrastivity(pw_path: Path, z_thresh: float = 1.0) -> dict:
    df = pd.read_csv(pw_path)
    if df.empty:
        return {"n_top_total": 0, "pw_yes_driven": 0, "pw_no_driven": 0,
                "mean_abs_diff": np.nan, "frac_yes": np.nan,
                "yes_no_jaccard": np.nan}

    df = df.assign(_signed=df["h_diff"], _abs=df["h_diff"].abs())

    yes_count = no_count = total = 0
    abs_diffs = []
    yes_set_pooled: set = set()
    no_set_pooled:  set = set()

    for _, sub in df.groupby("celltype"):
        thresh = sub["_abs"].mean() + z_thresh * sub["_abs"].std()
        top = sub[sub["_abs"] > thresh]
        if top.empty:
            top = sub.sort_values("_abs", ascending=False).head(1)
        yes_count += int((top["_signed"] > 0).sum())
        no_count  += int((top["_signed"] < 0).sum())
        total     += len(top)
        abs_diffs.extend(top["_abs"].tolist())

        yes_set_pooled |= set(top.loc[top["_signed"] > 0, "pathway"])
        no_set_pooled  |= set(top.loc[top["_signed"] < 0, "pathway"])

    yes_no_jaccard = _jaccard(yes_set_pooled, no_set_pooled)

    return {
        "n_top_total":      total,
        "pw_yes_driven":    yes_count,
        "pw_no_driven":     no_count,
        "mean_abs_diff":    float(np.mean(abs_diffs)) if abs_diffs else np.nan,
        "frac_yes":         yes_count / max(total, 1),
        "yes_no_jaccard":   yes_no_jaccard,
    }


def collect(iraegis_dir: Path) -> pd.DataFrame:
    rows = []
    for cohort_dir in sorted(iraegis_dir.iterdir()):
        if not cohort_dir.is_dir():
            continue
        pw = cohort_dir / "cell_explainability" / "cell_pathway_attribution.csv"
        if not pw.exists():
            continue

        cc = class_contrastivity(pw)
        rows.append({
            "cohort":            cohort_dir.name,
            "n_top_pw_total":    cc["n_top_total"],
            "pw_yes_driven":     cc["pw_yes_driven"],
            "pw_no_driven":      cc["pw_no_driven"],
            "frac_yes":          round(cc["frac_yes"], 3) if not np.isnan(cc["frac_yes"]) else "",
            "mean_abs_diff":     round(cc["mean_abs_diff"], 4) if not np.isnan(cc["mean_abs_diff"]) else "",
            # low → Yes/No use DIFFERENT pathways (truly contrastive)
            "yes_no_pw_jaccard": round(cc["yes_no_jaccard"], 3) if not np.isnan(cc["yes_no_jaccard"]) else "",
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iraegis-dir", default="results/iraegis")
    ap.add_argument("--out-dir",     default="results")
    args = ap.parse_args()

    iraegis = Path(args.iraegis_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect(iraegis)
    if df.empty:
        print("No cohorts with cell_explainability/ outputs found.")
        return

    out_csv = out_dir / "explainability_validation.csv"
    df.to_csv(out_csv, index=False)
    try: shown = out_csv.resolve().relative_to(Path.cwd())
    except ValueError: shown = out_csv
    print(f"\nWrote {len(df)} rows → {shown}")


if __name__ == "__main__":
    main()
