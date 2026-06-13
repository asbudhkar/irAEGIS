#!/usr/bin/env python3
# Compute the cross-cohort shared gene vocabulary from per-cohort h5ad files.

from __future__ import annotations
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def cohort_qc_genes(h5ad_path: Path, min_cells: int) -> set[str]:
    # Return the set of genes detected in >= min_cells cells of this cohort.
    a = ad.read_h5ad(h5ad_path)
    X = a.X
    if sp.issparse(X):
        n_detected = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        n_detected = (np.asarray(X) > 0).sum(axis=0)
    keep_mask = n_detected >= min_cells
    return set(np.asarray(a.var_names)[keep_mask].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ads", nargs="+", required=True,
                    help="Cohort h5ad paths (≥ 2 for a meaningful intersection)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output gene-list text file (one gene per line)")
    ap.add_argument("--min-cells", type=int, default=10,
                    help="Drop genes detected in fewer than this many cells "
                         "per cohort before intersecting (default: 10)")
    args = ap.parse_args()

    shared: set[str] | None = None
    for p in args.h5ads:
        p = Path(p)
        if not p.exists():
            raise SystemExit(f"h5ad not found: {p}")
        genes = cohort_qc_genes(p, args.min_cells)
        print(f"  {p.name:50s}  {len(genes):>6} genes (≥{args.min_cells} cells)")
        shared = genes if shared is None else shared & genes

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for g in sorted(shared):
            f.write(g + "\n")
    try: shown = args.out.resolve().relative_to(Path.cwd())
    except ValueError: shown = args.out
    print(f"\nIntersection: {len(shared)} genes → {shown}")


if __name__ == "__main__":
    main()
