#!/usr/bin/env python3

#irAEGIS model training
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.config import (DEVICE, RANDOM_STATE, RESULTS_IRAEGIS,
                          cohort_h5ad, SIM_PRIOR_NPZ)
from utils.profiler import start as prof_start, stop as prof_stop
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.model_utils import PathwayAE
from models.iraegis.train_utils import (
    train_ae, precompute_embeddings,
    train_cell_irae_per_ct,
    train_h_concat_gated_concat_en,
    extract_patient_w_eff,
    AE_LATENT_DIM, AE_DROPOUT, AE_N_EPOCHS,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True,
                    help="dataset_id to train on, e.g. GSE216329_integrated_pre_ici")
    ap.add_argument("--ae-only",   action="store_true",
                    help="Run AE only")
    ap.add_argument("--skip-ae",   action="store_true",
                    help="Load saved AE from out_dir, skip AE training")
    ap.add_argument("--norm", type=str, default="ctbn", choices=["bn", "ln", "ctbn"],
                    help="Pathway normalization: ctbn (CT-conditional BN, default), "
                         "bn (BatchNorm), or ln (LayerNorm)")
    ap.add_argument("--act", type=str, default="gelu", choices=["gelu", "relu"],
                    help="Activation: gelu (default) or relu")
    ap.add_argument("--ae-epochs",  type=int, default=AE_N_EPOCHS)
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility / stability runs. "
                         "Saves to seed_<N>/ subdirectory. "
                         "AE is reused from the base directory (--skip-ae implied).")
    ap.add_argument("--sim", action="store_true",
                    help="Use simulated data (processed_h5ad/simulated_data.h5ad)")
    ap.add_argument("--h5ad", default=None,
                    help="Override h5ad path (takes precedence over --sim default)")
    ap.add_argument("--prior", default=None,
                    help="Override pathway prior NPZ path")
    ap.add_argument("--prior-genes-only", action="store_true",
                    help="Drop genes with all-zero mask rows (17k → ~3.7k)")
    ap.add_argument("--gene-list",
                    default="datasets/processed_h5ad/shared_genes.txt",
                    help="Restrict training to genes listed in this text file "
                         "(one gene per line). Defaults to the cross-cohort "
                         "shared vocabulary used in this study. Pass empty "
                         "string to use the cohort's native vocabulary.")
    ap.add_argument("--ae-ct-aux-weight", type=float, default=None,
                    help="Override AE auxiliary CT classification weight α "
                         "(default 0.3). Set 0 for ablation.")
    ap.add_argument("--ae-decorr-weight", type=float, default=None,
                    help="Override AE pathway decorrelation weight β "
                         "(default 0.1). Set 0 for ablation.")
    ap.add_argument("--ablation-tag", type=str, default=None,
                    help="Subdirectory tag for ablation runs, e.g. 'noaux' "
                         "(produces results/iraegis/<cohort>/ablation_noaux/)")
    args = ap.parse_args()

    method = "iraegis_train"
    if args.seed is not None:
        method = None  # don't profile seed runs
    else:
        prof_start(method, args.cohort)

    # Set seed
    global RANDOM_STATE
    import random as _stdlib_random
    import utils.config as _cfg
    if args.seed is not None:
        RANDOM_STATE = args.seed
        _cfg.RANDOM_STATE = args.seed
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _stdlib_random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    _mps_active = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    if _mps_active:
        torch.mps.manual_seed(RANDOM_STATE)
    
    # MPS warning
    torch.use_deterministic_algorithms(True, warn_only=_mps_active)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.seed is not None:
        print(f"[Seed] RANDOM_STATE = {args.seed}")
        results_root = Path(os.environ.get("IRAEGIS_RESULTS_DIR",
                                           REPO_ROOT / "results" / "iraegis"))
        base_dir = results_root / args.cohort
        if (base_dir / "ae_encoder.pt").exists() and not args.skip_ae:
            args.skip_ae = True
            print(f"[Seed] AE found at {base_dir / 'ae_encoder.pt'} — implying --skip-ae")

    # h5ad and prior paths.
    # datasets/processed_h5ad/<cohort_id>.h5ad
    def _rel(p):
        try: return Path(p).resolve().relative_to(REPO_ROOT)
        except ValueError: return Path(p)

    if args.h5ad:
        h5ad_path = Path(args.h5ad)
        print(f"[Override] h5ad: {_rel(h5ad_path)}")
    else:
        h5ad_path = None
        print(f"[Default] h5ad: {_rel(cohort_h5ad(args.cohort))}")

    if args.prior:
        prior_path = Path(args.prior)
        print(f"[Override] prior: {_rel(prior_path)}")
    elif args.sim:
        prior_path = SIM_PRIOR_NPZ
    else:
        prior_path = None

    # Build output directory based on variant flags
    variant_suffix = ""
    if args.norm != "ctbn" or args.act != "gelu":
        variant_suffix = f"{args.norm}_{args.act}"

    BASE_DIR = RESULTS_IRAEGIS / args.cohort
    if variant_suffix:
        BASE_DIR = BASE_DIR / variant_suffix
    if args.ablation_tag:
        BASE_DIR = BASE_DIR / f"ablation_{args.ablation_tag}"

    if args.seed is not None:
        OUT_DIR = BASE_DIR / f"seed_{args.seed}"
    else:
        OUT_DIR = BASE_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}")
    print(f"Device: {DEVICE}")

    # Load data
    (X, obs, gene_names, ct_groups, ct_ids,
     pat_ids, pat_labels, prior) = load_cohort_data(
        args.cohort, h5ad_path=h5ad_path, prior_path=prior_path,
        prior_genes_only=args.prior_genes_only,
        gene_list_path=args.gene_list if args.gene_list else None,
        split_ct_groups=["T_cells", "Monocytes", "Dendritic"])

    n_genes    = X.shape[1]
    n_pathways = prior["mask"].shape[1]
    n_ct       = len(ct_groups)
    mask_t     = torch.tensor(prior["mask"], dtype=torch.float32)
    pw_names   = prior["pathway_names"]

    unique_pats = sorted(pat_labels.keys())
    n_patients  = len(unique_pats)

    print(f"\nGenes={n_genes}  Pathways={n_pathways}  CTs={n_ct}  "
          f"Patients={n_patients}")

    # AE
    ae = PathwayAE(n_genes, n_pathways, mask_t, AE_LATENT_DIM, AE_DROPOUT,
                   norm=args.norm, act=args.act, n_ct=n_ct)
    print(f"  Architecture: norm={args.norm}, act={args.act}")
    ae_path = next(
        (d / "ae_encoder.pt" for d in (OUT_DIR, BASE_DIR)
         if (d / "ae_encoder.pt").exists()),
        OUT_DIR / "ae_encoder.pt")

    _load_ae = False
    if args.skip_ae and ae_path.exists():
        state = torch.load(ae_path, map_location=DEVICE, weights_only=True)
        remap = {}
        for k in list(state.keys()):
            for old_prefix in ("pw_bn.", "pw_ln."):
                if k.startswith(old_prefix):
                    remap[k] = "pw_norm." + k[len(old_prefix):]
        for old_k, new_k in remap.items():
            state[new_k] = state.pop(old_k)
        if state.get("mask", state.get("pw_weight", next(iter(state.values())))).shape[0] == n_genes:
            print("\n Loading saved AE ...")
            state = {k: v for k, v in state.items() if not k.startswith("ct_head.")}
            ae.load_state_dict(state, strict=False)
            ae.to(DEVICE)
            _load_ae = True
        else:
            print(f"\nSaved AE has wrong gene count "
                  f"({state['mask'].shape[0]} vs {n_genes}) — retraining")
    if not _load_ae:
        print(f"\nTraining AE on {len(X):,} cells ...")
        if args.ae_ct_aux_weight is not None or args.ae_decorr_weight is not None:
            print(f"  [ablation] α(ct_aux)={args.ae_ct_aux_weight}  "
                  f"β(decorr)={args.ae_decorr_weight}")
        ae_hist = train_ae(
            ae, X, n_epochs=args.ae_epochs, ct_ids=ct_ids,
            ct_aux_weight=args.ae_ct_aux_weight,
            decorr_weight=args.ae_decorr_weight,
        )
        torch.save(ae.state_dict(), BASE_DIR / "ae_encoder.pt")  # always save to base
        pd.DataFrame(ae_hist).to_csv(BASE_DIR / "ae_history.csv", index=False)
        del ae_hist        # release training-loss list
        print(f"  AE saved → {BASE_DIR / 'ae_encoder.pt'}")

    np.save(BASE_DIR / "gene_names.npy", np.array(gene_names, dtype=str))

    # Persist the training-time ct_groups list for downstream tasks
    with open(BASE_DIR / "ct_groups.json", "w") as f:
        json.dump({"ct_groups": list(ct_groups), "n_ct": int(n_ct)}, f, indent=2)

    # Pre-compute h
    print("\n[Pre-compute] Encoding all cells ...")
    h_path = next(
        (d / "h_cells.npy" for d in (OUT_DIR, BASE_DIR)
         if (d / "h_cells.npy").exists()),
        OUT_DIR / "h_cells.npy")
    z_path = next(
        (d / "z_cells.npy" for d in (OUT_DIR, BASE_DIR)
         if (d / "z_cells.npy").exists()),
        OUT_DIR / "z_cells.npy")

    _load_hz = False
    n_cells = len(X)
    if args.skip_ae and h_path.exists() and z_path.exists():
        h = np.load(h_path)
        z = np.load(z_path)
        if h.shape[1] == n_pathways and h.shape[0] == n_cells:
            print(f"  Loaded h {h.shape}, z {z.shape}")
            _load_hz = True
        else:
            print(f"  Cached h shape {h.shape} doesn't match "
                  f"({n_cells} cells, {n_pathways} pathways) — re-encoding")
    if not _load_hz:
        h, z = precompute_embeddings(ae, X, obs, BASE_DIR, ct_ids=ct_ids)

    if args.ae_only:
        print("--ae-only: stopping after AE + precompute.")
        return

    # Free the dense expression matrix
    del X
    import gc; gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Per-CT irAE classifiers
    print(f"\nPer cell-type irAE classifiers (patient-level CV)")
    ct_models, ct_cv_summaries, irae_oof = train_cell_irae_per_ct(
        h, pat_ids, ct_ids, pat_labels, ct_groups, n_pathways)

    # Save CV results
    with open(OUT_DIR / "irae_per_ct_summary.json", "w") as f:
        json.dump(ct_cv_summaries, f, indent=2)
    np.save(OUT_DIR / "irae_cell_oof_probs.npy", irae_oof)

    irae_oof_df = pd.DataFrame({
        "cell_idx": np.arange(len(irae_oof)),
        "patient": pat_ids,
        "ct_id": ct_ids,
        "ct_name": [ct_groups[c] for c in ct_ids],
        "label": [pat_labels[p] for p in pat_ids],
        "oof_prob": irae_oof,
    })
    irae_oof_df.to_csv(OUT_DIR / "irae_cell_oof_predictions.csv", index=False)

    # Patient-level irAEGIS classifier
    print(f"\nPatient-level results.")
    gated_concat_en_result = train_h_concat_gated_concat_en(
        h, pat_ids, ct_ids, pat_labels, ct_groups)
    if gated_concat_en_result:
        with open(OUT_DIR / "h_concat_gated_concat_en_summary.json", "w") as f:
            json.dump(gated_concat_en_result["cv_summary"], f, indent=2)

    # Final AUC and AUPRC results
    if gated_concat_en_result:
        s = gated_concat_en_result["cv_summary"]
        auc   = s.get("mean_auc", float("nan"))
        auprc = s.get("mean_fold_ap", float("nan"))
        print(f"\n{'='*40}")
        print(f"  Patient AUC   = {auc:.4f}")
        print(f"  Patient AUPRC = {auprc:.4f}")
        print(f"{'='*40}")

    # Patient W_eff extraction for per-patient explanation (explain_patient.py)
    print(f"\n[W_eff] Refit gated stacking on all patients for per-patient attribution ...")
    w_eff_payload = extract_patient_w_eff(
        h, pat_ids, ct_ids, pat_labels, ct_groups)
    w_eff_dir = REPO_ROOT / "results" / "iraegis_oof" / args.cohort
    w_eff_dir.mkdir(parents=True, exist_ok=True)
    np.savez(w_eff_dir / "patient_classifier_W_eff.npz", **w_eff_payload)
    print(f"  saved {len(w_eff_payload['selected_cts'])} selected CTs "
          f"→ {w_eff_dir / 'patient_classifier_W_eff.npz'}")

    if method is not None:
        prof_stop()

    # Save per-CT model checkpoints for explainability
    ct_model_dir = OUT_DIR / "irae_per_ct"
    ct_model_dir.mkdir(exist_ok=True)
    for ct_name, model in ct_models.items():
        safe_name = ct_name.replace(" ", "_").replace("/", "_")
        torch.save(model.state_dict(), ct_model_dir / f"{safe_name}.pt")

    print(f"\nDone.  Results → {OUT_DIR}")

if __name__ == "__main__":
    main()
