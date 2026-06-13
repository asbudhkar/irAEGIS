#!/usr/bin/env python3
# Per-patient explainability.

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "per_patient_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_top25(h_cells, mask):
    cells = h_cells[mask]
    if cells.shape[0] < 4:
        return cells.mean(axis=0) if len(cells) else np.zeros(h_cells.shape[1])
    norms = np.linalg.norm(cells, axis=1)
    cutoff = np.percentile(norms, 75)
    top = cells[norms >= cutoff]
    return top.mean(axis=0) if len(top) else cells.mean(axis=0)


# Same cell type grouping used at training time
REGROUP = {
    "Naive B cells": "B_cells", "Plasma B cells": "B_cells",
    "Pre-B cells": "B_cells", "Pro-B cells": "B_cells",
    "Natural killer  cells": "NK_cells", "Natural killer cells": "NK_cells",
    "Progenitor cells": "HSC_Prog", "HSC/MPP cells": "HSC_Prog",
    "Erythroid-like and erythroid precursor cells": "Erythroid",
    "ISG expressing immune cells": "ISG_immune",
}

def load_artifacts(cohort):
    train_dir = REPO / "results" / "iraegis" / cohort
    base      = REPO / "results" / "iraegis_oof" / cohort
    h = np.load(train_dir / "h_cells.npy")
    meta = pd.read_csv(train_dir / "cell_meta.csv")
    meta["final_celltype"] = meta["final_celltype"].replace(REGROUP)
    cls = np.load(base / "patient_classifier_W_eff.npz", allow_pickle=True)
    ct_groups_all = [str(c) for c in cls["ct_groups_all"]]
    selected_ct_names = [str(c) for c in cls["selected_cts"]]
    W_eff = {selected_ct_names[i]: cls[f"W_eff_{i}"] for i in range(len(selected_ct_names))}
    mu_train = {selected_ct_names[i]: cls[f"mu_train_{i}"] for i in range(len(selected_ct_names))}
    w_LR = cls["w_LR"]; b_LR = float(cls["b_LR"])
    ct_aucs = {ct_groups_all[i]: float(cls["ct_aucs"][i]) for i in range(len(ct_groups_all))}
    auc_gate = float(cls["auc_gate"])
    prior = np.load(REPO / "datasets/resources/pathway_prior.npz",
                     allow_pickle=True)
    pw_names = [str(p) for p in prior["pathway_names"]]
    oof_probs = None; oof_labels = None; pat_order = None
    for nm in ["iraegis_oof_probs", "patient_oof_probs"]:
        p = base / f"{nm}.npy"
        if p.exists(): oof_probs = np.load(p); break
    for nm in ["iraegis_oof_labels", "patient_oof_labels"]:
        p = base / f"{nm}.npy"
        if p.exists(): oof_labels = np.load(p); break
    if (base / "patient_order.npy").exists():
        pat_order = np.load(base / "patient_order.npy", allow_pickle=True)
    return dict(h=h, meta=meta, ct_groups_all=ct_groups_all,
                selected_ct_names=selected_ct_names, W_eff=W_eff,
                mu_train=mu_train, w_LR=w_LR, b_LR=b_LR,
                ct_aucs=ct_aucs, auc_gate=auc_gate, pw_names=pw_names,
                oof_probs=oof_probs, oof_labels=oof_labels,
                pat_order=pat_order)


def patient_explain(art, patient):
    meta = art["meta"]; h = art["h"]
    selected = art["selected_ct_names"]
    pat_mask = (meta.patient_id.astype(str) == str(patient)).values
    if pat_mask.sum() == 0:
        raise SystemExit(f"patient {patient} not in cohort")
    per_ct_logit = {}
    per_ct_pathway_contrib = {}
    for ct in selected:
        ct_mask = (meta.final_celltype == ct).values & pat_mask
        h_bar = aggregate_top25(h, ct_mask)
        W = art["W_eff"][ct]; mu = art["mu_train"][ct]
        h_centered = h_bar - mu
        ct_logit = float(np.dot(W, h_centered))
        per_ct_logit[ct] = {
            "n_cells_total": int(ct_mask.sum()),
            "h_bar": h_bar, "h_centered": h_centered,
            "per_CT_logit": ct_logit,
        }
        per_ct_pathway_contrib[ct] = W * h_centered
    ct_logits_vec = np.array([per_ct_logit[ct]["per_CT_logit"] for ct in selected])
    w_LR = art["w_LR"]; b_LR = art["b_LR"]
    patient_logit = float(b_LR + np.dot(w_LR, ct_logits_vec))
    patient_prob = 1.0 / (1.0 + np.exp(-patient_logit))
    ct_contrib = {ct: float(w_LR[i] * ct_logits_vec[i]) for i, ct in enumerate(selected)}
    pathway_contrib = {ct: w_LR[i] * per_ct_pathway_contrib[ct]
                        for i, ct in enumerate(selected)}
    
    oof_prob = None
    if art.get("oof_probs") is not None and art.get("pat_order") is not None:
        po = [str(x) for x in art["pat_order"]]
        if str(patient) in po:
            i = po.index(str(patient))
            oof_prob = float(art["oof_probs"][i])
    true_label = None
    if pat_mask.any():
        labs = meta.loc[pat_mask, "irAE_status"].astype(str).unique()
        if len(labs) == 1: true_label = labs[0]
    return dict(patient=patient, patient_logit=patient_logit,
                patient_prob=patient_prob, oof_prob=oof_prob, true_label=true_label,
                per_ct_logit=per_ct_logit, ct_contrib=ct_contrib,
                pathway_contrib=pathway_contrib)


def render_report(art, explain, cohort_short, out_pdf, out_png):
    pw_names = art["pw_names"]
    selected = art["selected_ct_names"]
    pat = explain["patient"]
    
    oof_prob = explain.get("oof_prob")
    in_sample_prob = explain["patient_prob"]
    prob = oof_prob if oof_prob is not None else in_sample_prob
    logit = explain["patient_logit"]; true_label = explain["true_label"]
    ct_contrib = explain["ct_contrib"]
    pathway_contrib = explain["pathway_contrib"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.4, 1.4], hspace=0.45, wspace=0.32)

    # (A) Prediction summary
    axA = fig.add_subplot(gs[0, 0])
    cohort_oof = art["oof_probs"]
    if cohort_oof is not None:
        cohort_oof = cohort_oof[~np.isnan(cohort_oof)]
        axA.hist(cohort_oof, bins=15, color="#bbbbbb", alpha=0.5,
                  edgecolor="white", label=f"Cohort OOF (n={len(cohort_oof)})")
    axA.axvline(prob, color="#d62728", linewidth=2.5,
                  label=f"This patient: {prob:.3f}")
    axA.axvline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.6,
                  label="Decision threshold (0.5)")
    axA.set_xlim(0, 1)
    axA.set_xlabel("Predicted irAE probability")
    axA.set_ylabel("Count")
    label_str = f"true label: {true_label}" if true_label else "true label: unknown"
    prob_kind = "OOF" if oof_prob is not None else "in-sample"
    axA.set_title(f"(A) Prediction summary\n"
                    f"patient: {pat}    logit: {logit:+.3f}    prob ({prob_kind}): {prob:.3f}\n"
                    f"{label_str}",
                    fontsize=10)
    axA.legend(loc="best", fontsize=8, frameon=True)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)

    # (B) Per-CT contributions
    axB = fig.add_subplot(gs[0, 1])
    cts_sorted = sorted(ct_contrib.keys(), key=lambda c: abs(ct_contrib[c]), reverse=True)
    vals = [ct_contrib[c] for c in cts_sorted]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    y = np.arange(len(cts_sorted))
    axB.barh(y, vals, color=colors, edgecolor="white")
    axB.set_yticks(y); axB.set_yticklabels(cts_sorted, fontsize=9)
    axB.axvline(0, color="grey", linewidth=0.5)
    axB.invert_yaxis()
    axB.set_xlabel("Contribution to patient logit\n(w_LR × per-CT-logit)")
    axB.set_title(f"(B) Per-CT contributions (gated, frozen w_LR)\n"
                    f"Red = pushes toward irAE-Yes, Blue = pushes toward No\n"
                    f"Σ = {sum(vals):+.3f}  (patient_logit = {logit:+.3f} = Σ + b_LR)",
                    fontsize=10)
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)

    # (C) Top pathways within top-2 CTs
    axC = fig.add_subplot(gs[1, :])
    top2_cts = cts_sorted[:2]
    n_top_pw = 10
    bar_data = []
    for ct in top2_cts:
        pw_vec = pathway_contrib[ct]
        order = np.argsort(-np.abs(pw_vec))[:n_top_pw]
        for k in order:
            bar_data.append({"CT": ct, "pathway": pw_names[k].replace("HALLMARK_", ""),
                              "contribution": float(pw_vec[k])})
    bar_df = pd.DataFrame(bar_data)
    bar_df["label"] = bar_df.apply(lambda r: f"{r['CT'][:18]}  ·  {r['pathway']}", axis=1)
    bar_df = bar_df.sort_values("contribution")
    y2 = np.arange(len(bar_df))
    colors2 = ["#d62728" if v > 0 else "#1f77b4" for v in bar_df["contribution"]]
    axC.barh(y2, bar_df["contribution"], color=colors2, edgecolor="white")
    axC.set_yticks(y2); axC.set_yticklabels(bar_df["label"], fontsize=8)
    axC.axvline(0, color="grey", linewidth=0.5)
    axC.set_xlabel("w_LR × W_eff[k] × (h̄_top25[k] − mu_train[k])  =  per-pathway contribution to patient logit")
    axC.set_title(f"(C) Top-{n_top_pw} pathway contributions per top-2 CT\n"
                    "Red = pushes toward irAE-Yes  •  Blue = pushes toward No\n"
                    "Direction is reliable: classifier weights are frozen at inference",
                    fontsize=10)
    for sp in ("top", "right"): axC.spines[sp].set_visible(False)

    # (D) Patient pathway deviation in the TOP contributing CT, using the SAME
    # top pathways selected in Panel C - i.e., the pathways that drove the
    # model's prediction for this patient. Shows their deviation from training
    # mean
    axD = fig.add_subplot(gs[2, 0])
    
    meta = art["meta"]; h = art["h"]
    other_pats = [p for p in meta.patient_id.astype(str).unique() if str(p) != str(pat)]

    def n_pats_with_cells(ct, thr=5):
        m_ct = (meta.final_celltype == ct).values
        return sum(((meta.patient_id.astype(str) == str(p)).values & m_ct).sum() >= thr
                    for p in other_pats)

    n_cells_per_ct = {ct: explain["per_ct_logit"][ct]["n_cells_total"] for ct in selected}
    top_ct = None
    for cand in cts_sorted:  # ordered by |ct_contrib|
        if n_cells_per_ct.get(cand, 0) >= 10 and n_pats_with_cells(cand) >= 3:
            top_ct = cand; break
    if top_ct is None:
        for cand in cts_sorted:
            if n_cells_per_ct.get(cand, 0) >= 1:
                top_ct = cand; break
        if top_ct is None:
            top_ct = max(selected, key=lambda c: n_pats_with_cells(c))

    pat_h = explain["per_ct_logit"][top_ct]["h_bar"]
    mu = art["mu_train"][top_ct]
    pat_dev = pat_h - mu

    pw_contrib = pathway_contrib[top_ct]
    order = np.argsort(-np.abs(pw_contrib))[:10]
    labels = [pw_names[k].replace("HALLMARK_", "")[:30] for k in order]
    pat_vals = pat_dev[order]

    cohort_dev = []
    for p in other_pats:
        m = (meta.patient_id.astype(str) == str(p)).values & (meta.final_celltype == top_ct).values
        if m.sum() < 5: continue
        cohort_dev.append(aggregate_top25(h, m) - mu)
    cohort_dev = np.stack(cohort_dev) if cohort_dev else None

    if cohort_dev is not None and len(cohort_dev) >= 3:
        cohort_med = np.median(cohort_dev[:, order], axis=0)
        cohort_q1 = np.percentile(cohort_dev[:, order], 25, axis=0)
        cohort_q3 = np.percentile(cohort_dev[:, order], 75, axis=0)
        cohort_min = np.min(cohort_dev[:, order], axis=0)
        cohort_max = np.max(cohort_dev[:, order], axis=0)
        cohort_label = f"Other cohort (median, IQR, range; n={len(cohort_dev)})"
        cohort_ok = True
    else:
        cohort_med = np.zeros(len(order))
        cohort_q1 = cohort_q3 = cohort_min = cohort_max = cohort_med
        cohort_label = "Cohort comparison unavailable"
        cohort_ok = False

    y3 = np.arange(len(labels))
    axD.axvline(0, color="black", linestyle="-", linewidth=0.7, alpha=0.5)
    if cohort_ok:
        for yi, lo, hi in zip(y3, cohort_min, cohort_max):
            axD.plot([lo, hi], [yi, yi], color="#444", linewidth=1.0,
                       alpha=0.45, zorder=2)
        
        for yi, q1, q3 in zip(y3, cohort_q1, cohort_q3):
            axD.plot([q1, q3], [yi, yi], color="#444", linewidth=4.0,
                       alpha=0.75, solid_capstyle="round", zorder=3)
        
        axD.scatter(cohort_med, y3, color="#222", s=55, marker="D",
                      edgecolor="white", linewidth=0.8, zorder=4,
                      label=cohort_label)
    axD.scatter(pat_vals, y3, color="#d62728", s=120, zorder=10,
                  edgecolor="white", linewidth=1.2,
                  label="This patient")
    axD.set_yticks(y3); axD.set_yticklabels(labels, fontsize=8)
    axD.set_xlabel("Deviation from training mean (h̄ − μ_train)\n"
                     "→ positive = elevated above baseline, negative = depressed")
    n_cells = explain["per_ct_logit"][top_ct]["n_cells_total"]
    axD.set_title(f"(D) Patient deviation on Panel-C top-pathways in {top_ct}\n"
                    f"(top contributing CT, n_cells = {n_cells})",
                    fontsize=10)
    axD.legend(loc="best", fontsize=8)
    axD.invert_yaxis()
    for sp in ("top", "right"): axD.spines[sp].set_visible(False)

    axE = fig.add_subplot(gs[2, 1])
    axE.axis("off")
    rows = []
    for ct in selected:
        n = explain["per_ct_logit"][ct]["n_cells_total"]
        auc = art["ct_aucs"].get(ct, np.nan)
        ct_logit = explain["per_ct_logit"][ct]["per_CT_logit"]
        contrib = ct_contrib.get(ct, 0.0)
        rows.append([ct[:22], f"{n}", f"{auc:.2f}", f"{ct_logit:+.3f}", f"{contrib:+.3f}"])
    col_labels = ["CT (gated)", "n cells\n(patient)", "Cell-OOF\nAUC",
                   "per-CT\nlogit", "Contribution\nto patient logit"]
    tbl = axE.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.5)
    axE.set_title(f"(E) Gated CTs (AUC ≥ {art['auc_gate']:.2f}) — model-trust context\n"
                    f"patient_logit = b_LR + Σ_i w_LR[i] × per_CT_logit[i]",
                    fontsize=10, pad=20)

    fig.suptitle(f"irAEGIS per-patient explainability report\n"
                  f"{cohort_short}  •  patient = {pat}",
                  fontsize=12, fontweight="bold", y=0.995)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    out_svg = out_pdf.with_suffix(".svg")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--patient", required=False, default=None,
                    help="If omitted, generates reports for ALL patients in cohort")
    ap.add_argument("--short", default=None)
    args = ap.parse_args()

    art = load_artifacts(args.cohort)
    short = args.short or args.cohort.replace("_pre_ici","").replace("_integrated","")

    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted(art["meta"].patient_id.astype(str).unique().tolist())

    for pat in patients:
        if str(pat) not in set(art["meta"].patient_id.astype(str)):
            print(f"  [skip] {pat} not in cohort")
            continue
        try:
            explain = patient_explain(art, pat)
            pat_clean = str(pat).replace("/","_").replace(" ","_")
            out_pdf = OUT_DIR / f"explain_{short}_{pat_clean}.pdf"
            out_png = OUT_DIR / f"explain_{short}_{pat_clean}.png"
            render_report(art, explain, short, out_pdf, out_png)
            shown = explain['oof_prob'] if explain['oof_prob'] is not None else explain['patient_prob']
            kind = "OOF" if explain['oof_prob'] is not None else "in-sample"
            print(f"  Saved: {out_pdf.name}  (prob[{kind}]={shown:.3f})")
        except Exception as e:
            print(f"  [error] {pat}: {e}")


if __name__ == "__main__":
    main()
