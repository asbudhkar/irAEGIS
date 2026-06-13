"""
Dynamic broad cell type grouping inferred from data.

Instead of hardcoding fine-grained cell type labels, this module uses
keyword matching on whatever `final_celltype` labels exist in the dataset,
then filters to groups that have enough patients and cells to be analyzable.
"""

from collections import OrderedDict

# Cells that must be excluded from training entirely.
EXCLUDE_LABEL = "__EXCLUDE__"

# Cell type labels that are not biologically meaningful for downstream
_EXCLUDE_LABELS = {
    "Unknown",
}

# Priority-ordered keyword rules. First match wins.
_KEYWORD_RULES = [
    # Adaptive immune — T cells (all subtypes)
    ("T_cells",    ["cd4", "cd8", "t cell", "t-cell", "t_cell",
                    "effector t", "memory t", "naive t",
                    "regulatory t", "treg", "gamma delta", "gd-t", "nkt"]),
    # B cells and plasma cells
    ("B_cells",    ["b cell", "b-cell", "b_cell", "naive b", "memory b",
                    "pro-b", "pre-b", "plasma b", "plasmablast",
                    "plasma cell"]),
    # NK cells
    ("NK_cells",   ["natural killer", "nk cell", "nk-cell", "nk_cell"]),
    # Monocytes (classical + non-classical)
    ("Monocytes",  ["monocyte"]),
    # Dendritic cells (myeloid + plasmacytoid)
    ("Dendritic",  ["dendritic"]),
    # Neutrophils
    ("Neutrophils",["neutrophil"]),
    # Macrophages
    ("Macrophages",["macrophage"]),
    # Erythroid lineage
    ("Erythroid",  ["erythroid"]),
    # Hematopoietic stem / progenitor cells
    ("HSC_Prog",   ["hsc", "mpp", "progenitor", "hematopoietic stem",
                    "multipotent"]),
    # Platelets / megakaryocytes
    ("Platelets",  ["platelet", "thrombocyte", "megakaryocyte"]),
    # ISG-expressing immune cells (interferon-stimulated gene signature)
    ("ISG_immune", ["isg expressing", "isg-expressing", "isg immune",
                    "interferon stimulated"]),
]

_FALLBACK_GROUP = "Other"


def _assign_group(label: str) -> str:
    # Return broad group name for a fine-grained cell type label.
    if label in _EXCLUDE_LABELS:
        return EXCLUDE_LABEL
    lower = label.lower()
    for group, keywords in _KEYWORD_RULES:
        if any(kw in lower for kw in keywords):
            return group
    return _FALLBACK_GROUP

def infer_celltype_groups(
    obs_df,
    min_patients: int = 10,
    min_cells_per_patient: int = 10,
    verbose: bool = True,
    split_groups: list | None = None,
):
    # Infer broad cell type groups from obs_df['final_celltype'].

    all_labels = obs_df["final_celltype"].unique()

    raw_mapping = {label: _assign_group(label) for label in all_labels}

    raw_groups: dict = {}
    for label, group in raw_mapping.items():
        if group == EXCLUDE_LABEL:
            continue
        raw_groups.setdefault(group, []).append(label)

    obs_valid = obs_df[obs_df["final_celltype"].map(raw_mapping) != EXCLUDE_LABEL]

    if split_groups:
        split_set = set(split_groups)
        group_order = [r[0] for r in _KEYWORD_RULES] + [_FALLBACK_GROUP]
        viable_groups = OrderedDict()
        for group in group_order:
            labels = raw_groups.get(group, [])
            if not labels:
                continue
            if group in split_set:
                for label in sorted(labels):
                    sub = obs_valid[obs_valid["final_celltype"] == label]
                    pat_counts = sub.groupby("patient_id").size()
                    eligible = pat_counts[pat_counts >= min_cells_per_patient]
                    if len(eligible) >= min_patients:
                        viable_groups[label] = [label]
            else:
                ct_mask = obs_valid["final_celltype"].isin(labels)
                sub = obs_valid[ct_mask]
                pat_counts = sub.groupby("patient_id").size()
                eligible = pat_counts[pat_counts >= min_cells_per_patient]
                if len(eligible) >= min_patients:
                    viable_groups[group] = sorted(labels)

        viable_label_set = {
            label
            for labels in viable_groups.values()
            for label in labels
        }
        final_mapping = {}
        for label in all_labels:
            raw = raw_mapping[label]
            if raw == EXCLUDE_LABEL:
                final_mapping[label] = EXCLUDE_LABEL
            elif label in viable_label_set:
                if raw in split_set and label in viable_groups:
                    final_mapping[label] = label
                else:
                    final_mapping[label] = raw
            else:
                final_mapping[label] = _FALLBACK_GROUP

    else:
        group_order = [r[0] for r in _KEYWORD_RULES] + [_FALLBACK_GROUP]
        viable_groups = OrderedDict()
        for group in group_order:
            labels = raw_groups.get(group, [])
            if not labels:
                continue
            ct_mask = obs_valid["final_celltype"].isin(labels)
            sub = obs_valid[ct_mask]
            pat_counts = sub.groupby("patient_id").size()
            eligible = pat_counts[pat_counts >= min_cells_per_patient]
            if len(eligible) >= min_patients:
                viable_groups[group] = sorted(labels)

        viable_label_set = {
            label
            for labels in viable_groups.values()
            for label in labels
        }
        final_mapping = {}
        for label in all_labels:
            if raw_mapping[label] == EXCLUDE_LABEL:
                final_mapping[label] = EXCLUDE_LABEL
            elif label in viable_label_set:
                final_mapping[label] = raw_mapping[label]
            else:
                final_mapping[label] = _FALLBACK_GROUP

    if verbose:
        _print_summary(obs_df, viable_groups, final_mapping)

    return viable_groups, final_mapping


def _print_summary(obs_df, viable_groups, mapping):
    excluded_cells = int((obs_df["final_celltype"].map(mapping) == EXCLUDE_LABEL).sum())
    if excluded_cells:
        excl_labels = [l for l, g in mapping.items() if g == EXCLUDE_LABEL]
        print(f"  [EXCLUDED from training — unidentifiable: "
              f"{excl_labels} → {excluded_cells:,} cells dropped]")

    print("=== Inferred cell type groups ===")
    for group, labels in viable_groups.items():
        ct_mask = obs_df["final_celltype"].isin(labels)
        n_pats  = obs_df[ct_mask]["patient_id"].nunique()
        n_cells = int(ct_mask.sum())
        print(f"  {group}: {n_pats} patients, {n_cells:,} cells")
        for label in labels:
            lmask = obs_df["final_celltype"] == label
            p = obs_df[lmask]["patient_id"].nunique()
            c = int(lmask.sum())
            print(f"    {label:<50} {c:>6} cells, {p} patients")

    other_fallback = [
        label for label, grp in mapping.items()
        if grp == _FALLBACK_GROUP
    ]
    if other_fallback:
        print(f"  Other (no keyword match, fell through):")
        for label in sorted(other_fallback):
            lmask = obs_df["final_celltype"] == label
            p = obs_df[lmask]["patient_id"].nunique()
            c = int(lmask.sum())
            print(f"    {label:<50} {c:>6} cells, {p} patients")
