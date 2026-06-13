# irAEGIS
irAEGIS: Pathway-guided interpretable prediction of immune-related adverse events


## Overview
we present irAEGIS, a pathway-guided interpretable model for predicting irAE risk from pre-treatment scRNA-seq data. irAEGIS incorporates prior biological knowledge through gene set constrained representation learning, aggregates pathway activities within individual cell types, and integrates cell type specific risk signals through a hierarchical prediction framework. This design enables predictions to be decomposed into contributions from individual cell types, pathways, and genes, providing interpretable explanations across multiple biological scales. 

irAEGIS is built using PyTorch.
Test on: macOS 15.6 (Apple Silicon) / Ubuntu 22.04 LTS, Python 3.10, PyTorch 2.x. MPS/CUDA supported.

## Table of Contents

- [Requirements](#requirements)
- [Folder structure](#folder-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Benchmarking](#benchmarking)
- [Tutorial](#tutorial)

## Requirements
Required modules can be installed via `requirements.txt` under the project root:
```
pip install -r requirements.txt
```
Key packages:
```
torch>=2.0
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.5
anndata>=0.10
scanpy>=1.10
xgboost>=2.1
matplotlib>=3.8
```

## Folder structure
```
irAEGIS/
├── README.md
├── requirements.txt
├── tutorials/
│   ├── 01_preprocess.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_interpretability.ipynb
├── models/
│   ├── __init__.py
│   ├── iraegis/                       # PathwayAE and training utilities
│   │   ├── model_utils.py
│   │   ├── train_utils.py
│   │   ├── data_utils.py
│   │   └── train.py
│   └── baselines/
│       ├── pseudobulk_lr.py
│       ├── rf_pseudobulk.py
│       ├── xgboost_pseudobulk.py
│       ├── cell_lr.py
│       ├── singledeep.py
│       ├── hierarchical_mil.py
│       └── scrat.py
├── utils/
│   ├── config.py
│   ├── data_helpers.py
│   ├── celltype_groups.py
│   └── profiler.py
├── explainability/
│   ├── cell_explain.py                
│   ├── fidelity_occlusion.py          
│   ├── class_contrastivity.py         
│   └── ground_truth_recovery.py       
└── scripts/
    ├── run_iraegis_inference.py             
    ├── explain_patient.py                 
    ├── build_shared_genes.py              
    └── simulation/                        
        ├── simulate_splatter_rs.R         
        └── simulate_splatter_ds.R         
```

## Installation

Clone irAEGIS:
```
git clone https://github.com/abudhkar/irAEGIS
cd irAEGIS
pip install -r requirements.txt
```

## Dataset

irAEGIS is benchmarked on four publicly available pre-ICI PBMC scRNA-seq cohorts:

| Cohort | GEO |
|---|---|
| GSE189125 | [GSE189125](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189125) |
| GSE216329 | [GSE216329](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE216329) |
| GSE249898 | [GSE249898](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249898) |
| GSE285888 | [GSE285888](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285888) |

The Hallmark gene-to-pathway prior is loaded from a separate `.npz` (`pathway_prior.npz`) built from the MSigDB Hallmark gene set collection (50 gene sets). The original Hallmark `.gmt` file can be downloaded from MSigDB:

- [MSigDB Hallmark v2026.1](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H)

## Training

#### 1. irAEGIS on one cohort
By default the loader reads `datasets/processed_h5ad/<cohort_id>.h5ad`. Override the path with `--h5ad`.

```
python -m models.iraegis.train --cohort GSE189125_pre_ici --prior-genes-only
python scripts/run_iraegis_inference.py --cohorts GSE189125_pre_ici
```

Custom h5ad path:
```
python -m models.iraegis.train --cohort GSE189125_pre_ici \
    --h5ad path/to/GSE189125.h5ad --prior-genes-only
python scripts/run_iraegis_inference.py --cohorts GSE189125_pre_ici
```

**Cross-cohort gene vocabulary** (optional). When comparing attributions across cohorts, train each on a shared gene set. Generate the intersection once with `scripts/build_shared_genes.py`, then pass `--gene-list` to `train.py`:
```
python scripts/build_shared_genes.py \
    --h5ads datasets/processed_h5ad/GSE189125_pre_ici.h5ad \
            datasets/processed_h5ad/GSE216329_integrated_pre_ici.h5ad ... \
    --out datasets/processed_h5ad/shared_genes.txt
python -m models.iraegis.train --cohort GSE189125_pre_ici --prior-genes-only \
    --gene-list datasets/processed_h5ad/shared_genes.txt
```
`--gene-list` defaults to that path; pass `--gene-list ""` to use the cohort's native vocabulary. Not needed for single-cohort use.

#### 2. All four real cohorts
```
for c in GSE189125_pre_ici GSE216329_integrated_pre_ici \
         GSE249898_integrated_pre_ici GSE285888_pre_ici; do
    python -m models.iraegis.train --cohort $c --prior-genes-only
done
python scripts/run_iraegis_inference.py
```

#### 3. Cell-level attribution
Writes `cell_pathway_attribution.csv` and `cell_gene_attribution.csv` under `results/iraegis/<cohort>/cell_explainability/`. All downstream interpretability scripts read these.
```
python explainability/cell_explain.py --cohort GSE189125_pre_ici
```

#### 4. Interpretability metrics
```
python explainability/fidelity_occlusion.py --cohorts GSE189125_pre_ici
python explainability/class_contrastivity.py
```

#### 5. Ground-truth recovery on simulated cohorts
The simulated benchmark has two settings:
- **Distributed signal (DS)** — weak irAE-associated signal spread across all four simulated immune cell types.
- **Restricted signal (RS)** — strong irAE signal confined to four of six simulated cell types.

Computes the manuscript's recovery metrics by comparing irAEGIS's top-attributed pathways and genes to the injected ground-truth pathways:
- **Pathway recall@5** — fraction of injected GT pathways recovered among irAEGIS's top-5 attributed pathways per signal-carrying cell type.
- **Gene precision@30** — fraction of irAEGIS's top-30 attributed genes per signal-carrying cell type that belong to an injected GT pathway.

```
python explainability/ground_truth_recovery.py                # all 6 sim cohorts
python explainability/ground_truth_recovery.py --cohorts DS_cohort1 RS_cohort1
```
Output: `results/ground_truth_recovery.csv` (one row per cohort).

#### 6. Per-patient explanation
```
python scripts/explain_patient.py --cohort GSE189125_pre_ici
```

## Benchmarking

irAEGIS is compared against seven baselines on the same LOOCV protocol. Run each baseline on one cohort:
```
python -m models.baselines.pseudobulk_lr --cohort GSE189125_pre_ici
python -m models.baselines.rf_pseudobulk --cohort GSE189125_pre_ici
python -m models.baselines.xgboost_pseudobulk --cohort GSE189125_pre_ici
python -m models.baselines.cell_lr --cohort GSE189125_pre_ici
python -m models.baselines.singledeep --cohort GSE189125_pre_ici
python -m models.baselines.hierarchical_mil --cohort GSE189125_pre_ici
python -m models.baselines.scrat --cohort GSE189125_pre_ici
```

## Tutorial
Three-notebook walk-through under `tutorials/`:

| Notebook | Contents |
|---|---|
| [tutorials/01_preprocess.ipynb](tutorials/01_preprocess.ipynb) | Cohort loading, pathway prior inspection, patient and cell-type composition |
| [tutorials/02_model_training.ipynb](tutorials/02_model_training.ipynb) | Train the pathway-masked AE, aggregate to (patient × cell type), run classifier + LOOCV, report AUC / AUPRC |
| [tutorials/03_interpretability.ipynb](tutorials/03_interpretability.ipynb) | Fidelity ΔAUC, class-contrastivity, ground-truth recovery on simulations |

## Cite

Please cite the irAEGIS paper if you use this code in your own work.
