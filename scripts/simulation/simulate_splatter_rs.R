#!/usr/bin/env Rscript
# =============================================================================
# simulate_splatter_rs.R
#
# Severe vs non-severe irAE simulation with 6 cell types, distributed
# signal across 4 CTs (1 pathway each), weaker per-CT effect.
#
# Cell types: T_cells, B_cells, Monocytes, NK_cells, Dendritic_cells, Neutrophils
# Signal CTs (4): T_cells, Monocytes, Dendritic_cells, NK_cells
# Noise CTs (2): B_cells, Neutrophils (no injected signal)
#
# Labels: Severe vs Non-severe
#
# 3 cohorts x 30 patients = 90 total (matches real data scale).
#
# Output (simulation_rs/):
#   sim_counts.mtx, sim_obs.csv, sim_var.csv,
#   ground_truth.csv, ground_truth_pathways.csv
#
# Usage:
#   Rscript scripts/simulate_splatter_rs.R
# =============================================================================

suppressPackageStartupMessages({
  library(splatter)
  library(SingleCellExperiment)
  library(Matrix)
})

set.seed(2024)

args      <- commandArgs(trailingOnly = FALSE)
file_flag <- grep("--file=", args, value = TRUE)
SCRIPT_DIR  <- if (length(file_flag) > 0) dirname(normalizePath(sub("--file=", "", file_flag))) else getwd()
PROJECT_DIR <- dirname(dirname(SCRIPT_DIR))

GMT_FILE    <- file.path(PROJECT_DIR, "datasets", "resources", "h.all.v2026.1.Hs.symbols.gmt")
GENE_FILE   <- file.path(PROJECT_DIR, "datasets", "processed_h5ad", "shared_genes.txt")
OUT_DIR     <- file.path(PROJECT_DIR, "datasets", "simulation_rs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Config
BATCHES <- list(
  batch1 = list(n_yes = 15, n_no = 15, name = "RS_cohort1"),
  batch2 = list(n_yes = 12, n_no = 18, name = "RS_cohort2"),
  batch3 = list(n_yes = 16, n_no = 14, name = "RS_cohort3")
)

# Total: 43 Severe / 47 Non-severe = 90 patients (30 per cohort)
CELLS_MEAN   <- 500
CELLS_SIZE   <- 3
N_GENES      <- 5000

CT_NAMES      <- c("T_cells", "B_cells", "Monocytes", "NK_cells",
                    "Dendritic_cells", "Neutrophils")
CT_ALPHA      <- c(8.0, 3.0, 5.0, 2.0, 2.5, 4.0)

# Signal parameters — weaker per-CT effect (distributed across 4 CTs).
DE_PROB_SIGNAL   <- 0.35    # 35% of pathway genes perturbed
DE_FAC_LOC       <- 0.70    # log2 FC ~ 1.62x per gene — moderate
DE_FAC_SCALE     <- 0.20    # gene-to-gene variance in FC
PAT_FC_SD        <- 0.35    # patient heterogeneity
PAT_ACTIVE_PROB  <- 0.80    # 20% of patients in signal group show no effect
CELL_FRACTION    <- 0.30    # 30% of signal-CT cells carry perturbation

# Ground-truth signals: 8 total (4 Severe + 4 Non-severe), 1 pathway per CT.
# Signal distributed across 4 CTs so no single CT is sufficient alone.
#
# Pathways injected.
#   T_cells:         Severe = IL-6/JAK/STAT3
#                    Non-severe = IL-2/STAT5
#   Monocytes:       Severe = TNFα/NF-κB
#                    Non-severe = OXPHOS
#   Dendritic_cells: Severe = IFN-α
#                    Non-severe = TGF-β
#   NK_cells:        Severe = INFLAMMATORY_RESPONSE
#                    Non-severe = APOPTOSIS
#
# B_cells and Neutrophils have NO injected signal (noise CTs).
SIGNAL_YES <- list(
  T_cells         = c("HALLMARK_IL6_JAK_STAT3_SIGNALING"),
  Monocytes       = c("HALLMARK_TNFA_SIGNALING_VIA_NFKB"),
  Dendritic_cells = c("HALLMARK_INTERFERON_ALPHA_RESPONSE"),
  NK_cells        = c("HALLMARK_INFLAMMATORY_RESPONSE")
)

SIGNAL_NO <- list(
  T_cells         = c("HALLMARK_IL2_STAT5_SIGNALING"),
  Monocytes       = c("HALLMARK_OXIDATIVE_PHOSPHORYLATION"),
  Dendritic_cells = c("HALLMARK_TGF_BETA_SIGNALING"),
  NK_cells        = c("HALLMARK_APOPTOSIS")
)

# Background noise
BG_NOISE_PROB   <- 0.08 
BG_NOISE_FC_LOC <- 0.4

# Per-cohort signal strength with batch noise constant
# across cohorts so the only variation is signal strength:
#   RS_cohort1: medium signal
#   RS_cohort2: weak signal   (probe of low-SNR regime)
#   RS_cohort3: strong signal (probe of high-SNR regime)
BATCH_SPLATTER_PARAMS <- list(
  RS_cohort1 = list(mean.rate = 0.35, bcv.common = 0.25, de_fac_loc = 0.70),
  RS_cohort2 = list(mean.rate = 0.35, bcv.common = 0.25, de_fac_loc = 0.55),
  RS_cohort3 = list(mean.rate = 0.35, bcv.common = 0.25, de_fac_loc = 0.85)
)

cat("Loading genes ...\n")
all_genes <- readLines(GENE_FILE)
all_genes <- all_genes[nchar(trimws(all_genes)) > 0]
cat(sprintf("  Shared genes: %d genes\n", length(all_genes)))

cat("Parsing Hallmark GMT...\n")
gmt_lines <- readLines(GMT_FILE)
pathways  <- list()
for (line in gmt_lines) {
  parts <- strsplit(line, "\t")[[1]]
  pathways[[parts[1]]] <- parts[-(1:2)]
}
cat(sprintf("  Loaded %d pathways\n", length(pathways)))

pathway_genes_in <- intersect(unique(unlist(pathways)), all_genes)
background_genes <- setdiff(all_genes, pathway_genes_in)
n_bg             <- N_GENES - length(pathway_genes_in)
if (n_bg < 0) { pathway_genes_in <- sample(pathway_genes_in, N_GENES); n_bg <- 0 }
bg_sample    <- sample(background_genes, min(n_bg, length(background_genes)))
GENE_UNIVERSE <- c(pathway_genes_in, bg_sample)
cat(sprintf("  Gene space: %d pathway + %d background = %d total\n",
            length(pathway_genes_in), length(bg_sample), length(GENE_UNIVERSE)))

# Dirichlet sampler
rdirichlet <- function(alpha) {
  x <- rgamma(length(alpha), shape = alpha, rate = 1)
  x / sum(x)
}

batch_hash <- function(batch_name) {
  digits <- gsub("[^0-9]", "", batch_name)
  as.integer(digits)
}

# Simulate one cell type for one batch 
simulate_celltype_batch <- function(ct_name, yes_pat_ids, no_pat_ids,
                                    cells_per_pat_yes, cells_per_pat_no,
                                    batch_name, splatter_params,
                                    pathway_signal_yes = NULL,
                                    pathway_signal_no  = NULL) {

  n_cells_yes   <- sum(cells_per_pat_yes)
  n_cells_no    <- sum(cells_per_pat_no)
  n_cells_total <- n_cells_yes + n_cells_no
  if (n_cells_total == 0) return(NULL)

  cat(sprintf("    %s / %s: %d Yes cells (%d pats), %d No cells (%d pats)\n",
              batch_name, ct_name,
              n_cells_yes, length(yes_pat_ids),
              n_cells_no,  length(no_pat_ids)))

  params <- newSplatParams(
    nGenes     = length(GENE_UNIVERSE),
    batchCells = n_cells_total,
    de.prob    = 0.0,
    mean.rate  = splatter_params$mean.rate,
    bcv.common = splatter_params$bcv.common,
    seed       = as.integer(nchar(ct_name) * 137 + batch_hash(batch_name) * 31 +
                            n_cells_total %% 1000)
  )
  sce <- splatSimulate(params, method = "single", verbose = FALSE)
  rownames(sce) <- GENE_UNIVERSE
  cnt <- as.matrix(counts(sce))

  no_pat_ids_cell  <- rep(no_pat_ids,  times = cells_per_pat_no)
  yes_pat_ids_cell <- rep(yes_pat_ids, times = cells_per_pat_yes)
  all_pat_ids      <- c(no_pat_ids_cell, yes_pat_ids_cell)
  groups           <- c(rep("Group1", n_cells_no), rep("Group2", n_cells_yes))

  shuf_seed <- as.integer(nchar(ct_name) * 211 + batch_hash(batch_name) * 43)
  set.seed(shuf_seed)
  shuf_idx     <- sample(n_cells_total)
  cnt          <- cnt[, shuf_idx, drop = FALSE]
  all_pat_ids  <- all_pat_ids[shuf_idx]
  groups       <- groups[shuf_idx]

  # Per-pathway DE gene tracking
  de_genes_yes_list <- list()
  de_genes_no_list  <- list()

  # Inject Yes signals
  # Patitent-specific DE GENES
  if (!is.null(pathway_signal_yes)) {
    for (pw_idx in seq_along(pathway_signal_yes)) {
      pw_name  <- pathway_signal_yes[pw_idx]
      pw_genes <- intersect(pathways[[pw_name]], GENE_UNIVERSE)
      if (length(pw_genes) == 0) {
        cat(sprintf("      [Yes] %s: 0 genes in universe — skipped\n", pw_name))
        next
      }
      n_sig <- max(1, round(length(pw_genes) * DE_PROB_SIGNAL))

      n_yes    <- length(yes_pat_ids)
      n_active <- round(n_yes * PAT_ACTIVE_PROB)
      set.seed(as.integer(nchar(ct_name) * 99 + batch_hash(batch_name) * 13 + pw_idx * 71))
      active_pats <- sample(yes_pat_ids, n_active, replace = FALSE)

      cat(sprintf("      [Yes] %s: %d/%d DE genes/patient, %d/%d active patients\n",
                  pw_name, n_sig, length(pw_genes), n_active, n_yes))

      all_de_genes <- c()
      fc_vals <- c()
      for (pat_i in seq_along(active_pats)) {
        pat <- active_pats[pat_i]
        # Patient-specific DE gene selection
        set.seed(as.integer(nchar(ct_name) * 42 + batch_hash(batch_name) * 7 +
                            pw_idx * 113 + pat_i * 997))
        de_genes_pat <- sample(pw_genes, n_sig)
        de_idx_pat   <- which(GENE_UNIVERSE %in% de_genes_pat)
        all_de_genes <- union(all_de_genes, de_genes_pat)

        pat_log2fc <- max(rnorm(1, mean = splatter_params$de_fac_loc, sd = PAT_FC_SD), 0.3)
        gene_fc    <- 2^rnorm(n_sig, mean = pat_log2fc, sd = DE_FAC_SCALE)
        pat_cells  <- which(all_pat_ids == pat)
        if (length(pat_cells) == 0) next
        n_affected <- max(1, round(length(pat_cells) * CELL_FRACTION))
        set.seed(as.integer(nchar(ct_name) * 17 + batch_hash(batch_name) * 29 +
                            pw_idx * 53 + nchar(pat) * 7))
        affected_cells <- sample(pat_cells, n_affected, replace = FALSE)
        for (j in seq_along(de_idx_pat))
          cnt[de_idx_pat[j], affected_cells] <-
              round(cnt[de_idx_pat[j], affected_cells] * gene_fc[j])
        fc_vals <- c(fc_vals, mean(gene_fc))
      }
      cnt[cnt < 0] <- 0L
      if (length(fc_vals) > 0)
        cat(sprintf("      Mean FC: %.2fx  Range: %.2fx-%.2fx  (fraction=%.0f%%)\n",
                    mean(fc_vals), min(fc_vals), max(fc_vals),
                    CELL_FRACTION * 100))

      de_genes_yes_list[[pw_name]] <- all_de_genes
    }
  }

  # Inject No signals
  if (!is.null(pathway_signal_no)) {
    for (pw_idx in seq_along(pathway_signal_no)) {
      pw_name    <- pathway_signal_no[pw_idx]
      pw_genes   <- intersect(pathways[[pw_name]], GENE_UNIVERSE)
      if (length(pw_genes) == 0) {
        cat(sprintf("      [No]  %s: 0 genes in universe — skipped\n", pw_name))
        next
      }
      n_sig      <- max(1, round(length(pw_genes) * DE_PROB_SIGNAL))

      n_no       <- length(no_pat_ids)
      n_active   <- round(n_no * PAT_ACTIVE_PROB)
      set.seed(as.integer(nchar(ct_name) * 88 + batch_hash(batch_name) * 23 + pw_idx * 97))
      active_no  <- sample(no_pat_ids, n_active, replace = FALSE)

      cat(sprintf("      [No]  %s: %d/%d DE genes/patient, %d/%d active patients\n",
                  pw_name, n_sig, length(pw_genes), n_active, n_no))

      all_de_genes <- c()
      fc_vals <- c()
      for (pat_i in seq_along(active_no)) {
        pat <- active_no[pat_i]
        # Patient-specific DE gene selection
        set.seed(as.integer(nchar(ct_name) * 55 + batch_hash(batch_name) * 17 +
                            pw_idx * 131 + pat_i * 991))
        de_genes_pat <- sample(pw_genes, n_sig)
        de_idx_pat   <- which(GENE_UNIVERSE %in% de_genes_pat)
        all_de_genes <- union(all_de_genes, de_genes_pat)

        pat_log2fc <- max(rnorm(1, mean = splatter_params$de_fac_loc, sd = PAT_FC_SD), 0.3)
        gene_fc    <- 2^rnorm(n_sig, mean = pat_log2fc, sd = DE_FAC_SCALE)
        pat_cells  <- which(all_pat_ids == pat)
        if (length(pat_cells) == 0) next
        n_affected <- max(1, round(length(pat_cells) * CELL_FRACTION))
        set.seed(as.integer(nchar(ct_name) * 19 + batch_hash(batch_name) * 31 +
                            pw_idx * 59 + nchar(pat) * 11))
        affected_cells <- sample(pat_cells, n_affected, replace = FALSE)
        for (j in seq_along(de_idx_pat))
          cnt[de_idx_pat[j], affected_cells] <-
              round(cnt[de_idx_pat[j], affected_cells] * gene_fc[j])
        fc_vals <- c(fc_vals, mean(gene_fc))
      }
      cnt[cnt < 0] <- 0L
      if (length(fc_vals) > 0)
        cat(sprintf("      Mean FC: %.2fx  Range: %.2fx-%.2fx  (fraction=%.0f%%)\n",
                    mean(fc_vals), min(fc_vals), max(fc_vals),
                    CELL_FRACTION * 100))

      de_genes_no_list[[pw_name]] <- all_de_genes
    }
  }

  # Background noise: random DE in ALL patients
  # Identify non-signal genes (background)
  all_signal_genes <- unique(c(unlist(de_genes_yes_list), unlist(de_genes_no_list)))
  bg_genes <- setdiff(GENE_UNIVERSE, all_signal_genes)
  n_bg_noise <- max(1, round(length(bg_genes) * BG_NOISE_PROB))
  set.seed(as.integer(nchar(ct_name) * 73 + batch_hash(batch_name) * 41))
  noise_genes <- sample(bg_genes, n_bg_noise)
  noise_idx   <- which(GENE_UNIVERSE %in% noise_genes)

  if (length(noise_idx) > 0) {
    for (ci in seq_len(ncol(cnt))) {
      gene_fc <- 2^rnorm(length(noise_idx), mean = BG_NOISE_FC_LOC, sd = 0.15)
      # Randomly up or down
      direction <- sample(c(1, -1), length(noise_idx), replace = TRUE)
      gene_fc <- ifelse(direction > 0, gene_fc, 1.0 / gene_fc)
      cnt[noise_idx, ci] <- round(pmax(cnt[noise_idx, ci] * gene_fc, 0))
    }
    cat(sprintf("      [BG noise] %d genes perturbed (both groups)\n", n_bg_noise))
  }

  list(
    counts            = Matrix(cnt, sparse = TRUE),
    pat_ids_cell      = all_pat_ids,
    groups            = groups,
    de_genes_yes_list = de_genes_yes_list,
    de_genes_no_list  = de_genes_no_list
  )
}

# Main simulation
cat("\n=== Simulating (6 CTs, 4 signal, 2 noise — 8 pathway signals) ===\n")
cat(sprintf("Batches: %d  |  Total patients: Severe=%d, Non-severe=%d\n",
            length(BATCHES),
            sum(sapply(BATCHES, `[[`, "n_yes")),
            sum(sapply(BATCHES, `[[`, "n_no"))))

n_yes_pw <- sum(sapply(SIGNAL_YES, length))
n_no_pw  <- sum(sapply(SIGNAL_NO, length))
cat(sprintf("Signal: %d Severe pathways + %d Non-severe pathways = %d total (across 4 signal CTs)\n",
            n_yes_pw, n_no_pw, n_yes_pw + n_no_pw))
cat(sprintf("Signal CTs: %s\n", paste(names(SIGNAL_YES), collapse = ", ")))
cat(sprintf("Noise CTs: B_cells, Neutrophils\n"))

all_counts   <- list()
all_metadata <- list()
ground_truth <- list()
cell_counter <- 0
pat_counter  <- 0

for (b_name in names(BATCHES)) {
  batch      <- BATCHES[[b_name]]
  cohort     <- batch$name
  sp_params  <- BATCH_SPLATTER_PARAMS[[cohort]]

  yes_pats <- paste0("pat_", cohort, "_Yes_", seq_len(batch$n_yes))
  no_pats  <- paste0("pat_", cohort, "_No_",  seq_len(batch$n_no))
  all_pats <- c(yes_pats, no_pats)
  n_pats   <- length(all_pats)

  set.seed(batch_hash(cohort) * 307)
  pat_order <- sample(n_pats)
  all_pats  <- all_pats[pat_order]

  cat(sprintf("\n--- Batch: %s (%s) | %d patients (%d Severe, %d Non-severe) ---\n",
              b_name, cohort, n_pats, batch$n_yes, batch$n_no))

  set.seed(pat_counter + 1)
  cells_total_per_pat <- rnbinom(n_pats, mu = CELLS_MEAN, size = CELLS_SIZE)
  cells_total_per_pat <- pmax(cells_total_per_pat, 50)

  set.seed(pat_counter + 2)
  ct_props_per_pat <- t(sapply(seq_len(n_pats), function(i) rdirichlet(CT_ALPHA)))
  colnames(ct_props_per_pat) <- CT_NAMES

  cells_per_pat_ct <- round(ct_props_per_pat * cells_total_per_pat)
  cells_per_pat_ct[cells_per_pat_ct < 0] <- 0

  cat(sprintf("  Cells/patient: mean=%d, min=%d, max=%d\n",
              round(mean(cells_total_per_pat)),
              min(cells_total_per_pat), max(cells_total_per_pat)))

  for (ct in CT_NAMES) {
    sig_yes <- SIGNAL_YES[[ct]]
    sig_no  <- SIGNAL_NO[[ct]]

    yes_all_idx <- which(all_pats %in% yes_pats)
    no_all_idx  <- which(all_pats %in% no_pats)

    cells_yes <- cells_per_pat_ct[yes_all_idx, ct]
    cells_no  <- cells_per_pat_ct[no_all_idx,  ct]

    res <- simulate_celltype_batch(
      ct_name          = ct,
      yes_pat_ids      = yes_pats,
      no_pat_ids       = no_pats,
      cells_per_pat_yes = cells_yes,
      cells_per_pat_no  = cells_no,
      batch_name       = cohort,
      splatter_params  = sp_params,
      pathway_signal_yes = sig_yes,
      pathway_signal_no  = sig_no
    )

    if (is.null(res)) next

    n_total    <- ncol(res$counts)
    cell_names <- paste0("cell_", seq_len(n_total) + cell_counter)
    cell_counter <- cell_counter + n_total
    colnames(res$counts) <- cell_names

    groups  <- res$groups
    labels  <- ifelse(groups == "Group2", "Severe", "Non-severe")
    pat_ids <- res$pat_ids_cell

    key <- paste0(cohort, "_", ct)
    all_counts[[key]] <- res$counts
    all_metadata[[key]] <- data.frame(
      cell_id         = cell_names,
      patient_id      = pat_ids,
      celltype        = ct,
      irAE_status     = labels,
      dataset_id      = cohort,
      batch_harmony   = cohort,
      patient_harmony = pat_ids,
      final_celltype  = ct,
      stringsAsFactors = FALSE
    )

    # Record ground truth
    for (pw_name in names(res$de_genes_yes_list)) {
      de_genes <- res$de_genes_yes_list[[pw_name]]
      if (length(de_genes) > 0) {
        ground_truth[[paste0(cohort, "_", ct, "_yes_", pw_name)]] <- data.frame(
          cohort    = cohort,
          celltype  = ct,
          pathway   = pw_name,
          gene      = de_genes,
          direction = "Severe",
          stringsAsFactors = FALSE
        )
      }
    }
    for (pw_name in names(res$de_genes_no_list)) {
      de_genes <- res$de_genes_no_list[[pw_name]]
      if (length(de_genes) > 0) {
        ground_truth[[paste0(cohort, "_", ct, "_no_", pw_name)]] <- data.frame(
          cohort    = cohort,
          celltype  = ct,
          pathway   = pw_name,
          gene      = de_genes,
          direction = "Non-severe",
          stringsAsFactors = FALSE
        )
      }
    }
  }

  pat_counter <- pat_counter + n_pats
}

cat("\n=== Combining ===\n")
combined_counts   <- do.call(cbind, all_counts)
combined_metadata <- do.call(rbind, all_metadata)
ground_truth_df   <- do.call(rbind, ground_truth)

pat_meta <- combined_metadata[!duplicated(combined_metadata$patient_id), ]
n_yes_total <- sum(pat_meta$irAE_status == "Severe")
n_no_total  <- sum(pat_meta$irAE_status == "Non-severe")

cat(sprintf("  Total cells:    %d\n", ncol(combined_counts)))
cat(sprintf("  Total genes:    %d\n", nrow(combined_counts)))
cat(sprintf("  Total patients: %d (Severe=%d, Non-severe=%d)\n",
            nrow(pat_meta), n_yes_total, n_no_total))
cat(sprintf("  Batches: %s\n", paste(unique(pat_meta$dataset_id), collapse = ", ")))
cat("\n=== Saving outputs ===\n")

writeMM(combined_counts, file.path(OUT_DIR, "sim_counts.mtx"))
cat("  Saved sim_counts.mtx\n")

write.csv(data.frame(gene = GENE_UNIVERSE),
          file.path(OUT_DIR, "sim_var.csv"), row.names = FALSE)
cat("  Saved sim_var.csv\n")

write.csv(combined_metadata, file.path(OUT_DIR, "sim_obs.csv"), row.names = FALSE)
cat("  Saved sim_obs.csv\n")

write.csv(ground_truth_df, file.path(OUT_DIR, "ground_truth.csv"), row.names = FALSE)
cat("  Saved ground_truth.csv\n")

# Pathway summary file
pw_summary <- aggregate(gene ~ cohort + celltype + pathway + direction,
                        data = ground_truth_df, FUN = length)
colnames(pw_summary)[colnames(pw_summary) == "gene"] <- "n_genes_injected"
pw_summary <- pw_summary[order(pw_summary$cohort, pw_summary$celltype,
                                pw_summary$pathway), ]
write.csv(pw_summary, file.path(OUT_DIR, "ground_truth_pathways.csv"),
          row.names = FALSE)
cat("  Saved ground_truth_pathways.csv\n")

cat("\n=== Ground Truth Summary ===\n")
n_pw_total <- length(unique(paste0(ground_truth_df$celltype, "_",
                                   ground_truth_df$pathway, "_",
                                   ground_truth_df$direction)))
n_pw_sev   <- length(unique(paste0(ground_truth_df$celltype[ground_truth_df$direction == "Severe"], "_",
                                   ground_truth_df$pathway[ground_truth_df$direction == "Severe"])))
n_pw_non   <- length(unique(paste0(ground_truth_df$celltype[ground_truth_df$direction == "Non-severe"], "_",
                                   ground_truth_df$pathway[ground_truth_df$direction == "Non-severe"])))

cat(sprintf("  Total injected pathways:  %d\n", n_pw_total))
cat(sprintf("  Severe pathways:          %d\n", n_pw_sev))
cat(sprintf("  Non-severe pathways:      %d\n", n_pw_non))
cat(sprintf("  Total injected genes:     %d (unique across all cohorts)\n",
            length(unique(ground_truth_df$gene))))

cat("\n  Per-cohort pathway summary:\n")
for (coh in unique(ground_truth_df$cohort)) {
  sub <- ground_truth_df[ground_truth_df$cohort == coh, ]
  pws <- unique(paste0(sub$celltype, " / ", sub$pathway, " (", sub$direction, ")"))
  cat(sprintf("    %s: %d pathways\n", coh, length(pws)))
  for (p in sort(pws)) {
    n_g <- nrow(sub[paste0(sub$celltype, " / ", sub$pathway, " (", sub$direction, ")") == p, ])
    cat(sprintf("      %s  [%d genes]\n", p, n_g))
  }
}

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("  Severe vs Non-severe irAE simulation\n"))
cat(sprintf("  6 CTs: 4 signal (T_cells, Monocytes, Dendritic_cells, NK_cells) + 2 noise (B_cells, Neutrophils)\n"))
cat(sprintf("  3 cohorts x 30 patients = 90 patients total\n"))
cat(sprintf("  8 pathway signals distributed across 4 CTs (1 per CT, weaker per-CT effect)\n"))
cat(sprintf("  Per-cohort DE_FAC_LOC: SIM_1=%.2f, SIM_2=%.2f, SIM_3=%.2f  PAT_ACTIVE=%.0f%%, CELL_FRAC=%.0f%%\n",
            BATCH_SPLATTER_PARAMS$RS_cohort1$de_fac_loc,
            BATCH_SPLATTER_PARAMS$RS_cohort2$de_fac_loc,
            BATCH_SPLATTER_PARAMS$RS_cohort3$de_fac_loc,
            PAT_ACTIVE_PROB * 100, CELL_FRACTION * 100))
cat(sprintf("  Output: %s/\n", OUT_DIR))
cat(sprintf("  Next: python data/build_sim_small_h5ad.py\n"))
