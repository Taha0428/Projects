# Sample Input Datasets for OmicsLab Multi-Omics Workstation

This directory contains pre-formatted sample input datasets ready to be uploaded directly into the OmicsLab web tool at `http://localhost:8050/`.

---

## 📁 Available Sample Files

| File Name | Samples | Description | Recommended For |
| :--- | :--- | :--- | :--- |
| **`cross_species_48_zebrafinch_human.csv`** | 48 | Cross-species cohort (24 Songbird/Zebra Finch + 24 Human cortical samples). Models biological differences (higher human baseline CpG methylation, compact chromatin, enhanced songbird FOXP2 plasticity). | Testing cross-species generalizability and translational neuroplasticity. |
| **`human_only_32_cortical.csv`** | 32 | Pure human postmortem/iPSC cortical tissue dataset (8 subjects per learning stage). Incorporates human-specific cellular heterogeneity and higher genetic diversity. | Testing purely on human brain neurobiology. |
| **`cross_species_80_large.csv`** | 80 | High-powered cross-species cohort (40 Songbird + 40 Human). 10 biological replicates per condition. | Testing high-throughput ML models with deep statistical power. |
| **`cohort_8_samples_quick_test.csv`** | 8 | Lightweight dataset (2 samples per condition: Untutored, Early 24h, 7d, Mastery 30d). | Rapid testing, verifying upload functionality in under 5 seconds. |
| **`cohort_24_samples_multiomics.csv`** | 24 | Complete standard animal multi-omics cohort with 6 biological replicates across all 4 tutoring epochs. | Full exploratory benchmarking and presentation. |
| **`cohort_48_samples_scaled.csv`** | 48 | Large-scale animal experimental cohort with 12 biological replicates per group. | Testing high-throughput ML performance and scaling. |
| **`template_empty_cohort.csv`** | 4 | Clean 4-sample reference structure showing all required column headers and value ranges. | Creating your own custom biological datasets. |

---

## 🧬 Required CSV Column Schema

Each dataset contains columns across all 4 multi-omics modalities:

1. **Metadata & Phenotype**:
   * `sample_id`: Unique identifier for each sample (e.g. `ANML_01_Untu_R1`).
   * `animal_id`: Biological replicate ID (used for Stratified Group K-Fold to prevent biological leakage).
   * `condition`: Experimental epoch (`Untutored_Isolate`, `Tutored_Early_24h`, `Tutored_Subchronic_7d`, `Tutored_Mastery_30d`).
   * `tutoring_hours`: Numeric continuous hours (`0`, `24`, `168`, `720`).
   * `true_plasticity_score`: Continuous progression score ($0.0 \to 1.0$).

2. **Single-Cell Deconvolution Proportions**:
   * `scProp_Glutamatergic_Neuron`, `scProp_GABAergic_Interneuron`, `scProp_Oligodendrocyte`, `scProp_Astrocytes`, `scProp_Microglia`.

3. **Transcriptomic Features (RNA-seq)**:
   * Prefixed with `RNA_` (e.g., `RNA_BDNF`, `RNA_EGR1`, `RNA_FOS`, `RNA_ARC`, `RNA_TET2`, etc.).

4. **Epigenomic Histone Marks (ChIP-seq)**:
   * Active promoter marks: `ChIP_H3K4me3_<GENE>`
   * Repressive polycomb marks: `ChIP_H3K27me3_<GENE>`

5. **DNA Methylation Features (WGBS)**:
   * Percentage CpG methylation: `WGBS_CpG_<GENE>` ($0.0\% \to 100.0\%$).

---

## 🚀 How to Use These Files in the Web Tool

1. Open your browser at **`http://localhost:8050/`**.
2. Navigate to **1. Ingest Data & Run** (the default starting tab).
3. Under **Option A: Upload Custom CSV**:
   * Click the drop area or drag and drop any file from this `sample_inputs/` folder (e.g., `cohort_24_samples_multiomics.csv`).
4. The schema inspector will immediately validate the samples, display the column breakdown, and render a preview table.
5. Click **`[ 🚀 Execute Full Pipeline ]`** to watch the real-time terminal process all 4 analytical stages!
