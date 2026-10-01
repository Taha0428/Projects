# Multi-Omics Epigenetic & Neuroplasticity Modeling Suite

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest%20%7C%20MLP-orange.svg)](https://scikit-learn.org/)
[![Attribution](https://img.shields.io/badge/Interpretability-SHAP-purple.svg)](https://github.com/slundberg/shap)
[![Status](https://img.shields.io/badge/Delivery-Production%20Verified-brightgreen.svg)]()

An end-to-end computational pipeline and interactive discovery workbench integrating **Transcriptomics (RNA-seq)**, **Histone Epigenomics (ChIP-seq H3K4me3 / H3K27me3)**, **Whole-Genome Bisulfite Sequencing (WGBS DNA Methylation)**, and **Single-Cell Cell-Type Specificity** across biological tutoring and neuroplasticity cohorts.

---

## Pipeline Architecture

```
[ Multi-Omics Data Inputs ]
      (RNA-seq, ChIP-seq, WGBS, Single-Cell / Cell-Type Specific)
                               │
                               ▼
            [ Stage 1: Quality Control & Processing ]
          • Quality Control: FastQC / MultiQC (Q-scores, duplication %, GC)
          • Alignment: STAR / HISAT2 (Unique mapping, splice junctions)
          • Peak & Methylation Calling: MACS3 / Bismark (FRiP, narrow/broad peaks, CpG %)
                               │
                               ▼
           [ Stage 2: Feature Matrix Construction ]
     • Differentially Expressed Genes (DEGs) [DESeq2 Wald Test, FDR < 0.05]
     • Differentially Methylated Regions (DMRs) [Promoter & Gene-Body Δβ Shifts]
     • Histone Enrichment Scores [H3K4me3 Active / H3K27me3 Polycomb Repressive]
     • Multi-Modal Interaction & Epigenetic Permissiveness Index
                               │
                               ▼
            [ Stage 3: Machine Learning & Modeling ]
          • Frameworks: Python (scikit-learn, XGBoost, MLP / PyTorch-aligned)
          • Models: XGBoost, Random Forests, Multi-Layer Perceptrons
          • Cross-Validation: Stratified Group K-Fold across biological animal replicates
          • Leakage Prevention: Zero cross-animal replicate contamination
                               │
                               ▼
        [ Stage 4: Feature Attribution & Interpretation ]
          • SHAP Value Calculation (Local waterfall attributions & global ranking)
          • Biomarker Identification (Master Regulatory Genes via MOPCI Concordance)
          • Plasticity Trajectory Mapping across Tutoring Conditions (Latent manifold & kinetic waves)
```

---

## Key Features

1. **Production-Ready & Fully Functional**:
   - Zero-dependency interactive web dashboard (`http.server` backend) with interactive SVG visualizations.
   - Comprehensive CLI (`run_pipeline.py`) supporting end-to-end execution, stage-specific runs, and web serving.
2. **Biological Integrity**:
   - **Stratified Group K-Fold Cross-Validation**: Animals (biological replicates) are grouped strictly so that training and test folds never share samples from the same biological donor.
   - **Multi-Omics Plasticity Concordance Index (MOPCI)**: Identifies master regulatory genes where transcriptional changes correlate with chromatin opening (H3K4me3 gain) and active DNA demethylation (CpG hypomethylation).
3. **Interactive Data Workbench**:
   - **Upload Custom Data**: Drag-and-drop or upload custom CSV datasets with your own counts and epigenetic assays.
   - **Download Template CSV**: Pre-formatted CSV structure for immediate data ingest.
   - **Live Gene Locus Explorer**: Query any gene (e.g., *BDNF*, *FOXP2*, *EGR1*, *TET2*) to view multi-omic expression, DMR status, SHAP importance, and trajectory kinetics.
   - **Cohort Synthesis**: Tailor sample sizes, biological replicates per condition, and random seeds.

---

## Directory Structure

```
Secret Project/
├── data/                                 # Datasets & synthetic cohort generator
│   └── synthetic_cohort.py               # Biologically realistic multi-omics cohort engine
├── multiomics_pipeline/                  # Core pipeline modules
│   ├── config.py                         # Global configuration & biological conditions
│   ├── pipeline.py                       # Master 4-stage pipeline orchestrator
│   ├── stage1_qc_processing/             # Stage 1: FastQC, MultiQC, STAR, MACS3, Bismark
│   │   ├── qc_fastqc.py
│   │   ├── alignment_star.py
│   │   └── peaks_methylation.py
│   ├── stage2_feature_matrix/            # Stage 2: DESeq2 DEGs, DMRs, Histone Scoring
│   │   ├── differential_expression.py
│   │   ├── differential_methylation.py
│   │   ├── histone_enrichment.py
│   │   └── matrix_integrator.py
│   ├── stage3_ml_modeling/               # Stage 3: XGBoost, Random Forest, MLP, Group K-Fold
│   │   ├── cross_validation.py
│   │   ├── train_models.py
│   │   └── evaluator.py
│   └── stage4_interpretation/            # Stage 4: SHAP attributions, Biomarkers, Trajectory
│       ├── shap_explainer.py
│       ├── biomarkers.py
│       └── plasticity_trajectory.py
├── pipeline_outputs/                     # Generated artifacts & CSV exports
│   ├── stage1_qc/                        # QC & alignment summaries
│   ├── stage2_features/                  # Normalized matrices, DEGs, DMRs
│   ├── stage3_models/                    # Leaderboards, confusion matrices
│   ├── stage4_interpretation/            # SHAP rankings, master regulators, trajectory points
│   └── web_export/                       # Consolidated JSON payload for web UI
├── web/                                  # Web dashboard frontend
│   ├── index.html                        # Multi-tab modern glassmorphic dashboard
│   ├── style.css                         # Dark theme, typography & responsive layouts
│   └── app.js                            # Interactive SVG charts, API polling & uploads
├── run_pipeline.py                       # Master CLI runner
├── serve_dashboard.py                    # REST API & Web Dashboard Server
├── DELIVERY_REPORT.md                    # Formal scientific deliverable report
└── README.md                             # Architectural documentation
```

---

## Quickstart Guide

### 1. Execute the Master Pipeline via CLI

Run the entire 4-stage pipeline end-to-end:
```bash
python run_pipeline.py --all
```

Run a specific stage:
```bash
python run_pipeline.py --stage 1  # Quality Control & Alignments
python run_pipeline.py --stage 2  # Feature Matrix Construction (DESeq2, DMRs)
python run_pipeline.py --stage 3  # Machine Learning & Group K-Fold CV
python run_pipeline.py --stage 4  # SHAP Feature Attribution & Trajectory
```

### 2. Launch the Interactive Web Dashboard

Start the live interactive workbench:
```bash
python run_pipeline.py --serve
# Or directly:
python serve_dashboard.py 8050
```

Open your browser at `http://localhost:8050`.

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | `GET` | Returns current pipeline execution status and dataset source |
| `/api/results` | `GET` | Returns consolidated multi-omics analysis JSON payload |
| `/api/download-template` | `GET` | Downloads sample CSV template for custom data upload |
| `/api/gene-query?gene=GENE` | `GET` | Real-time multi-omic profile lookup for any gene locus |
| `/api/upload-cohort` | `POST` | Uploads custom CSV dataset and launches analysis pipeline |
| `/api/generate-custom-cohort`| `POST` | Synthesizes a new biological cohort with specified parameters |
| `/api/run-pipeline` | `POST` | Triggers asynchronous re-run of Stages 1 through 4 |

---

## Mathematical Formulations

### Stage 2: Differential Expression (DESeq2 Approximation)
For gene $i$ and condition $k$:
$$\log_2 \text{FC}_i = \log_2\left(\frac{\bar{y}_{i, \text{tutored}} + \epsilon}{\bar{y}_{i, \text{control}} + \epsilon}\right)$$
Wald statistic with Benjamini-Hochberg FDR correction:
$$W_i = \frac{\hat{\beta}_i}{\text{SE}(\hat{\beta}_i)}, \quad P_{\text{adj}(i)} = \min_{j \ge i} \left(\frac{m}{j} P_{(j)}\right)$$

### Stage 2: Epigenetic Permissiveness Index
Integrates active promoter mark ($\text{H3K4me3}$), polycomb repressive mark ($\text{H3K27me3}$), and DNA methylation ($\beta$):
$$\text{EPI}_i = \frac{\text{ChIP}_{\text{H3K4me3}, i}}{\text{ChIP}_{\text{H3K27me3}, i} + \text{WGBS}_{\text{CpG}, i} + \epsilon}$$

### Stage 4: Multi-Omics Plasticity Concordance Index (MOPCI)
$$\text{MOPCI}_i = 3.0 \cdot \overline{|\phi_i|} + 1.5 \cdot |\log_2 \text{FC}_i| + 2.0 \cdot |\Delta \beta_i|$$
Where $\overline{|\phi_i|}$ is the aggregate global SHAP importance of gene $i$ across modalities.
