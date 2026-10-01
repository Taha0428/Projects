"""
Multi-Omics Epigenetic & Transcriptomic Modeling Configuration
Defines pipeline paths, biological conditions, target assays, and hyperparameters.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

# Root directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "pipeline_outputs"
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class MultiOmicsConfig:
    # Biological Tutoring Experimental Conditions
    conditions: List[str] = field(default_factory=lambda: [
        "Untutored_Isolate",
        "Tutored_Early_24h",
        "Tutored_Subchronic_7d",
        "Tutored_Mastery_30d"
    ])
    
    # Assays
    assays: List[str] = field(default_factory=lambda: [
        "RNA-seq", 
        "ChIP-seq_H3K4me3", 
        "ChIP-seq_H3K27me3", 
        "WGBS_DNAme",
        "scRNA_Deconv"
    ])
    
    # Cell Types for Single-Cell Specificity
    cell_types: List[str] = field(default_factory=lambda: [
        "Glutamatergic_Neuron",
        "GABAergic_Interneuron",
        "Oligodendrocyte",
        "Astrocytes",
        "Microglia"
    ])
    
    # Sample Design
    n_animals: int = 24  # Biological replicates
    replicates_per_condition: int = 6
    n_genes: int = 120
    
    # Key neuroplasticity & epigenetic regulator genes
    core_plasticity_genes: List[str] = field(default_factory=lambda: [
        "BDNF", "FOXP2", "EGR1", "FOS", "ARC", "NPAS4", 
        "DNMT3A", "TET2", "KDM6A", "EZH2", "CREB1", "HDAC2",
        "CAMK2A", "GRIN2B", "SYP", "SYN1", "MEF2C", "KMT2A"
    ])
    
    # Machine Learning Settings
    cv_n_splits: int = 5
    random_state: int = 42
    
    # Paths
    raw_data_path: Path = DATA_DIR / "raw_simulated"
    qc_reports_path: Path = OUTPUT_DIR / "stage1_qc"
    feature_matrix_path: Path = OUTPUT_DIR / "stage2_features"
    models_path: Path = OUTPUT_DIR / "stage3_models"
    interpretation_path: Path = OUTPUT_DIR / "stage4_interpretation"

config = MultiOmicsConfig()
