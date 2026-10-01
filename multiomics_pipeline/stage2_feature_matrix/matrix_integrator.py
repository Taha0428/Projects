"""
Stage 2: Feature Matrix Construction & Multi-Omics Integrator
Unifies RNA-seq, ChIP-seq (H3K4me3, H3K27me3), WGBS, and Single-Cell deconvolution
into an integrated, standardized machine learning feature matrix.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..config import config

@dataclass
class IntegratedMatrixBundle:
    raw_matrix: pd.DataFrame
    scaled_feature_matrix: pd.DataFrame
    feature_names: List[str]
    metadata: pd.DataFrame
    replicate_groups: np.ndarray
    target_labels: np.ndarray
    target_continuous: np.ndarray
    feature_modality_map: Dict[str, str]
    scaler: Any = None

class FeatureMatrixIntegrator:
    """
    Harmonizes multi-modal datasets and produces standardized modeling tensors.
    """
    def __init__(self, cohort_df: pd.DataFrame):
        self.df = cohort_df
        
    def build_integrated_matrix(self) -> IntegratedMatrixBundle:
        meta_cols = ["sample_id", "animal_id", "condition", "tutoring_hours", "true_plasticity_score"]
        meta_df = self.df[meta_cols].copy()
        
        feature_cols = [c for c in self.df.columns if c not in meta_cols]
        raw_feat_df = self.df[feature_cols].copy()
        
        # Add engineered multi-omic interaction features:
        # e.g., Epigenetic Activation Score = H3K4me3 / (H3K27me3 + WGBS)
        for g in config.core_plasticity_genes:
            rna_col = f"RNA_{g}"
            k4_col = f"ChIP_H3K4me3_{g}"
            k27_col = f"ChIP_H3K27me3_{g}"
            wgbs_col = f"WGBS_CpG_{g}"
            
            if all(c in raw_feat_df.columns for c in [rna_col, k4_col, k27_col, wgbs_col]):
                # Epigenetic Permissiveness Index
                epi_index = raw_feat_df[k4_col] / (raw_feat_df[k27_col] + raw_feat_df[wgbs_col] + 0.1)
                raw_feat_df[f"EpiPermissiveness_{g}"] = np.round(epi_index, 4)
                
        # Modality Map
        modality_map = {}
        for col in raw_feat_df.columns:
            if col.startswith("RNA_"):
                modality_map[col] = "Transcriptomics (RNA-seq)"
            elif col.startswith("ChIP_H3K4me3_"):
                modality_map[col] = "Epigenomics (H3K4me3 Active)"
            elif col.startswith("ChIP_H3K27me3_"):
                modality_map[col] = "Epigenomics (H3K27me3 Repressive)"
            elif col.startswith("WGBS_CpG_"):
                modality_map[col] = "DNA Methylation (WGBS)"
            elif col.startswith("scProp_"):
                modality_map[col] = "Single-Cell Cell Proportion"
            elif col.startswith("EpiPermissiveness_"):
                modality_map[col] = "Integrated Epigenetic Index"
            else:
                modality_map[col] = "Other"
                
        # Standardize features
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(raw_feat_df)
        scaled_feat_df = pd.DataFrame(scaled_vals, columns=raw_feat_df.columns)
        
        # Replicate groups for Group K-Fold (crucial for zero data leakage across biological animals)
        # Each animal is an independent biological entity
        replicate_groups = meta_df["animal_id"].values
        
        # Condition labels for multi-class classification
        cond_map = {c: i for i, c in enumerate(config.conditions)}
        target_labels = meta_df["condition"].map(cond_map).values
        target_continuous = meta_df["true_plasticity_score"].values
        
        return IntegratedMatrixBundle(
            raw_matrix=raw_feat_df,
            scaled_feature_matrix=scaled_feat_df,
            feature_names=list(raw_feat_df.columns),
            metadata=meta_df,
            replicate_groups=replicate_groups,
            target_labels=target_labels,
            target_continuous=target_continuous,
            feature_modality_map=modality_map,
            scaler=scaler
        )
