"""
Stage 4: Plasticity Trajectory Mapping across Tutoring Conditions
Maps high-dimensional multi-omics profiles into continuous latent developmental trajectories.
"""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from ..config import config

@dataclass
class PlasticityTrajectoryBundle:
    sample_coordinates: pd.DataFrame
    trajectory_curve: List[Dict[str, float]]
    explained_variance_ratio: List[float]
    condition_centroids: Dict[str, Dict[str, float]]
    gene_kinetics_waves: Dict[str, List[Dict[str, float]]]

class PlasticityTrajectoryMapper:
    """
    Constructs latent multi-omic manifold and pseudotemporal tutoring trajectories.
    """
    def __init__(self, X_scaled: pd.DataFrame, metadata: pd.DataFrame):
        self.X_scaled = X_scaled
        self.metadata = metadata
        
    def fit_trajectory(self) -> PlasticityTrajectoryBundle:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(self.X_scaled)
        
        # Ensure direction aligns with tutoring progression (PC1 positive correlation with tutoring hours)
        pc1_corr = np.corrcoef(coords[:, 0], self.metadata["tutoring_hours"])[0, 1]
        if pc1_corr < 0:
            coords[:, 0] = -coords[:, 0]
            
        coords_df = self.metadata.copy()
        coords_df["Latent_Dim1"] = np.round(coords[:, 0], 3)
        coords_df["Latent_Dim2"] = np.round(coords[:, 1], 3)
        
        # Calculate pseudotime from PC1 projection normalized 0.0 to 1.0
        p_min, p_max = coords_df["Latent_Dim1"].min(), coords_df["Latent_Dim1"].max()
        coords_df["Inferred_Pseudotime"] = np.round((coords_df["Latent_Dim1"] - p_min) / (p_max - p_min + 1e-8), 4)
        
        # Centroids per condition
        centroids = {}
        for cond in config.conditions:
            sub = coords_df[coords_df["condition"] == cond]
            centroids[cond] = {
                "dim1": float(round(sub["Latent_Dim1"].mean(), 3)),
                "dim2": float(round(sub["Latent_Dim2"].mean(), 3)),
                "mean_pseudotime": float(round(sub["Inferred_Pseudotime"].mean(), 3))
            }
            
        # Fit polynomial trajectory backbone curve
        # Sort conditions chronologically
        centroid_points = [
            (centroids[cond]["dim1"], centroids[cond]["dim2"]) 
            for cond in config.conditions
        ]
        
        dim1_vals = [p[0] for p in centroid_points]
        dim2_vals = [p[1] for p in centroid_points]
        
        poly = np.polyfit(dim1_vals, dim2_vals, deg=2)
        curve_dim1 = np.linspace(min(dim1_vals) - 0.5, max(dim1_vals) + 0.5, 40)
        curve_dim2 = np.polyval(poly, curve_dim1)
        
        trajectory_curve = [
            {"dim1": float(round(d1, 3)), "dim2": float(round(d2, 3))}
            for d1, d2 in zip(curve_dim1, curve_dim2)
        ]
        
        # Multi-omic gene kinetics waves along pseudotime
        # Track 6 hallmark genes across pseudotime bins
        hallmarks = ["BDNF", "FOXP2", "EGR1", "TET2", "DNMT3A", "CAMK2A"]
        sorted_meta = coords_df.sort_values("Inferred_Pseudotime")
        
        waves = {}
        for g in hallmarks:
            rna_col = f"RNA_{g}"
            k4_col = f"ChIP_H3K4me3_{g}"
            meth_col = f"WGBS_CpG_{g}"
            
            wave_points = []
            for _, row in sorted_meta.iterrows():
                pt = row["Inferred_Pseudotime"]
                s_id = row["sample_id"]
                # Look up original raw values
                wave_points.append({
                    "pseudotime": pt,
                    "condition": row["condition"],
                    "rna": float(round(self.X_scaled.loc[row.name, rna_col] if rna_col in self.X_scaled else 0.0, 3)),
                    "h3k4me3": float(round(self.X_scaled.loc[row.name, k4_col] if k4_col in self.X_scaled else 0.0, 3)),
                    "dna_methylation": float(round(self.X_scaled.loc[row.name, meth_col] if meth_col in self.X_scaled else 0.0, 3))
                })
            waves[g] = wave_points
            
        return PlasticityTrajectoryBundle(
            sample_coordinates=coords_df,
            trajectory_curve=trajectory_curve,
            explained_variance_ratio=[float(round(v, 4)) for v in pca.explained_variance_ratio_],
            condition_centroids=centroids,
            gene_kinetics_waves=waves
        )
