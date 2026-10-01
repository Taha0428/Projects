"""
Stage 2: Histone Enrichment & Bivalent Promoter Scoring (ChIP-seq H3K4me3 / H3K27me3)
Calculates promoter active marks (H3K4me3) vs polycomb repressive marks (H3K27me3).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

class HistoneEnrichmentAnalyzer:
    """
    Computes gene-locus specific histone enrichment ratios and bivalent chromatin states.
    """
    def __init__(self, cohort_df: pd.DataFrame):
        self.df = cohort_df
        
    def run_histone_scoring(self) -> pd.DataFrame:
        k4_cols = [c for c in self.df.columns if c.startswith("ChIP_H3K4me3_")]
        results = []
        
        for k4_col in k4_cols:
            gene = k4_col.replace("ChIP_H3K4me3_", "")
            k27_col = f"ChIP_H3K27me3_{gene}"
            
            k4_vals = self.df[k4_col].values
            k27_vals = self.df[k27_col].values
            
            # Group by condition
            for cond in self.df["condition"].unique():
                mask = self.df["condition"] == cond
                sub_k4 = float(np.mean(k4_vals[mask]))
                sub_k27 = float(np.mean(k27_vals[mask]))
                
                # Bivalence / Activation Ratio
                act_ratio = float(np.log2((sub_k4 + 0.1) / (sub_k27 + 0.1)))
                
                # Chromatin State
                if sub_k4 >= 5.0 and sub_k27 < 3.0:
                    state = "Active_Promoter"
                elif sub_k4 < 3.0 and sub_k27 >= 5.0:
                    state = "Repressed_Polycomb"
                elif sub_k4 >= 4.0 and sub_k27 >= 4.0:
                    state = "Poised_Bivalent"
                else:
                    state = "Low_Signal"
                    
                results.append({
                    "gene": gene,
                    "condition": cond,
                    "mean_h3k4me3": round(sub_k4, 3),
                    "mean_h3k27me3": round(sub_k27, 3),
                    "chromatin_activation_ratio": round(act_ratio, 3),
                    "chromatin_state": state
                })
                
        return pd.DataFrame(results)
