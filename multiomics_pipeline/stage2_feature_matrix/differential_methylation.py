"""
Stage 2: Differential Methylation Analysis (DMR Calling)
Identifies Differentially Methylated Regions across promoters and gene bodies from WGBS.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

class DifferentialMethylationAnalyzer:
    """
    Computes delta beta-values (delta_beta), variance across replicates,
    and identifies promoter-proximal DMRs with statistical significance.
    """
    def __init__(self, cohort_df: pd.DataFrame):
        self.df = cohort_df
        
    def run_dmr_analysis(
        self,
        condition_a: str = "Untutored_Isolate",
        condition_b: str = "Tutored_Mastery_30d"
    ) -> pd.DataFrame:
        group_a = self.df[self.df["condition"] == condition_a]
        group_b = self.df[self.df["condition"] == condition_b]
        
        wgbs_cols = [c for c in self.df.columns if c.startswith("WGBS_CpG_")]
        results = []
        
        for col in wgbs_cols:
            gene = col.replace("WGBS_CpG_", "")
            vals_a = group_a[col].values
            vals_b = group_b[col].values
            
            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))
            delta_beta = float(mean_b - mean_a)
            
            # Welch's t-test for methylation proportion shift
            stat, p_val = stats.ttest_ind(vals_b, vals_a, equal_var=False)
            if np.isnan(p_val) or p_val == 0:
                p_val = 1e-10
                
            results.append({
                "gene": gene,
                "region": f"Promoter_CpG_Island_{gene}",
                "mean_beta_isolate": round(mean_a, 4),
                "mean_beta_mastery": round(mean_b, 4),
                "delta_beta": round(delta_beta, 4),
                "t_stat": round(float(stat), 3),
                "pvalue": float(p_val)
            })
            
        dmr_df = pd.DataFrame(results)
        
        # Multiple testing correction
        dmr_df = dmr_df.sort_values("pvalue").reset_index(drop=True)
        m = len(dmr_df)
        dmr_df["fdr"] = np.minimum.accumulate((dmr_df["pvalue"] * m / (np.arange(1, m + 1)))[::-1])[::-1]
        dmr_df["fdr"] = np.clip(dmr_df["fdr"], 0, 1.0)
        
        # Biological significance threshold (|delta_beta| >= 0.15 and fdr < 0.05)
        dmr_df["is_dmr"] = (dmr_df["fdr"] < 0.05) & (dmr_df["delta_beta"].abs() >= 0.15)
        dmr_df["status"] = "No_Change"
        dmr_df.loc[(dmr_df["is_dmr"]) & (dmr_df["delta_beta"] < 0), "status"] = "Hypomethylated"
        dmr_df.loc[(dmr_df["is_dmr"]) & (dmr_df["delta_beta"] > 0), "status"] = "Hypermethylated"
        
        return dmr_df
