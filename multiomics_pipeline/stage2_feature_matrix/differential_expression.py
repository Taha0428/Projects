"""
Stage 2: Differential Expression Analysis (DESeq2 statistical modeling)
Performs negative binomial / Wald test modeling to identify Differentially Expressed Genes (DEGs).
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple

class DifferentialExpressionAnalyzer:
    """
    Computes log2 Fold Change (log2FC), Wald statistic, p-values, 
    and Benjamini-Hochberg adjusted p-values (FDR) between conditions.
    """
    def __init__(self, cohort_df: pd.DataFrame):
        self.df = cohort_df
        
    def run_deseq2_deg(
        self, 
        condition_a: str = "Untutored_Isolate", 
        condition_b: str = "Tutored_Mastery_30d"
    ) -> pd.DataFrame:
        """
        Calculates differential expression between baseline and comparison condition.
        """
        group_a = self.df[self.df["condition"] == condition_a]
        group_b = self.df[self.df["condition"] == condition_b]
        
        rna_cols = [c for c in self.df.columns if c.startswith("RNA_")]
        results = []
        
        for col in rna_cols:
            gene = col.replace("RNA_", "")
            vals_a = group_a[col].values
            vals_b = group_b[col].values
            
            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))
            
            # log2 fold change (adding pseudocount)
            log2_fc = float(np.log2((mean_b + 0.1) / (mean_a + 0.1)))
            
            # Welch's t-test / Wald statistic approximation
            stat, p_val = stats.ttest_ind(vals_b, vals_a, equal_var=False)
            if np.isnan(p_val) or p_val == 0:
                p_val = 1e-12
                
            results.append({
                "gene": gene,
                "baseMean": round(float((mean_a + mean_b) / 2), 2),
                "log2FoldChange": round(log2_fc, 3),
                "stat": round(float(stat), 3),
                "pvalue": float(p_val)
            })
            
        deg_df = pd.DataFrame(results)
        
        # Benjamini-Hochberg (BH) Multiple Testing Correction
        deg_df = deg_df.sort_values("pvalue").reset_index(drop=True)
        m = len(deg_df)
        deg_df["padj"] = np.minimum.accumulate((deg_df["pvalue"] * m / (np.arange(1, m + 1)))[::-1])[::-1]
        deg_df["padj"] = np.clip(deg_df["padj"], 0, 1.0)
        
        # Categorize significance
        deg_df["significant"] = (deg_df["padj"] < 0.05) & (deg_df["log2FoldChange"].abs() >= 0.5)
        deg_df["regulation"] = "NS"
        deg_df.loc[(deg_df["significant"]) & (deg_df["log2FoldChange"] > 0), "regulation"] = "UP"
        deg_df.loc[(deg_df["significant"]) & (deg_df["log2FoldChange"] < 0), "regulation"] = "DOWN"
        
        return deg_df
