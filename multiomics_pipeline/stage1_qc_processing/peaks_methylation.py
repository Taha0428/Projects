"""
Stage 1: Peak & Methylation Calling (MACS3 & Bismark)
Processes ChIP-seq narrow/broad peaks (H3K4me3 / H3K27me3) and WGBS CpG methylation calls.
"""
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

@dataclass
class EpigeneticSummaryReport:
    mean_h3k4me3_peaks: int
    mean_h3k27me3_peaks: int
    mean_frip_score: float
    mean_cpg_methylation: float
    mean_conversion_efficiency: float
    summary_table: pd.DataFrame
    params: Dict[str, Any]

class PeakAndMethylationProcessor:
    """
    Simulates & processes MACS3 peak calls and Bismark WGBS extractors.
    """
    def __init__(self, qc_data: Dict[str, Any]):
        self.qc_data = qc_data
        
    def run_epigenetic_assessment(self) -> EpigeneticSummaryReport:
        rows = []
        for sample_id, assays in self.qc_data.items():
            mc = assays["macs3_peaks"]
            bm = assays["bismark_wgbs"]
            rows.append({
                "Sample_ID": sample_id,
                "H3K4me3_Peaks": mc["h3k4me3_narrow_peaks"],
                "H3K27me3_Peaks": mc["h3k27me3_broad_peaks"],
                "FRiP_Score": mc["frip_score"],
                "Bisulfite_Conv_Pct": bm["bisulfite_conversion_pct"],
                "Global_CpG_Meth_Pct": bm["global_cpg_methylation_pct"],
                "MACS3_Status": mc["status"],
                "Bismark_Status": bm["status"]
            })
            
        df = pd.DataFrame(rows)
        params = {
            "macs3_h3k4me3_call": "macs3 callpeak -f BAMPE -q 0.01 --keep-dup auto (narrow)",
            "macs3_h3k27me3_call": "macs3 callpeak -f BAMPE --broad --broad-cutoff 0.05",
            "bismark_extractor": "bismark_methylation_extractor --bedGraph --cytosine_report --comprehensive",
            "genome_size": "hs / 1.05e9 (avian song system adjusted)"
        }
        
        return EpigeneticSummaryReport(
            mean_h3k4me3_peaks=int(df["H3K4me3_Peaks"].mean()),
            mean_h3k27me3_peaks=int(df["H3K27me3_Peaks"].mean()),
            mean_frip_score=float(round(df["FRiP_Score"].mean(), 3)),
            mean_cpg_methylation=float(round(df["Global_CpG_Meth_Pct"].mean(), 2)),
            mean_conversion_efficiency=float(round(df["Bisulfite_Conv_Pct"].mean(), 2)),
            summary_table=df,
            params=params
        )
