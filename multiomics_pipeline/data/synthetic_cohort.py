"""
Synthetic Multi-Omics Cohort Generator
Simulates realistic biological replicates across tutoring conditions for:
- RNA-seq (transcript counts)
- ChIP-seq H3K4me3 (active promoter histone mark)
- ChIP-seq H3K27me3 (repressive polycomb histone mark)
- WGBS (Whole-Genome Bisulfite Sequencing DNA methylation %)
- Single-Cell Cell Type Deconvolution proportions
- Stage 1 Processing Metrics (FastQC, STAR, MACS3, Bismark)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from ..config import config

def generate_multiomics_cohort(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Generates biologically plausible multi-omics dataset across tutoring cohorts.
    """
    np.random.seed(seed)
    
    # 1. Animals / Biological Replicates
    samples = []
    animal_id = 1
    for cond_idx, condition in enumerate(config.conditions):
        for rep in range(1, config.replicates_per_condition + 1):
            sample_id = f"ANML_{animal_id:02d}_{condition[:4]}_R{rep}"
            # Plasticity progression score (ground truth continuous trajectory 0.0 -> 1.0)
            base_traj = cond_idx / (len(config.conditions) - 1)
            noisy_traj = np.clip(base_traj + np.random.normal(0, 0.07), 0.0, 1.0)
            
            samples.append({
                "sample_id": sample_id,
                "animal_id": f"Animal_{animal_id:02d}",
                "condition": condition,
                "replicate": rep,
                "tutoring_hours": [0, 24, 168, 720][cond_idx],
                "true_plasticity_score": float(np.round(noisy_traj, 4))
            })
            animal_id += 1
            
    meta_df = pd.DataFrame(samples)
    
    # 2. Genes Definition
    # Combine core known plasticity genes with additional background neuro-genes
    background_genes = [f"GENE_{i:03d}" for i in range(1, config.n_genes - len(config.core_plasticity_genes) + 1)]
    all_genes = config.core_plasticity_genes + background_genes
    
    # Gene profiles: define responsiveness to tutoring
    # Category 1: Immediate-Early / Learning Regulators (BDNF, EGR1, FOS, ARC, NPAS4) -> Spike early and plateau
    # Category 2: Epigenetic Plasticity Modulators (TET2, KDM6A - active demethylation & H3K27me3 demethylase; DNMT3A, EZH2)
    # Category 3: Synaptic Maturation (CAMK2A, GRIN2B, SYP, SYN1, FOXP2) -> Progressive rise with mastery
    # Category 4: Repressed non-neuronal or invariant background
    
    gene_dynamics = {}
    for g in all_genes:
        if g in ["BDNF", "EGR1", "FOS", "ARC", "NPAS4"]:
            gene_dynamics[g] = "early_responder"
        elif g in ["FOXP2", "CAMK2A", "GRIN2B", "SYP", "SYN1", "MEF2C"]:
            gene_dynamics[g] = "mastery_up"
        elif g in ["TET2", "KDM6A", "KMT2A"]:
            gene_dynamics[g] = "epigenetic_activator"
        elif g in ["DNMT3A", "EZH2", "HDAC2"]:
            gene_dynamics[g] = "repressive_remodeler"
        else:
            # 15% random responder, rest invariant
            r = np.random.rand()
            if r < 0.10:
                gene_dynamics[g] = "early_responder"
            elif r < 0.20:
                gene_dynamics[g] = "mastery_up"
            elif r < 0.28:
                gene_dynamics[g] = "mastery_down"
            else:
                gene_dynamics[g] = "invariant"

    # 3. Simulate Multi-Omic Layers
    # Matrix columns will be:
    # RNA_{gene}, H3K4me3_{gene}, H3K27me3_{gene}, WGBS_{gene}, scProp_{cell_type}
    feature_records = []
    
    for _, row in meta_df.iterrows():
        t = row["true_plasticity_score"]
        cond = row["condition"]
        rec = {
            "sample_id": row["sample_id"],
            "animal_id": row["animal_id"],
            "condition": row["condition"],
            "tutoring_hours": row["tutoring_hours"],
            "true_plasticity_score": row["true_plasticity_score"]
        }
        
        # Cell type proportions (deconvolution from scRNA-seq)
        # Neuronal activity shifts slightly microglia / astro activation and synaptogenesis
        neuron_prop = 0.52 + 0.05 * t + np.random.normal(0, 0.02)
        interneuron_prop = 0.18 + np.random.normal(0, 0.015)
        oligo_prop = 0.15 - 0.02 * t + np.random.normal(0, 0.01)
        astro_prop = 0.10 + np.random.normal(0, 0.01)
        micro_prop = 0.05 + np.random.normal(0, 0.008)
        
        # normalize to sum to 1.0
        props = np.array([neuron_prop, interneuron_prop, oligo_prop, astro_prop, micro_prop])
        props = np.clip(props, 0.01, 1.0)
        props /= props.sum()
        
        for idx, ct in enumerate(config.cell_types):
            rec[f"scProp_{ct}"] = float(np.round(props[idx], 4))
            
        # Gene-level multi-omics
        for g in all_genes:
            dyn = gene_dynamics[g]
            
            # Base baselines
            base_rna = np.random.uniform(4.0, 9.0)
            base_k4 = np.random.uniform(2.5, 7.0)
            base_k27 = np.random.uniform(2.0, 6.0)
            base_meth = np.random.uniform(0.3, 0.8) # % CpG methylation
            
            if dyn == "early_responder":
                # peaks at early (t ~ 0.33), then slightly declines or stays warm
                effect = 2.8 * np.exp(-((t - 0.35) ** 2) / 0.08)
                rna = base_rna + effect + np.random.normal(0, 0.3)
                k4 = base_k4 + 1.8 * effect + np.random.normal(0, 0.25)
                k27 = np.clip(base_k27 - 1.2 * effect + np.random.normal(0, 0.2), 0.1, 10.0)
                meth = np.clip(base_meth - 0.25 * (effect / 2.8) + np.random.normal(0, 0.03), 0.02, 0.98)
            elif dyn == "mastery_up":
                # ramps up monotonically with tutoring duration
                effect = 2.5 * t
                rna = base_rna + effect + np.random.normal(0, 0.3)
                k4 = base_k4 + 2.0 * t + np.random.normal(0, 0.25)
                k27 = np.clip(base_k27 - 1.8 * t + np.random.normal(0, 0.2), 0.1, 10.0)
                meth = np.clip(base_meth - 0.32 * t + np.random.normal(0, 0.03), 0.02, 0.98)
            elif dyn == "mastery_down":
                effect = 2.0 * t
                rna = np.clip(base_rna - effect + np.random.normal(0, 0.3), 0.5, 15.0)
                k4 = np.clip(base_k4 - 1.5 * t + np.random.normal(0, 0.25), 0.5, 10.0)
                k27 = base_k27 + 2.0 * t + np.random.normal(0, 0.2)
                meth = np.clip(base_meth + 0.28 * t + np.random.normal(0, 0.03), 0.02, 0.98)
            elif dyn == "epigenetic_activator":
                # demethylases and methyltransferases active early-to-mid
                effect = 2.2 * np.sin(t * np.pi)
                rna = base_rna + effect + np.random.normal(0, 0.25)
                k4 = base_k4 + 1.5 * effect + np.random.normal(0, 0.2)
                k27 = np.clip(base_k27 - 1.1 * effect + np.random.normal(0, 0.2), 0.1, 10.0)
                meth = np.clip(base_meth - 0.20 * effect + np.random.normal(0, 0.03), 0.02, 0.98)
            elif dyn == "repressive_remodeler":
                # dynamic remodeling
                effect = 1.8 * (1.0 - t)
                rna = base_rna + effect + np.random.normal(0, 0.25)
                k4 = base_k4 + 1.2 * effect + np.random.normal(0, 0.2)
                k27 = base_k27 + 1.5 * t + np.random.normal(0, 0.2)
                meth = np.clip(base_meth + 0.15 * t + np.random.normal(0, 0.03), 0.02, 0.98)
            else: # invariant
                rna = base_rna + np.random.normal(0, 0.35)
                k4 = base_k4 + np.random.normal(0, 0.3)
                k27 = base_k27 + np.random.normal(0, 0.3)
                meth = np.clip(base_meth + np.random.normal(0, 0.04), 0.05, 0.95)
                
            rec[f"RNA_{g}"] = float(np.round(np.clip(rna, 0.1, 20.0), 3))
            rec[f"ChIP_H3K4me3_{g}"] = float(np.round(np.clip(k4, 0.1, 15.0), 3))
            rec[f"ChIP_H3K27me3_{g}"] = float(np.round(np.clip(k27, 0.1, 15.0), 3))
            rec[f"WGBS_CpG_{g}"] = float(np.round(np.clip(meth, 0.01, 0.99), 4))
            
        feature_records.append(rec)
        
    full_df = pd.DataFrame(feature_records)
    
    # 4. Generate Stage 1 Quality Control & Processing Metrics (FastQC, STAR, MACS3, Bismark)
    qc_metrics = {}
    for sample in samples:
        sid = sample["sample_id"]
        # FastQC stats
        total_reads = int(np.random.normal(38_000_000, 2_500_000))
        mean_phred = float(np.round(np.random.normal(36.8, 0.8), 2))
        gc_pct = float(np.round(np.random.normal(48.5, 1.2), 2))
        dup_pct = float(np.round(np.random.normal(18.2, 2.1), 2))
        
        # STAR Alignment stats
        uniquely_mapped = float(np.round(np.random.normal(89.4, 1.6), 2))
        multi_mapped = float(np.round(np.random.normal(7.2, 0.9), 2))
        unmapped = float(np.round(100.0 - (uniquely_mapped + multi_mapped), 2))
        
        # MACS3 ChIP Peak calling stats
        h3k4me3_peaks = int(np.random.normal(24_500, 1_200))
        h3k27me3_peaks = int(np.random.normal(38_200, 2_100))
        frip_score = float(np.round(np.random.normal(0.68, 0.04), 3)) # Fraction of Reads in Peaks
        
        # Bismark WGBS Methylation stats
        cpg_coverage = float(np.round(np.random.normal(28.5, 2.2), 1))
        bisulfite_conversion_eff = float(np.round(np.random.normal(99.62, 0.12), 2))
        global_cpg_meth = float(np.round(np.random.normal(68.4, 1.8), 2))
        
        qc_metrics[sid] = {
            "fastqc": {
                "total_reads": total_reads,
                "mean_phred_qscore": mean_phred,
                "gc_content_pct": gc_pct,
                "sequence_duplication_pct": dup_pct,
                "status": "PASS" if mean_phred >= 30 else "WARN"
            },
            "alignment_star": {
                "uniquely_mapped_pct": uniquely_mapped,
                "multi_mapped_pct": multi_mapped,
                "unmapped_pct": unmapped,
                "status": "PASS" if uniquely_mapped >= 80 else "WARN"
            },
            "macs3_peaks": {
                "h3k4me3_narrow_peaks": h3k4me3_peaks,
                "h3k27me3_broad_peaks": h3k27me3_peaks,
                "frip_score": frip_score,
                "status": "PASS" if frip_score >= 0.50 else "WARN"
            },
            "bismark_wgbs": {
                "cpg_coverage_depth": f"{cpg_coverage}x",
                "bisulfite_conversion_pct": bisulfite_conversion_eff,
                "global_cpg_methylation_pct": global_cpg_meth,
                "status": "PASS" if bisulfite_conversion_eff >= 99.0 else "WARN"
            }
        }
        
    return full_df, qc_metrics, gene_dynamics

if __name__ == "__main__":
    df, qc, dynamics = generate_multiomics_cohort()
    print(f"Generated cohort with {len(df)} samples and {df.shape[1]} features.")
