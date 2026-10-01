"""
Cross-Species Multi-Omics Cohort Generator
Generates biologically realistic multi-omics data for BOTH:
  - Songbird (Zebra Finch) vocal learning cohorts
  - Human postmortem/iPSC-derived cortical tissue cohorts

Species-specific biological differences modeled:
  - Humans have ~15-25% higher baseline CpG methylation than songbirds
  - Human chromatin is generally more compact (higher H3K27me3 baselines)
  - Songbird FOXP2/vocal plasticity genes show stronger tutoring responses
  - Human samples show wider inter-individual variance (genetic diversity)
  - Cell-type proportions differ (human cortex has more astrocytes/oligodendrocytes)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from multiomics_pipeline.config import config

# ─── Species-Specific Biology Parameters ───
SPECIES_PROFILES = {
    "Zebra_Finch": {
        "label": "ZF",
        "id_prefix": "ZF",
        "description": "Songbird vocal learning model (Area X / HVC / RA nuclei)",
        "rna_baseline_shift": 0.0,        # reference
        "k4_baseline_shift": 0.0,
        "k27_baseline_shift": 0.0,
        "meth_baseline_shift": 0.0,       # reference methylation
        "inter_individual_noise": 1.0,     # reference noise
        "cell_type_profile": {             # songbird auditory forebrain
            "Glutamatergic_Neuron": 0.52,
            "GABAergic_Interneuron": 0.18,
            "Oligodendrocyte": 0.13,
            "Astrocytes": 0.10,
            "Microglia": 0.07
        },
        "foxp2_vocal_boost": 1.8,         # FOXP2 is critical for vocal learning in songbirds
        "bdnf_responsiveness": 1.0,
    },
    "Human": {
        "label": "HS",
        "id_prefix": "HS",
        "description": "Human cortical tissue (prefrontal cortex / Broca's area)",
        "rna_baseline_shift": 0.8,        # slightly higher baseline transcription
        "k4_baseline_shift": 0.3,
        "k27_baseline_shift": 1.2,        # more compact chromatin
        "meth_baseline_shift": 0.12,      # 12% higher CpG methylation
        "inter_individual_noise": 1.6,     # more genetic diversity
        "cell_type_profile": {             # human cortex proportions
            "Glutamatergic_Neuron": 0.44,
            "GABAergic_Interneuron": 0.15,
            "Oligodendrocyte": 0.18,
            "Astrocytes": 0.16,
            "Microglia": 0.07
        },
        "foxp2_vocal_boost": 0.6,         # FOXP2 is important but less dramatic in humans
        "bdnf_responsiveness": 1.3,       # BDNF response is strong in human cortex
    }
}

# ─── Gene dynamics (shared across species, magnitude differs) ───
def get_gene_dynamics(all_genes, seed=42):
    np.random.seed(seed + 999)
    dynamics = {}
    for g in all_genes:
        if g in ["BDNF", "EGR1", "FOS", "ARC", "NPAS4"]:
            dynamics[g] = "early_responder"
        elif g in ["FOXP2", "CAMK2A", "GRIN2B", "SYP", "SYN1", "MEF2C"]:
            dynamics[g] = "mastery_up"
        elif g in ["TET2", "KDM6A", "KMT2A"]:
            dynamics[g] = "epigenetic_activator"
        elif g in ["DNMT3A", "EZH2", "HDAC2"]:
            dynamics[g] = "repressive_remodeler"
        else:
            r = np.random.rand()
            if r < 0.10:
                dynamics[g] = "early_responder"
            elif r < 0.20:
                dynamics[g] = "mastery_up"
            elif r < 0.28:
                dynamics[g] = "mastery_down"
            else:
                dynamics[g] = "invariant"
    return dynamics


def generate_cross_species_cohort(
    reps_per_condition: int = 6,
    seed: int = 42,
    species_list=None
):
    """
    Generates a combined cross-species multi-omics cohort.
    Returns a DataFrame with identical column schema to the standard pipeline input.
    """
    if species_list is None:
        species_list = ["Zebra_Finch", "Human"]

    np.random.seed(seed)
    conditions = config.conditions
    background_genes = [f"GENE_{i:03d}" for i in range(1, config.n_genes - len(config.core_plasticity_genes) + 1)]
    all_genes = config.core_plasticity_genes + background_genes
    gene_dynamics = get_gene_dynamics(all_genes, seed)

    all_records = []
    global_animal_id = 1

    for species_name in species_list:
        sp = SPECIES_PROFILES[species_name]
        noise_scale = sp["inter_individual_noise"]

        for cond_idx, condition in enumerate(conditions):
            for rep in range(1, reps_per_condition + 1):
                sample_id = f"{sp['id_prefix']}_{global_animal_id:02d}_{condition[:4]}_R{rep}"
                animal_id = f"Animal_{global_animal_id:02d}"

                # Plasticity trajectory (0→1 across conditions)
                base_traj = cond_idx / (len(conditions) - 1)
                noisy_traj = float(np.clip(
                    base_traj + np.random.normal(0, 0.07 * noise_scale), 0.0, 1.0
                ))
                t = noisy_traj

                rec = {
                    "sample_id": sample_id,
                    "animal_id": animal_id,
                    "condition": condition,
                    "tutoring_hours": [0, 24, 168, 720][cond_idx],
                    "true_plasticity_score": round(noisy_traj, 4),
                }

                # ── Cell-type proportions ──
                ct_base = sp["cell_type_profile"]
                props = np.array([
                    ct_base["Glutamatergic_Neuron"] + 0.05 * t + np.random.normal(0, 0.02),
                    ct_base["GABAergic_Interneuron"] + np.random.normal(0, 0.015),
                    ct_base["Oligodendrocyte"] - 0.02 * t + np.random.normal(0, 0.01),
                    ct_base["Astrocytes"] + np.random.normal(0, 0.01),
                    ct_base["Microglia"] + np.random.normal(0, 0.008),
                ])
                props = np.clip(props, 0.01, 1.0)
                props /= props.sum()
                for idx, ct in enumerate(config.cell_types):
                    rec[f"scProp_{ct}"] = float(round(props[idx], 4))

                # ── Gene-level multi-omics ──
                for g in all_genes:
                    dyn = gene_dynamics[g]

                    base_rna = np.random.uniform(4.0, 9.0) + sp["rna_baseline_shift"]
                    base_k4 = np.random.uniform(2.5, 7.0) + sp["k4_baseline_shift"]
                    base_k27 = np.random.uniform(2.0, 6.0) + sp["k27_baseline_shift"]
                    base_meth = np.clip(
                        np.random.uniform(0.3, 0.8) + sp["meth_baseline_shift"], 0.05, 0.95
                    )

                    # Species-specific gene boosts
                    gene_species_factor = 1.0
                    if g == "FOXP2":
                        gene_species_factor = sp["foxp2_vocal_boost"]
                    elif g == "BDNF":
                        gene_species_factor = sp["bdnf_responsiveness"]

                    n = noise_scale  # inter-individual noise

                    if dyn == "early_responder":
                        effect = 2.8 * gene_species_factor * np.exp(-((t - 0.35)**2) / 0.08)
                        rna = base_rna + effect + np.random.normal(0, 0.3 * n)
                        k4 = base_k4 + 1.8 * effect + np.random.normal(0, 0.25 * n)
                        k27 = np.clip(base_k27 - 1.2 * effect + np.random.normal(0, 0.2 * n), 0.1, 15.0)
                        meth = np.clip(base_meth - 0.25 * (effect / 2.8) + np.random.normal(0, 0.03 * n), 0.02, 0.98)
                    elif dyn == "mastery_up":
                        effect = 2.5 * gene_species_factor * t
                        rna = base_rna + effect + np.random.normal(0, 0.3 * n)
                        k4 = base_k4 + 2.0 * t * gene_species_factor + np.random.normal(0, 0.25 * n)
                        k27 = np.clip(base_k27 - 1.8 * t + np.random.normal(0, 0.2 * n), 0.1, 15.0)
                        meth = np.clip(base_meth - 0.32 * t + np.random.normal(0, 0.03 * n), 0.02, 0.98)
                    elif dyn == "mastery_down":
                        effect = 2.0 * t
                        rna = np.clip(base_rna - effect + np.random.normal(0, 0.3 * n), 0.5, 20.0)
                        k4 = np.clip(base_k4 - 1.5 * t + np.random.normal(0, 0.25 * n), 0.5, 15.0)
                        k27 = base_k27 + 2.0 * t + np.random.normal(0, 0.2 * n)
                        meth = np.clip(base_meth + 0.28 * t + np.random.normal(0, 0.03 * n), 0.02, 0.98)
                    elif dyn == "epigenetic_activator":
                        effect = 2.2 * np.sin(t * np.pi)
                        rna = base_rna + effect + np.random.normal(0, 0.25 * n)
                        k4 = base_k4 + 1.5 * effect + np.random.normal(0, 0.2 * n)
                        k27 = np.clip(base_k27 - 1.1 * effect + np.random.normal(0, 0.2 * n), 0.1, 15.0)
                        meth = np.clip(base_meth - 0.20 * effect + np.random.normal(0, 0.03 * n), 0.02, 0.98)
                    elif dyn == "repressive_remodeler":
                        effect = 1.8 * (1.0 - t)
                        rna = base_rna + effect + np.random.normal(0, 0.25 * n)
                        k4 = base_k4 + 1.2 * effect + np.random.normal(0, 0.2 * n)
                        k27 = base_k27 + 1.5 * t + np.random.normal(0, 0.2 * n)
                        meth = np.clip(base_meth + 0.15 * t + np.random.normal(0, 0.03 * n), 0.02, 0.98)
                    else:  # invariant
                        rna = base_rna + np.random.normal(0, 0.35 * n)
                        k4 = base_k4 + np.random.normal(0, 0.3 * n)
                        k27 = base_k27 + np.random.normal(0, 0.3 * n)
                        meth = np.clip(base_meth + np.random.normal(0, 0.04 * n), 0.05, 0.95)

                    rec[f"RNA_{g}"] = float(round(np.clip(rna, 0.1, 20.0), 3))
                    rec[f"ChIP_H3K4me3_{g}"] = float(round(np.clip(k4, 0.1, 15.0), 3))
                    rec[f"ChIP_H3K27me3_{g}"] = float(round(np.clip(k27, 0.1, 15.0), 3))
                    rec[f"WGBS_CpG_{g}"] = float(round(np.clip(meth, 0.01, 0.99), 4))

                all_records.append(rec)
                global_animal_id += 1

    df = pd.DataFrame(all_records)
    return df


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent.parent / "sample_inputs"
    output_dir.mkdir(exist_ok=True)

    # 1. Standard cross-species cohort (48 total: 24 songbird + 24 human)
    print("Generating cross-species cohort (48 samples: 24 Zebra Finch + 24 Human)...")
    df_48 = generate_cross_species_cohort(reps_per_condition=6, seed=42)
    path_48 = output_dir / "cross_species_48_zebrafinch_human.csv"
    df_48.to_csv(path_48, index=False)
    print(f"  -> Saved: {path_48}  ({len(df_48)} samples, {df_48.shape[1]} columns)")

    # 2. Larger cohort (80 total: 40 songbird + 40 human)
    print("Generating large cross-species cohort (80 samples: 40 ZF + 40 Human)...")
    df_80 = generate_cross_species_cohort(reps_per_condition=10, seed=123)
    path_80 = output_dir / "cross_species_80_large.csv"
    df_80.to_csv(path_80, index=False)
    print(f"  -> Saved: {path_80}  ({len(df_80)} samples, {df_80.shape[1]} columns)")

    # 3. Human-only cohort (32 subjects)
    print("Generating human-only cohort (32 samples)...")
    df_human = generate_cross_species_cohort(reps_per_condition=8, seed=77, species_list=["Human"])
    path_human = output_dir / "human_only_32_cortical.csv"
    df_human.to_csv(path_human, index=False)
    print(f"  -> Saved: {path_human}  ({len(df_human)} samples, {df_human.shape[1]} columns)")

    # Summary
    print("\n" + "=" * 60)
    print("CROSS-SPECIES DATASETS GENERATED SUCCESSFULLY")
    print("=" * 60)
    for p in [path_48, path_80, path_human]:
        print(f"  [FILE] {p.name}")
    print("\nUpload any of these via the dashboard or run:")
    print("  python run_pipeline.py --input sample_inputs/cross_species_48_zebrafinch_human.csv")
