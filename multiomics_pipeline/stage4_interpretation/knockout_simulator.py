"""
Stage 4: In Silico Gene Knockout & Perturbation Simulation Engine
Simulates genetic knockouts, knockdowns, and epigenetic modifications,
predicting shifts in neuroplasticity trajectories and cellular learning states.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from ..config import config, OUTPUT_DIR

# Known regulatory connections in neuroplasticity
REGULATORY_CASCADE_NETWORK = {
    "BDNF": ["SYN1", "SYP", "ARC", "CAMK2A", "GRIN2B"],
    "FOXP2": ["BDNF", "MEF2C", "SYN1", "EGR1"],
    "CREB1": ["BDNF", "FOS", "ARC", "EGR1", "NPAS4"],
    "EGR1": ["ARC", "SYN1", "SYP"],
    "FOS": ["ARC", "BDNF", "NPAS4"],
    "TET2": ["BDNF", "FOXP2", "KDM6A"],
    "DNMT3A": ["BDNF", "EGR1", "ARC"],
    "EZH2": ["FOXP2", "BDNF", "MEF2C"],
    "KDM6A": ["BDNF", "SYN1", "GRIN2B"],
    "ARC": ["SYP", "GRIN2B"]
}

class InSilicoKnockoutSimulator:
    """
    Simulates in silico multi-omics perturbations (knockouts, overexpression, methylation shifts)
    using trained Stage 3 models and precomputed baseline profiles.
    """
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (OUTPUT_DIR / "stage3_models")
        self.features_dir = OUTPUT_DIR / "stage2_features"
        
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.feature_names = []
        self.raw_matrix = None
        self.metadata = None
        self.condition_averages = {}
        
        self._load_artifacts()

    def _load_artifacts(self):
        clf_path = self.models_dir / "best_classifier.joblib"
        reg_path = self.models_dir / "best_regressor.joblib"
        scaler_path = self.models_dir / "feature_scaler.joblib"
        raw_mat_path = self.features_dir / "integrated_multiomics_raw_matrix.csv"
        meta_path = self.features_dir / "cohort_metadata.csv"
        
        if clf_path.exists() and reg_path.exists():
            try:
                self.classifier = joblib.load(clf_path)
                self.regressor = joblib.load(reg_path)
            except Exception:
                pass
                
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception:
                pass
                
        if raw_mat_path.exists() and meta_path.exists():
            try:
                self.raw_matrix = pd.read_csv(raw_mat_path)
                self.metadata = pd.read_csv(meta_path)
                self.feature_names = list(self.raw_matrix.columns)
                
                # Compute condition average vectors
                for cond in config.conditions:
                    cond_indices = self.metadata[self.metadata["condition"] == cond].index
                    if len(cond_indices) > 0:
                        self.condition_averages[cond] = self.raw_matrix.iloc[cond_indices].mean(axis=0)
            except Exception:
                pass

    def is_ready(self) -> bool:
        return (
            self.classifier is not None and 
            self.regressor is not None and 
            self.scaler is not None and 
            len(self.condition_averages) > 0
        )

    def get_available_genes(self) -> List[str]:
        if not self.feature_names:
            return config.core_plasticity_genes
        genes = set()
        for col in self.feature_names:
            if col.startswith("RNA_"):
                genes.add(col.replace("RNA_", ""))
        # Sort core genes first
        sorted_genes = [g for g in config.core_plasticity_genes if g in genes]
        remaining = sorted([g for g in genes if g not in config.core_plasticity_genes])
        return sorted_genes + remaining

    def simulate(
        self,
        gene: str,
        rna_factor: float = 0.05,       # 0.0 = complete knockout, 1.0 = wild-type, 2.5 = overexpression
        meth_delta: float = 0.0,        # -0.5 = hypomethylation, 0.0 = normal, +0.5 = hypermethylation
        histone_factor: float = 0.0,    # -1.0 = repressive, 0.0 = baseline, +1.0 = active
        baseline_condition: str = "Tutored_Mastery_30d"
    ) -> Dict[str, Any]:
        """
        Executes in silico multi-omics perturbation simulation.
        """
        if not self.is_ready():
            self._load_artifacts()
            if not self.is_ready():
                raise RuntimeError("Knockout Simulator artifacts not yet trained or loaded.")
                
        gene = gene.strip().upper()
        if baseline_condition not in self.condition_averages:
            baseline_condition = "Tutored_Mastery_30d" if "Tutored_Mastery_30d" in self.condition_averages else list(self.condition_averages.keys())[0]

        # 1. Start with baseline raw vector
        base_raw = self.condition_averages[baseline_condition].copy()
        pert_raw = base_raw.copy()

        # 2. Perturb target gene omics features
        rna_col = f"RNA_{gene}"
        k4_col = f"ChIP_H3K4me3_{gene}"
        k27_col = f"ChIP_H3K27me3_{gene}"
        wgbs_col = f"WGBS_CpG_{gene}"
        epi_col = f"EpiPermissiveness_{gene}"

        if rna_col in pert_raw:
            pert_raw[rna_col] = max(0.0, float(pert_raw[rna_col]) * rna_factor)
        if k4_col in pert_raw:
            pert_raw[k4_col] = max(0.1, float(pert_raw[k4_col]) * (1.0 + histone_factor))
        if k27_col in pert_raw:
            pert_raw[k27_col] = max(0.1, float(pert_raw[k27_col]) * (1.0 - histone_factor * 0.5))
        if wgbs_col in pert_raw:
            pert_raw[wgbs_col] = float(np.clip(pert_raw[wgbs_col] + meth_delta, 0.01, 0.99))
        if epi_col in pert_raw:
            pert_raw[epi_col] = pert_raw[k4_col] / (pert_raw[k27_col] + pert_raw[wgbs_col] + 0.1)

        # 3. Simulate regulatory cascade on downstream targets
        cascade_effects = []
        downstream_targets = REGULATORY_CASCADE_NETWORK.get(gene, [])
        for target in downstream_targets:
            t_rna = f"RNA_{target}"
            if t_rna in pert_raw:
                # If primary gene is knocked down, secondary gene drops by ~25% of the primary delta
                primary_delta_ratio = rna_factor - 1.0
                cascade_ratio = 1.0 + (primary_delta_ratio * 0.35)
                pert_raw[t_rna] = max(0.0, float(pert_raw[t_rna]) * cascade_ratio)
                cascade_effects.append({
                    "gene": target,
                    "baseline_rna": round(float(base_raw[t_rna]), 2),
                    "perturbed_rna": round(float(pert_raw[t_rna]), 2),
                    "pct_change": round(float((cascade_ratio - 1.0) * 100), 1)
                })

        # 4. Standardize baseline and perturbed vectors
        X_base_scaled = pd.DataFrame(
            self.scaler.transform(pd.DataFrame([base_raw], columns=self.feature_names)),
            columns=self.feature_names
        )
        X_pert_scaled = pd.DataFrame(
            self.scaler.transform(pd.DataFrame([pert_raw], columns=self.feature_names)),
            columns=self.feature_names
        )

        # 5. Model Inference (Probabilities & Plasticity Continuous Score)
        cond_classes = config.conditions
        
        # Classifier Probabilities
        if hasattr(self.classifier, "predict_proba"):
            base_probs = self.classifier.predict_proba(X_base_scaled)[0]
            pert_probs = self.classifier.predict_proba(X_pert_scaled)[0]
        else:
            base_pred = self.classifier.predict(X_base_scaled)[0]
            pert_pred = self.classifier.predict(X_pert_scaled)[0]
            base_probs = np.zeros(len(cond_classes))
            base_probs[base_pred] = 1.0
            pert_probs = np.zeros(len(cond_classes))
            pert_probs[pert_pred] = 1.0

        # Regressor Continuous Plasticity Scores
        base_score = float(self.regressor.predict(X_base_scaled)[0])
        pert_score = float(self.regressor.predict(X_pert_scaled)[0])
        score_delta = round(pert_score - base_score, 4)
        score_pct_change = round(((pert_score - base_score) / (abs(base_score) + 1e-5)) * 100, 1)

        # Format probabilities
        base_prob_dict = {cond_classes[i]: round(float(base_probs[i]), 4) for i in range(len(cond_classes))}
        pert_prob_dict = {cond_classes[i]: round(float(pert_probs[i]), 4) for i in range(len(cond_classes))}
        delta_prob_dict = {cond_classes[i]: round(float(pert_probs[i] - base_probs[i]), 4) for i in range(len(cond_classes))}

        # Predicted Cellular Fate
        predicted_fate = cond_classes[int(np.argmax(pert_probs))]
        initial_fate = cond_classes[int(np.argmax(base_probs))]

        # Biological Interpretation
        narrative = self._generate_biological_narrative(
            gene=gene,
            rna_factor=rna_factor,
            meth_delta=meth_delta,
            histone_factor=histone_factor,
            initial_fate=initial_fate,
            predicted_fate=predicted_fate,
            score_delta=score_delta,
            mastery_delta=delta_prob_dict.get("Tutored_Mastery_30d", 0.0)
        )

        return {
            "gene": gene,
            "perturbation_parameters": {
                "rna_factor": rna_factor,
                "meth_delta": meth_delta,
                "histone_factor": histone_factor,
                "baseline_condition": baseline_condition
            },
            "initial_state": {
                "dominant_condition": initial_fate,
                "plasticity_score": round(base_score, 4),
                "probabilities": base_prob_dict
            },
            "perturbed_state": {
                "dominant_condition": predicted_fate,
                "plasticity_score": round(pert_score, 4),
                "probabilities": pert_prob_dict
            },
            "deltas": {
                "score_delta": score_delta,
                "score_pct_change": score_pct_change,
                "probabilities": delta_prob_dict
            },
            "downstream_cascade": cascade_effects,
            "mechanistic_insight": narrative
        }

    def _generate_biological_narrative(
        self,
        gene: str,
        rna_factor: float,
        meth_delta: float,
        histone_factor: float,
        initial_fate: str,
        predicted_fate: str,
        score_delta: float,
        mastery_delta: float
    ) -> str:
        pert_type = "Knockout/Knockdown" if rna_factor < 0.5 else ("Overexpression" if rna_factor > 1.5 else "Epigenetic Tuning")
        
        if rna_factor < 0.5:
            if mastery_delta < -0.15:
                impact = f"leads to a severe phenotypic collapse (-{abs(round(mastery_delta*100, 1))}% in Tutored_Mastery), reverting neural states back towards {predicted_fate}."
            else:
                impact = f"partially attenuates learning plasticity, causing an overall neuroplasticity score change of {score_delta:+.3f}."
            bio_role = f"{gene} is required as an active molecular gatekeeper; suppressing it deprives the synaptic machinery of essential neuroplastic remodeling."
        elif rna_factor > 1.5:
            if score_delta > 0.05 or mastery_delta > 0.10:
                impact = f"accelerates the transition towards advanced plasticity (+{round(mastery_delta*100, 1)}% in Mastery), elevating cellular readiness."
            else:
                impact = f"maintains robust plasticity index without ectopic cellular stress."
            bio_role = f"Elevating {gene} potentiates chromatin accessibility and sustains rapid transcription of synaptic effector genes."
        else:
            impact = f"induces a subtle epigenetic shift with net plasticity delta of {score_delta:+.3f}."
            bio_role = f"Epigenetic marks at the {gene} promoter control its baseline responsiveness during memory consolidation."

        return f"In silico simulation of {gene} ({pert_type}) {impact} {bio_role}"
