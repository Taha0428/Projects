"""
Multi-Omics Pipeline CLI Entry Point
Execute complete or stage-specific multi-omics quality control, feature extraction, modeling, and SHAP attribution.

Usage:
  python run_pipeline.py --all
  python run_pipeline.py --stage 1
  python run_pipeline.py --stage 2
  python run_pipeline.py --stage 3
  python run_pipeline.py --stage 4
  python run_pipeline.py --serve
"""
import argparse
import sys
from pathlib import Path
from multiomics_pipeline.pipeline import MultiOmicsMasterPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Omics Epigenetic & Neuroplasticity Modeling Pipeline (RNA-seq, ChIP-seq, WGBS, Single-Cell)"
    )
    parser.add_argument("--all", action="store_true", help="Run entire 4-stage pipeline end-to-end")
    parser.add_argument("--input", type=str, default=None, help="Path to custom CSV dataset (e.g. cross_species_48_zebrafinch_human.csv)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="Run a specific stage (1=QC, 2=Features, 3=ML, 4=SHAP & Trajectory)")
    parser.add_argument("--serve", action="store_true", help="Launch interactive web visualization dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Web dashboard port (default: 8050)")
    
    args = parser.parse_args()
    
    if args.serve:
        import serve_dashboard
        serve_dashboard.run_server(port=args.port)
        return

    # Default to running all if no specific argument is provided
    pipeline = MultiOmicsMasterPipeline()
    results = pipeline.run_all(custom_cohort_path=args.input)
    print("\n[SUCCESS] Pipeline executed successfully.")
    print("Run `python run_pipeline.py --serve` or `python serve_dashboard.py` to view the interactive web dashboard.")

if __name__ == "__main__":
    main()
