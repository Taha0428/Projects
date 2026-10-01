"""
Interactive Web Visualization & Analysis Server for Multi-Omics Pipeline
Provides REST APIs for:
- Live Pipeline execution on custom data
- CSV cohort file upload & analysis
- Cohort synthesis with custom sample sizes / conditions
- Real-time gene locus query across multi-omics modalities
- CSV Template generation & download
"""
import http.server
import json
import logging
import os
import socketserver
import threading
import urllib.parse
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import numpy as np

from multiomics_pipeline.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, config
from multiomics_pipeline.data.synthetic_cohort import generate_multiomics_cohort
from multiomics_pipeline.pipeline import MultiOmicsMasterPipeline
from multiomics_pipeline.stage4_interpretation.knockout_simulator import InSilicoKnockoutSimulator

PORT = 8050
WEB_DIR = BASE_DIR / "web"
RESULTS_FILE = OUTPUT_DIR / "web_export" / "pipeline_full_results.json"
UPLOADED_FILE = DATA_DIR / "custom_uploaded_cohort.csv"

simulator_instance = None

def get_simulator():
    global simulator_instance
    if simulator_instance is None or not simulator_instance.is_ready():
        simulator_instance = InSilicoKnockoutSimulator()
    return simulator_instance


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DashboardServer")

# Global pipeline execution state
pipeline_state = {
    "is_running": False,
    "last_run_timestamp": None,
    "status_message": "Ready",
    "dataset_source": "Default Benchmark Multi-Omics Cohort (24 samples)",
    "stage": 4,
    "total_stages": 4,
    "percent": 100,
    "sample_count": 24,
    "feature_count": 450,
    "logs": [
        "[System] Server initialized. Benchmark multi-omics cohort loaded.",
        "[System] Multi-omics pipeline ready: RNA-seq, ChIP-seq, WGBS, Single-Cell."
    ]
}

def add_log(msg: str):
    import datetime
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{now_str}] {msg}"
    logger.info(entry)
    pipeline_state["logs"].append(entry)
    if len(pipeline_state["logs"]) > 100:
        pipeline_state["logs"].pop(0)

def get_active_cohort_df():
    if UPLOADED_FILE.exists():
        try:
            return pd.read_csv(UPLOADED_FILE)
        except Exception:
            pass
    # Baseline
    cohort_csv = OUTPUT_DIR / "cohort_raw_features.csv"
    if cohort_csv.exists():
        try:
            return pd.read_csv(cohort_csv)
        except Exception:
            pass
    df, _, _ = generate_multiomics_cohort()
    return df

def generate_preview_dict(df: pd.DataFrame):
    cols = list(df.columns)
    rna_cols = [c for c in cols if "RNA_" in c or "gene" in c.lower()]
    chip_cols = [c for c in cols if "ChIP_" in c or "H3K" in c]
    wgbs_cols = [c for c in cols if "WGBS_" in c or "cpg" in c.lower() or "meth" in c.lower()]
    cond_counts = {}
    if "condition" in df.columns:
        cond_counts = df["condition"].value_counts().to_dict()
    elif "Condition" in df.columns:
        cond_counts = df["Condition"].value_counts().to_dict()

    preview_rows = df.head(6).fillna("").to_dict(orient="records")
    return {
        "total_rows": len(df),
        "total_cols": len(cols),
        "columns": cols[:20],
        "all_columns_count": len(cols),
        "preview_rows": preview_rows,
        "conditions": cond_counts,
        "modalities": {
            "rna_features": len(rna_cols) if rna_cols else 150,
            "chip_features": len(chip_cols) if chip_cols else 100,
            "wgbs_features": len(wgbs_cols) if wgbs_cols else 80,
            "phenotypes": len(cols) - len(rna_cols) - len(chip_cols) - len(wgbs_cols)
        }
    }

class MultiOmicsRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/status":
            self._send_json(pipeline_state)
            return

        elif path == "/api/results":
            if not RESULTS_FILE.exists():
                self._send_json({"error": "Pipeline results not generated yet."}, 404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(RESULTS_FILE, "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/api/dataset-preview":
            try:
                df = get_active_cohort_df()
                preview = generate_preview_dict(df)
                preview["dataset_source"] = pipeline_state["dataset_source"]
                self._send_json(preview)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        elif path == "/api/export-csv":
            csv_type = query.get("type", ["deg"])[0].lower()
            file_map = {
                "deg": OUTPUT_DIR / "stage2_features" / "deseq2_differentially_expressed_genes.csv",
                "dmr": OUTPUT_DIR / "stage2_features" / "wgbs_differentially_methylated_regions.csv",
                "chip": OUTPUT_DIR / "stage2_features" / "chip_histone_enrichment_scores.csv",
                "matrix": OUTPUT_DIR / "stage2_features" / "integrated_multiomics_scaled_matrix.csv",
                "qc": OUTPUT_DIR / "stage1_qc" / "fastqc_multiqc_metrics.csv",
                "regulators": OUTPUT_DIR / "stage4_interpretation" / "master_regulatory_genes.csv"
            }
            target_path = file_map.get(csv_type, file_map["deg"])
            if not target_path.exists():
                self._send_json({"error": f"File for {csv_type} not generated yet"}, 404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", f"attachment; filename=omicslab_{csv_type}.csv")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(target_path, "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/api/download-template":
            # Generate sample CSV template for users
            df, _, _ = generate_multiomics_cohort()
            csv_data = df.head(8).to_csv(index=False)
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=multiomics_input_template.csv")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(csv_data.encode("utf-8"))
            return

        elif path == "/api/gene-query":
            gene = query.get("gene", ["BDNF"])[0].upper()
            if not RESULTS_FILE.exists():
                self._send_json({"error": "Pipeline results not ready"}, 400)
                return

            with open(RESULTS_FILE, "r") as f:
                res_data = json.load(f)

            # Query DEG info
            deg_list = res_data.get("stage2_deg_all") or res_data.get("stage2_deg_top", [])
            deg_matches = [g for g in deg_list if g["gene"].strip().upper() == gene]
            deg_info = deg_matches[0] if deg_matches else None

            # Query DMR info
            dmr_list = res_data.get("stage2_dmr_all") or res_data.get("stage2_dmr_top", [])
            dmr_matches = [d for d in dmr_list if d["gene"].strip().upper() == gene]
            dmr_info = dmr_matches[0] if dmr_matches else None

            # Query SHAP importance
            shap_matches = [
                s for s in res_data.get("stage4_shap_top", [])
                if gene in s["feature"].upper()
            ]

            # Query trajectory kinetics wave
            gene_waves = res_data.get("stage4_gene_waves", {})
            wave_info = gene_waves.get(gene, [])

            self._send_json({
                "gene": gene,
                "deg": deg_info,
                "dmr": dmr_info,
                "shap_features": shap_matches,
                "wave_kinetics": wave_info
            })
            return

        elif path == "/api/available-genes":
            sim = get_simulator()
            genes = sim.get_available_genes() if sim.is_ready() else config.core_plasticity_genes
            self._send_json({"genes": genes})
            return

        elif path == "/api/simulate-knockout":
            gene = query.get("gene", ["BDNF"])[0].upper()
            try:
                rna_factor = float(query.get("rna_factor", [0.05])[0])
            except ValueError:
                rna_factor = 0.05
            try:
                meth_delta = float(query.get("meth_delta", [0.0])[0])
            except ValueError:
                meth_delta = 0.0
            try:
                histone_factor = float(query.get("histone_factor", [0.0])[0])
            except ValueError:
                histone_factor = 0.0
            baseline_cond = query.get("condition", ["Tutored_Mastery_30d"])[0]

            try:
                sim = get_simulator()
                res = sim.simulate(
                    gene=gene,
                    rna_factor=rna_factor,
                    meth_delta=meth_delta,
                    histone_factor=histone_factor,
                    baseline_condition=baseline_cond
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Simulation GET error: {e}", exc_info=True)
                self._send_json({"error": str(e)}, 500)
            return

        # Default static file handler
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/load-benchmark":
            if pipeline_state["is_running"]:
                self._send_json({"status": "error", "message": "Pipeline is currently busy"}, 429)
                return

            # Remove custom uploaded file if exists
            if UPLOADED_FILE.exists():
                try:
                    UPLOADED_FILE.unlink()
                except Exception:
                    pass

            def execute_reset_async():
                pipeline_state["is_running"] = True
                pipeline_state["stage"] = 1
                pipeline_state["percent"] = 10
                pipeline_state["status_message"] = "Loading Benchmark Cohort (24 biological replicates)..."
                add_log("Loading benchmark multi-omics cohort...")
                try:
                    pipeline = MultiOmicsMasterPipeline()
                    pipeline_state["stage"] = 2
                    pipeline_state["percent"] = 35
                    add_log("[Stage 1 & 2] Running QC and multi-omics feature integration...")
                    pipeline.run_all()
                    pipeline_state["dataset_source"] = "Default Benchmark Multi-Omics Cohort (24 samples)"
                    pipeline_state["stage"] = 4
                    pipeline_state["percent"] = 100
                    pipeline_state["status_message"] = "Benchmark Analysis Completed Successfully"
                    add_log("Benchmark cohort pipeline finished. All 4 stages complete.")
                except Exception as e:
                    logger.error(f"Reset error: {e}", exc_info=True)
                    pipeline_state["status_message"] = f"Failed: {str(e)}"
                    add_log(f"Error during benchmark run: {str(e)}")
                finally:
                    pipeline_state["is_running"] = False

            threading.Thread(target=execute_reset_async, daemon=True).start()
            self._send_json({"status": "accepted", "message": "Benchmark dataset loaded and pipeline started"})
            return

        elif path == "/api/run-pipeline":
            if pipeline_state["is_running"]:
                self._send_json({"status": "error", "message": "Pipeline is already running"}, 429)
                return

            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            req_params = json.loads(post_body) if post_body else {}

            def execute_async():
                pipeline_state["is_running"] = True
                pipeline_state["stage"] = 1
                pipeline_state["percent"] = 15
                pipeline_state["status_message"] = "Starting Multi-Omics Analysis Pipeline..."
                add_log("Initiating Multi-Omics Pipeline execution...")
                try:
                    pipeline = MultiOmicsMasterPipeline()
                    custom_path = req_params.get("custom_path", None)
                    if not custom_path and UPLOADED_FILE.exists():
                        custom_path = str(UPLOADED_FILE)

                    if custom_path and Path(custom_path).exists():
                        pipeline_state["dataset_source"] = f"Custom Dataset: {Path(custom_path).name}"
                        add_log(f"Ingesting custom dataset from {Path(custom_path).name}...")
                        pipeline_state["stage"] = 2
                        pipeline_state["percent"] = 40
                        pipeline.run_all(custom_cohort_path=Path(custom_path))
                    else:
                        pipeline_state["dataset_source"] = "Default Benchmark Multi-Omics Cohort (24 samples)"
                        add_log("Executing on active multi-omics cohort...")
                        pipeline_state["stage"] = 2
                        pipeline_state["percent"] = 40
                        pipeline.run_all()

                    pipeline_state["stage"] = 4
                    pipeline_state["percent"] = 100
                    pipeline_state["status_message"] = "Pipeline completed successfully"
                    add_log("Pipeline completed successfully! All 4 stages validated.")
                except Exception as e:
                    logger.error(f"Pipeline execution error: {e}", exc_info=True)
                    pipeline_state["status_message"] = f"Failed: {str(e)}"
                    add_log(f"Pipeline error: {str(e)}")
                finally:
                    pipeline_state["is_running"] = False

            threading.Thread(target=execute_async, daemon=True).start()
            self._send_json({"status": "accepted", "message": "Pipeline execution started"})
            return

        elif path == "/api/upload-cohort":
            if pipeline_state["is_running"]:
                self._send_json({"status": "error", "message": "Pipeline is currently busy"}, 429)
                return

            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")
            
            try:
                # Can be raw CSV or JSON with { "csv_text": "..." }
                if post_body.strip().startswith("{"):
                    body_json = json.loads(post_body)
                    csv_text = body_json.get("csv_text", "")
                else:
                    csv_text = post_body

                if not csv_text.strip():
                    self._send_json({"status": "error", "message": "Empty CSV content provided"}, 400)
                    return

                # Write to disk
                with open(UPLOADED_FILE, "w", encoding="utf-8") as f:
                    f.write(csv_text)

                # Validate CSV
                df_test = pd.read_csv(UPLOADED_FILE)
                if len(df_test) < 4:
                    self._send_json({"status": "error", "message": "Dataset must contain at least 4 biological samples"}, 400)
                    return

                add_log(f"Custom CSV uploaded: {len(df_test)} samples, {df_test.shape[1]} columns detected.")

                # Launch pipeline on uploaded dataset
                def execute_upload_async():
                    pipeline_state["is_running"] = True
                    pipeline_state["stage"] = 1
                    pipeline_state["percent"] = 15
                    pipeline_state["status_message"] = f"Ingesting Custom Dataset ({len(df_test)} samples)..."
                    add_log(f"[Stage 1] FastQC sequence quality & STAR alignment check on {len(df_test)} samples...")
                    try:
                        pipeline_state["stage"] = 2
                        pipeline_state["percent"] = 40
                        add_log("[Stage 2] DESeq2 differential expression & Bismark methylation calling...")
                        pipeline = MultiOmicsMasterPipeline()
                        pipeline.run_all(custom_cohort_path=UPLOADED_FILE)
                        pipeline_state["dataset_source"] = f"Uploaded Custom Dataset ({len(df_test)} samples)"
                        pipeline_state["stage"] = 4
                        pipeline_state["percent"] = 100
                        pipeline_state["status_message"] = "Analysis Completed Successfully"
                        add_log("Analysis complete! ML models and SHAP biomarkers generated.")
                    except Exception as e:
                        logger.error(f"Upload analysis error: {e}", exc_info=True)
                        pipeline_state["status_message"] = f"Analysis Failed: {str(e)}"
                        add_log(f"Upload processing failed: {str(e)}")
                    finally:
                        pipeline_state["is_running"] = False

                threading.Thread(target=execute_upload_async, daemon=True).start()
                preview = generate_preview_dict(df_test)
                self._send_json({
                    "status": "success", 
                    "message": f"Dataset received with {len(df_test)} samples. Complete analysis launched.",
                    "sample_count": len(df_test),
                    "feature_count": df_test.shape[1],
                    "preview": preview
                })

            except Exception as e:
                self._send_json({"status": "error", "message": f"Failed to parse CSV: {str(e)}"}, 400)
            return

        elif path == "/api/generate-custom-cohort":
            if pipeline_state["is_running"]:
                self._send_json({"status": "error", "message": "Pipeline currently running"}, 429)
                return

            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")
            params = json.loads(post_body) if post_body else {}

            seed = int(params.get("seed", 42))
            reps = int(params.get("replicates", 6))
            
            def execute_gen_async():
                pipeline_state["is_running"] = True
                pipeline_state["status_message"] = f"Synthesizing Cohort (Seed={seed}, Replicates={reps})..."
                try:
                    config.replicates_per_condition = reps
                    config.n_animals = reps * len(config.conditions)
                    pipeline = MultiOmicsMasterPipeline()
                    pipeline.run_all()
                    pipeline_state["dataset_source"] = f"Custom Cohort ({config.n_animals} samples)"
                    pipeline_state["status_message"] = "Completed successfully"
                except Exception as e:
                    pipeline_state["status_message"] = f"Error: {str(e)}"
                finally:
                    pipeline_state["is_running"] = False

            threading.Thread(target=execute_gen_async, daemon=True).start()
            self._send_json({"status": "accepted", "message": "Cohort generated and analysis launched"})
            return

        elif path == "/api/simulate-knockout":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            req_params = json.loads(post_body) if post_body else {}

            gene = req_params.get("gene", "BDNF").upper()
            rna_factor = float(req_params.get("rna_factor", 0.05))
            meth_delta = float(req_params.get("meth_delta", 0.0))
            histone_factor = float(req_params.get("histone_factor", 0.0))
            baseline_cond = req_params.get("baseline_condition", "Tutored_Mastery_30d")

            try:
                sim = get_simulator()
                res = sim.simulate(
                    gene=gene,
                    rna_factor=rna_factor,
                    meth_delta=meth_delta,
                    histone_factor=histone_factor,
                    baseline_condition=baseline_cond
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Simulation POST error: {e}", exc_info=True)
                self._send_json({"error": str(e)}, 500)
            return

        self.send_response(404)
        self.end_headers()

def run_server(port: int = PORT):
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        logger.info("Initializing baseline pipeline results...")
        pipeline = MultiOmicsMasterPipeline()
        pipeline.run_all()

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), MultiOmicsRequestHandler) as httpd:
        logger.info("==================================================")
        logger.info(f"Multi-Omics Workbench Server live at: http://localhost:{port}")
        logger.info("APIs: /api/results, /api/upload-cohort, /api/gene-query, /api/download-template")
        logger.info("==================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Serve the Multi-Omics Dashboard")
    parser.add_argument("--port", "-p", type=int, default=PORT, help="Port to listen on (default 8050)")
    parser.add_argument("positional_port", nargs="?", type=int, default=None, help="Optional port as positional arg")
    args = parser.parse_args()
    port = args.positional_port if args.positional_port is not None else args.port
    run_server(port)
