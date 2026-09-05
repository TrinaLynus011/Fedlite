import sys
import subprocess
import os

def run_command(command, description):
    print(f"\n==================================================")
    print(f" RUNNING PHASE: {description}")
    print(f" Command: {command}")
    print(f"==================================================")
    # We run using the virtual environment python interpreter to ensure correct packages are used
    venv_python = os.path.join(".venv", "Scripts", "python") if os.name == 'nt' else os.path.join(".venv", "bin", "python")
    
    # Replace python in command with venv_python
    cmd_list = command.split()
    if cmd_list[0] == "python":
        cmd_list[0] = venv_python
        
    try:
        result = subprocess.run(cmd_list, check=True)
        print(f"SUCCESS: {description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with return code {e.returncode}.")
        return False

def show_help():
    print("""
FedLite-Care Run Pipeline Manager

Usage:
  python run_pipeline.py <phase>

Available phases:
  audit         - Inspects the raw dataset and generates audit files
  preprocess    - Splits dataset and runs target encoding / scaling
  baseline      - Trains centralized models (LR, MLP, XGBoost)
  federated     - Runs federated simulations (IID and Non-IID label skew)
  privacy       - Evaluates local differential privacy tradeoffs
  compression   - Evaluates Top-K update sparsification ratios
  edge          - Runs edge deployment simulations and quantizes MLP
  explain       - Generates global and local SHAP explanation plots
  ablation      - Orchestrates the full ablation study matrix (A0-A7)
  report        - Performs subgroups analysis and compiles publication reports
  validate      - Evaluates research quality checklist and split integrity
  all           - Runs all phases end-to-end
""")

def main():
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)
        
    phase = sys.argv[1].lower()
    
    if phase == "audit":
        run_command("python src/data/audit.py", "Dataset Audit")
    elif phase == "preprocess":
        s1 = run_command("python src/data/split.py", "Patient-level Dataset Splitting")
        s2 = run_command("python src/preprocessing/pipeline.py", "Dataset Preprocessing Pipeline")
        s3 = run_command("python src/data/missingness_analysis.py", "Missingness Analysis & Plotting")
        s4 = run_command("python src/data/target_distribution_plot.py", "Target Class Distribution Plotting")
        if not (s1 and s2 and s3 and s4):
            sys.exit(1)
    elif phase == "baseline":
        if not run_command("python src/models/train_baselines.py", "Centralized Baselines Training"):
            sys.exit(1)
    elif phase == "federated":
        if not run_command("python src/federated/train_federated.py", "Federated Learning Simulations"):
            sys.exit(1)
    elif phase == "privacy":
        if not run_command("python src/privacy/train_privacy.py", "Differential Privacy Tradeoffs"):
            sys.exit(1)
    elif phase == "compression":
        if not run_command("python src/compression/train_compression.py", "Communication Compression"):
            sys.exit(1)
    elif phase == "edge":
        s1 = run_command("python src/quantization/quantizer.py", "INT8 Dynamic Quantization")
        s2 = run_command("python src/edge/benchmarker.py", "Edge Resource Benchmarking")
        if not (s1 and s2):
            sys.exit(1)
    elif phase == "explain":
        if not run_command("python src/explainability/shap_explainer.py", "SHAP Explainability Plots"):
            sys.exit(1)
    elif phase == "ablation":
        if not run_command("python src/evaluation/ablation.py", "Full Ablation Matrix Study"):
            sys.exit(1)
    elif phase == "report":
        s1 = run_command("python src/evaluation/subgroup_analysis.py", "Demographic Subgroups Analysis")
        s2 = run_command("python src/utils/evidence_generator.py", "Paper Evidence Assembling")
        if not (s1 and s2):
            sys.exit(1)
    elif phase == "validate":
        if not run_command("python src/evaluation/research_quality_report.py", "Research Quality Verification"):
            sys.exit(1)
    elif phase == "all":
        phases = [
            ("python src/data/audit.py", "Dataset Audit"),
            ("python src/data/split.py", "Patient-level Splitting"),
            ("python src/preprocessing/pipeline.py", "Preprocessing Pipeline"),
            ("python src/data/missingness_analysis.py", "Missingness Plotting"),
            ("python src/data/target_distribution_plot.py", "Target Distribution Plotting"),
            ("python src/models/train_baselines.py", "Centralized Baselines"),
            ("python src/federated/train_federated.py", "Federated Simulations"),
            ("python src/privacy/train_privacy.py", "Privacy Tradeoffs"),
            ("python src/compression/train_compression.py", "Communication Compression"),
            ("python src/quantization/quantizer.py", "INT8 Quantization"),
            ("python src/edge/benchmarker.py", "Edge Benchmarking"),
            ("python src/explainability/shap_explainer.py", "SHAP Explanations"),
            ("python src/evaluation/ablation.py", "Ablation Study Matrix"),
            ("python src/evaluation/subgroup_analysis.py", "Subgroups Analysis"),
            ("python src/utils/evidence_generator.py", "Evidence System & Report Generation"),
            ("python src/evaluation/research_quality_report.py", "Research Quality Verification")
        ]
        
        for cmd, desc in phases:
            if not run_command(cmd, desc):
                print(f"\nFATAL: Pipeline failed at phase: {desc}")
                sys.exit(1)
        print("\n==================================================")
        print(" SUCCESS: Entire FedLite-Care Pipeline completed.")
        print("==================================================")
    else:
        print(f"Unknown phase: {phase}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
