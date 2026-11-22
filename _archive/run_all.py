"""Run the whole pipeline in one command:  python run_all.py"""
import os, subprocess, sys

STEPS = ["data_generator.py", "prepare_data.py", "features.py", "train_model.py", "experiments.py", "detector.py", "reports.py"]
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
for step in STEPS:
    print(f"\n=== Running {step} ===")
    if subprocess.run([sys.executable, step]).returncode != 0:
        sys.exit(f"Stopped: {step} failed.")
print("\nAll done. Start the dashboard with:  streamlit run app.py")
