import os
import subprocess
import sys
from pathlib import Path

# Ensure package root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peek_future.env import IS_KAGGLE, DATA_ROOT, CKPT_ROOT, RESULTS_ROOT


def run_cmd(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    print("=" * 60)
    print("Setting up peek-future environment")
    print(f"IS_KAGGLE: {IS_KAGGLE}")
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"CKPT_ROOT: {CKPT_ROOT}")
    print(f"RESULTS_ROOT: {RESULTS_ROOT}")
    print("=" * 60)

    # 1. Install dependencies if needed
    if IS_KAGGLE:
        req_file = REPO_ROOT / "requirements.txt"
        if req_file.exists():
            print("Installing requirements on Kaggle via pip...")
            run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"])

    # 2. Check Torch & CUDA
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("Warning: PyTorch not installed yet.")

    # 3. Download spaCy model
    print("Checking spaCy transformer model (en_core_web_trf)...")
    try:
        import spacy
        try:
            spacy.load("en_core_web_trf")
            print("en_core_web_trf already installed.")
        except Exception:
            print("Downloading en_core_web_trf...")
            run_cmd([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
    except ImportError:
        print("Warning: spaCy not installed yet.")

    print("\nEnvironment setup check complete.")


if __name__ == "__main__":
    main()
