import os
import sys
import platform
import json
import random
import numpy as np
import torch
import pandas as pd
import sklearn
from pathlib import Path
import warnings

def repository_root() -> Path:
    """
    Returns the absolute path to the repository root, 
    derived dynamically from this file's location.
    """
    # src/training/reproducibility.py -> src/training -> src -> repo_root
    return Path(__file__).resolve().parent.parent.parent

def set_global_seed(seed: int):
    """
    Seeds random, NumPy, PyTorch CPU, CUDA, and MPS.
    Enables deterministic algorithms where available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        warnings.warn(f"Could not enable deterministic algorithms: {e}")
        
    os.environ['PYTHONHASHSEED'] = str(seed)

def collect_environment_info() -> dict:
    """
    Collects system, Python, and package version information.
    """
    try:
        import rdkit
        rdkit_version = rdkit.__version__
    except ImportError:
        rdkit_version = "not installed"
        
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not installed"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # Minimal Git SHA check without crashing if not in git repo
    git_sha = "unknown"
    dirty = False
    try:
        import subprocess
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        dirty = len(status) > 0
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "device": device,
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit_version,
            "transformers": transformers_version
        },
        "git": {
            "sha": git_sha,
            "dirty": dirty
        }
    }
