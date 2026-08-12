import os
import json
from enum import Enum
import pandas as pd

class TrainerState(Enum):
    INITIALIZED = 1
    TRAINING = 2
    MODEL_SELECTED = 3
    VALIDATION_FROZEN = 4
    TEST_EVALUATED = 5
    COMPLETE = 6

class StateGuardedTrainer:
    def __init__(self, run_dir: str):
        self.state = TrainerState.INITIALIZED
        self.run_dir = run_dir
        
    def begin_training(self):
        assert self.state == TrainerState.INITIALIZED
        self.state = TrainerState.TRAINING
        
    def model_selected(self):
        assert self.state == TrainerState.TRAINING
        self.state = TrainerState.MODEL_SELECTED
        
    def validation_frozen(self):
        assert self.state == TrainerState.MODEL_SELECTED
        self.state = TrainerState.VALIDATION_FROZEN
        
    def evaluate_test(self):
        if self.state.value < TrainerState.VALIDATION_FROZEN.value:
            raise RuntimeError("Attempted to access test data before validation is frozen!")
        self.state = TrainerState.TEST_EVALUATED
        
    def complete(self, metrics: dict):
        assert self.state == TrainerState.TEST_EVALUATED
        
        # Write metrics
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Write completion marker
        with open(os.path.join(self.run_dir, "completion.json"), "w") as f:
            json.dump({"status": "complete"}, f)
            
        self.state = TrainerState.COMPLETE

def verify_manifest(manifest_path: str):
    """Verifies that the benchmark manifest hashes are consistent."""
    import hashlib
    with open(manifest_path, 'r') as f:
        data = json.load(f)
        
    claimed_hash = data.pop("manifest_hash", None)
    if not claimed_hash:
        raise ValueError("Manifest lacks a hash.")
        
    manifest_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
    actual_hash = hashlib.sha256(manifest_bytes).hexdigest()[:12]
    
    if claimed_hash != actual_hash:
        raise ValueError(f"Manifest hash mismatch: {claimed_hash} != {actual_hash}")
    
    data["manifest_hash"] = claimed_hash
    return data
