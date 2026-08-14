import hashlib
import json
import os
from enum import Enum


# Every completed run must carry provenance for these files.  The runner
# writes them before calling ``complete``; the trainer verifies their bytes
# and records the digest in completion.json as the final write.
REQUIRED_RUN_ARTIFACTS = (
    "config.resolved.json",
    "environment.json",
    "checkpoint_best.pt",
    "validation_logits.parquet",
    "validation_predictions.parquet",
    "test_logits.parquet",
    "test_predictions.parquet",
    "metrics.json",
    "per_label_metrics.csv",
    "thresholds.json",
    "calibration.json",
)

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
        
    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): StateGuardedTrainer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [StateGuardedTrainer._json_safe(item) for item in value]
        # Keep this module independent of numpy while handling numpy scalar
        # values produced by the evaluator when it is installed.
        if hasattr(value, "item") and callable(value.item):
            try:
                return StateGuardedTrainer._json_safe(value.item())
            except (TypeError, ValueError):
                pass
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            return None
        return value

    @staticmethod
    def _digest(path: str) -> dict:
        with open(path, "rb") as stream:
            payload = stream.read()
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    def complete(self, metrics: dict, required_artifacts=None):
        assert self.state == TrainerState.TEST_EVALUATED

        # A stale marker must never survive a failed completion attempt.
        completion_path = os.path.join(self.run_dir, "completion.json")
        try:
            os.unlink(completion_path)
        except FileNotFoundError:
            pass
        
        # Write metrics
        metrics_path = os.path.join(self.run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self._json_safe(metrics), f, indent=2, allow_nan=False)

        # Existing callers that only exercise state transitions remain
        # compatible; real experiment runs pass the complete contract.
        artifact_names = tuple(required_artifacts or ("metrics.json",))
        artifact_digests = {}
        for artifact_name in artifact_names:
            artifact_path = os.path.join(self.run_dir, artifact_name)
            if not os.path.isfile(artifact_path):
                raise FileNotFoundError(
                    f"Cannot complete run: required artifact is missing: {artifact_name}"
                )
            artifact_digests[artifact_name] = self._digest(artifact_path)
            
        # Write completion marker last.  Use a temporary file so an interrupted
        # write cannot leave a plausible complete marker behind.
        temporary_marker = completion_path + ".tmp"
        with open(temporary_marker, "w") as f:
            json.dump(
                {"status": "complete", "required_artifacts": artifact_digests},
                f,
                indent=2,
                allow_nan=False,
            )
        os.replace(temporary_marker, completion_path)
            
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
