import pytest
import os
import json
import tempfile
from src.training.engine import (
    REQUIRED_RUN_ARTIFACTS,
    StateGuardedTrainer,
    TrainerState,
    verify_manifest,
)

def test_trainer_state_guards():
    with tempfile.TemporaryDirectory() as td:
        trainer = StateGuardedTrainer(td)
        
        # Should raise error if accessing test data now
        with pytest.raises(RuntimeError, match="Attempted to access test data before validation is frozen!"):
            trainer.evaluate_test()
            
        trainer.begin_training()
        assert trainer.state == TrainerState.TRAINING
        
        trainer.model_selected()
        assert trainer.state == TrainerState.MODEL_SELECTED
        
        trainer.validation_frozen()
        assert trainer.state == TrainerState.VALIDATION_FROZEN
        
        # Now evaluate test should work
        trainer.evaluate_test()
        assert trainer.state == TrainerState.TEST_EVALUATED
        
        trainer.complete({"test_macro_ap": 0.5})
        assert trainer.state == TrainerState.COMPLETE
        
        assert os.path.exists(os.path.join(td, "metrics.json"))
        assert os.path.exists(os.path.join(td, "completion.json"))

def test_verify_manifest():
    import hashlib
    data = {"benchmark_id": "test", "seed": 42}
    raw = json.dumps(data, sort_keys=True).encode('utf-8')
    h = hashlib.sha256(raw).hexdigest()[:12]
    
    data["manifest_hash"] = h
    
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "manifest.json")
        with open(path, "w") as f:
            json.dump(data, f)
            
        # Should succeed
        verified = verify_manifest(path)
        assert verified["seed"] == 42
        
        # Tamper
        data["seed"] = 43
        with open(path, "w") as f:
            json.dump(data, f)
            
        with pytest.raises(ValueError, match="Manifest hash mismatch"):
            verify_manifest(path)


def test_completion_marker_proves_required_artifact_bytes():
    with tempfile.TemporaryDirectory() as td:
        for artifact in REQUIRED_RUN_ARTIFACTS:
            path = os.path.join(td, artifact)
            with open(path, "wb") as stream:
                stream.write(artifact.encode())
        trainer = StateGuardedTrainer(td)
        trainer.begin_training()
        trainer.model_selected()
        trainer.validation_frozen()
        trainer.evaluate_test()
        trainer.complete({"macro_ap": 0.5}, required_artifacts=REQUIRED_RUN_ARTIFACTS)

        with open(os.path.join(td, "completion.json")) as stream:
            marker = json.load(stream)
        assert marker["status"] == "complete"
        assert set(marker["required_artifacts"]) == set(REQUIRED_RUN_ARTIFACTS)
        assert marker["required_artifacts"]["metrics.json"]["bytes"] == os.path.getsize(
            os.path.join(td, "metrics.json")
        )


def test_missing_required_artifact_cannot_leave_complete_marker():
    with tempfile.TemporaryDirectory() as td:
        trainer = StateGuardedTrainer(td)
        trainer.begin_training()
        trainer.model_selected()
        trainer.validation_frozen()
        trainer.evaluate_test()
        with pytest.raises(FileNotFoundError, match="required artifact"):
            trainer.complete({"macro_ap": 0.5}, required_artifacts=REQUIRED_RUN_ARTIFACTS)
        assert not os.path.exists(os.path.join(td, "completion.json"))
