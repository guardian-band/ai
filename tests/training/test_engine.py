import pytest
import os
import json
import tempfile
from src.training.engine import StateGuardedTrainer, TrainerState, verify_manifest

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
