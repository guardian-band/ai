from pathlib import Path


def test_cpu_inference_source_has_no_heavy_runtime_imports():
    source = Path("src/inference/cpu_pair_predictor.py").read_text()
    assert "transformers" not in source
    assert "torch_geometric" not in source
