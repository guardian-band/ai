import numpy as np

from scripts.audit_cold_mpnn import _bootstrap_delta, _macro_ap


def test_macro_ap_and_bootstrap_report_are_deterministic():
    targets = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    baseline = np.asarray([[1, -1], [-1, 1], [0.5, -0.5], [-0.5, 0.5]], dtype=np.float32)
    fused = baseline + np.asarray([[0.2, -0.2], [-0.2, 0.2], [0.2, -0.2], [-0.2, 0.2]])
    assert _macro_ap(fused, targets) >= _macro_ap(baseline, targets)
    first = _bootstrap_delta(baseline, fused, targets, samples=20, seed=42)
    second = _bootstrap_delta(baseline, fused, targets, samples=20, seed=42)
    assert first == second
    assert 0 < first["samples_valid"] <= 20
