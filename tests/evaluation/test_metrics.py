import unittest
import numpy as np
from src.evaluation.metrics import (
    compute_macro_average_precision,
    compute_micro_average_precision,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_macro_auroc,
    compute_comprehensive_metrics
)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # 3 samples, 4 labels
        self.y_true = np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 0]
        ])
        self.y_pred = np.array([
            [0.9, 0.1, 0.8, 0.2],
            [0.2, 0.8, 0.7, 0.1],
            [0.85, 0.95, 0.1, 0.05]
        ])

    def test_macro_ap(self):
        macro_ap = compute_macro_average_precision(self.y_true, self.y_pred)
        self.assertGreater(macro_ap, 0.5)
        self.assertLessEqual(macro_ap, 1.0)

    def test_precision_at_k(self):
        p_at_2 = compute_precision_at_k(self.y_true, self.y_pred, k=2)
        self.assertGreaterEqual(p_at_2, 0.0)
        self.assertLessEqual(p_at_2, 1.0)

    def test_comprehensive_metrics(self):
        metrics = compute_comprehensive_metrics(self.y_true, self.y_pred, k=2)
        self.assertIn("macro_ap", metrics)
        self.assertIn("micro_ap", metrics)
        self.assertIn("precision_at_2", metrics)
        self.assertIn("macro_auroc", metrics)

if __name__ == "__main__":
    unittest.main()
