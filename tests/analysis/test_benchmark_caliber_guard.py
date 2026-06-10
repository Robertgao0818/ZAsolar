"""Cross-caliber diff refusal in run_benchmark (F1-gap Tier A1 / C12).

docs/evaluation_protocol.md §1.3: two numbers with different iou_caliber /
eval_profile must never be silently compared. Empty caliber (legacy results)
is allowed through for backward compatibility.
"""
import unittest

from scripts.analysis.run_benchmark import add_baseline_deltas, build_suite_summary


def _grid_row(model="m1", suite="s1", caliber="0.3", profile="installation",
              tp=5, fp=1, fn=2):
    return {
        "model_key": model, "model_tag": model, "suite_id": suite,
        "suite_role": "primary", "leakage_risk": "low", "region": "ct",
        "grid_id": "G0001", "output_subdir": "x",
        "gt_count": tp + fn, "pred_count": tp + fp,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "iou_caliber": caliber, "eval_profile": profile, "merge_mode": "",
    }


def _suite_row(model="m1", suite="s1", caliber="0.3", profile="installation"):
    return {
        "model_key": model, "model_tag": model, "suite_id": suite,
        "suite_role": "primary", "leakage_risk": "low",
        "grid_count": 1, "gt_count_total": 7,
        "tp_total": 5, "fp_total": 1, "fn_total": 2,
        "precision_micro": 0.83, "recall_micro": 0.71, "f1_micro": 0.77,
        "precision_macro": 0.83, "recall_macro": 0.71, "f1_macro": 0.77,
        "mean_iou_weighted": 0.6, "iou_ge_0.5_rate_weighted": 0.7,
        "iou_caliber": caliber, "eval_profile": profile,
    }


class CaliberGuardTests(unittest.TestCase):
    def test_mixed_caliber_diff_refused(self):
        rows = [
            _suite_row(model="base", caliber="0.1"),
            _suite_row(model="cand", caliber="0.3"),
        ]
        with self.assertRaisesRegex(RuntimeError, "CALIBER"):
            add_baseline_deltas(rows, baseline_key="base")

    def test_mixed_profile_diff_refused(self):
        rows = [
            _suite_row(model="base", profile="installation"),
            _suite_row(model="cand", profile="legacy_instance"),
        ]
        with self.assertRaisesRegex(RuntimeError, "CALIBER"):
            add_baseline_deltas(rows, baseline_key="base")

    def test_same_caliber_diff_allowed(self):
        rows = [
            _suite_row(model="base", caliber="0.3"),
            _suite_row(model="cand", caliber="0.3"),
        ]
        out = add_baseline_deltas(rows, baseline_key="base")
        self.assertIn("delta_f1_micro", out[1])

    def test_legacy_empty_caliber_allowed_through(self):
        rows = [
            _suite_row(model="base", caliber=""),
            _suite_row(model="cand", caliber="0.3"),
        ]
        out = add_baseline_deltas(rows, baseline_key="base")
        self.assertIn("delta_f1_micro", out[1])

    def test_mixed_caliber_within_suite_aggregation_refused(self):
        rows = [
            _grid_row(caliber="0.1"),
            _grid_row(caliber="0.3"),
        ]
        with self.assertRaisesRegex(RuntimeError, "CALIBER"):
            build_suite_summary(rows)

    def test_homogeneous_aggregation_carries_caliber(self):
        rows = [_grid_row(caliber="0.3"), _grid_row(caliber="0.3")]
        summ = build_suite_summary(rows)
        self.assertEqual(summ[0]["iou_caliber"], "0.3")
        self.assertEqual(summ[0]["eval_profile"], "installation")


if __name__ == "__main__":
    unittest.main()
