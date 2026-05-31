import unittest

from scripts.analysis.gemini_fp_review_two_stage import merge_decision, needs_stage2


def _stage1(pv_present, lookalike_type="none", quality_flag="usable"):
    return {
        "candidate_id": "G0001_pred000001",
        "pv_present": pv_present,
        "confidence": 0.95,
        "quality_flag": quality_flag,
        "label": "pv" if pv_present is True else "not_pv" if pv_present is False else "",
        "lookalike_type": lookalike_type,
        "reason": "stage1 reason",
        "gemini_error": "",
        "model": "gemini-3-flash-agent",
        "decision_source": "gemini_fp_review_multiscale",
    }


def _stage2(pv_present, quality_flag="usable"):
    return {
        "candidate_id": "G0001_pred000001",
        "pv_present": pv_present,
        "confidence": 0.9,
        "quality_flag": quality_flag,
        "label": "pv" if pv_present is True else "not_pv" if pv_present is False else "",
        "lookalike_type": "skylight",
        "reason": "stage2 reason",
        "gemini_error": "",
        "model": "gemini-3-flash-agent",
        "decision_source": "gemini_fp_review_skylight_stage2",
    }


class GeminiFPReviewTwoStageTests(unittest.TestCase):
    def test_stage1_pv_keeps_without_stage2(self):
        merged = merge_decision(_stage1(True))

        self.assertEqual(merged["production_action"], "keep")
        self.assertEqual(merged["production_decision_source"], "stage1_pv")
        self.assertFalse(merged["auto_drop"])
        self.assertFalse(merged["stage2_required"])
        self.assertTrue(merged["pv_present"])

    def test_stage1_non_skylight_not_pv_auto_drops(self):
        merged = merge_decision(_stage1(False, lookalike_type="water_heater"))

        self.assertEqual(merged["production_action"], "drop")
        self.assertEqual(merged["production_decision_source"], "stage1_not_pv_non_skylight")
        self.assertTrue(merged["auto_drop"])
        self.assertFalse(merged["stage2_required"])
        self.assertFalse(merged["pv_present"])

    def test_stage1_skylight_not_pv_requires_stage2(self):
        stage1 = _stage1(False, lookalike_type="skylight")

        self.assertTrue(needs_stage2(stage1))

        merged = merge_decision(stage1, _stage2(True))
        self.assertEqual(merged["production_action"], "keep")
        self.assertEqual(merged["production_decision_source"], "stage2_skylight_keep")
        self.assertTrue(merged["stage2_required"])
        self.assertTrue(merged["stage2_applied"])
        self.assertFalse(merged["auto_drop"])
        self.assertTrue(merged["pv_present"])
        self.assertEqual(merged["stage2_reason"], "stage2 reason")

    def test_missing_stage2_for_skylight_goes_to_review_not_auto_drop(self):
        merged = merge_decision(_stage1(False, lookalike_type="skylight"))

        self.assertEqual(merged["production_action"], "review")
        self.assertEqual(merged["production_decision_source"], "stage2_missing_or_abstain")
        self.assertTrue(merged["stage2_required"])
        self.assertFalse(merged["stage2_applied"])
        self.assertFalse(merged["auto_drop"])
        self.assertTrue(merged["requires_human_review"])
        self.assertIsNone(merged["pv_present"])

    def test_abstained_stage2_for_skylight_goes_to_review_not_auto_drop(self):
        merged = merge_decision(
            _stage1(False, lookalike_type="skylight"),
            _stage2(None, quality_flag="unusable"),
        )

        self.assertEqual(merged["production_action"], "review")
        self.assertEqual(merged["production_decision_source"], "stage2_missing_or_abstain")
        self.assertFalse(merged["stage2_applied"])
        self.assertFalse(merged["auto_drop"])
        self.assertTrue(merged["requires_human_review"])
        self.assertIsNone(merged["pv_present"])


if __name__ == "__main__":
    unittest.main()
