import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.apply_two_stage_decisions import (
    grid_for_path,
    load_decisions,
    synthesize_stage1_decision,
)


def _write_jsonl(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return Path(fh.name)


def _row(grid, pid, *, auto_drop, action, path="/p/Gx/predictions_metric.gpkg", review=False):
    return {
        "candidate_id": f"{grid}_pred{pid:06d}",
        "grid_id": grid,
        "pred_id": pid,
        "predictions_path": path,
        "auto_drop": auto_drop,
        "production_action": action,
        "requires_human_review": review,
    }


class ApplyTwoStageDecisionsTests(unittest.TestCase):
    def test_drop_and_keep_collected(self):
        p = _write_jsonl([
            _row("G1", 0, auto_drop=True, action="drop"),
            _row("G1", 1, auto_drop=False, action="keep"),
        ])
        decisions, conflicts, violations = load_decisions([p])
        self.assertEqual(len(decisions), 2)
        self.assertEqual(conflicts, [])
        self.assertEqual(violations, [])

    def test_cross_file_conflict_resolves_to_keep(self):
        # same prediction, one file says drop, the other keep -> conservative keep wins
        path = "/p/G1/predictions_metric.gpkg"
        a = _write_jsonl([_row("G1", 5, auto_drop=True, action="drop", path=path)])
        b = _write_jsonl([_row("G1", 5, auto_drop=False, action="keep", path=path)])
        decisions, conflicts, violations = load_decisions([a, b])
        key = (str(Path(path).resolve()), 5)
        self.assertIn(key, decisions)
        self.assertFalse(decisions[key]["auto_drop"])  # non-drop wins
        self.assertEqual(len(conflicts), 1)

    def test_integrity_violation_flagged(self):
        # auto_drop=true but action != drop is a fail-closed integrity violation
        p = _write_jsonl([_row("G1", 0, auto_drop=True, action="review")])
        _decisions, _conflicts, violations = load_decisions([p])
        self.assertEqual(len(violations), 1)
        self.assertIn("auto_drop=true", violations[0])

    def test_grid_for_path_prefers_grid_id_field_over_path(self):
        # reviewed-gpkg layout: parent.name is 'review', must use grid_id field
        dmap = {0: {"grid_id": "G0890", "pred_id": 0}}
        self.assertEqual(
            grid_for_path(dmap, "/r/G0890/review/G0890_reviewed.gpkg"), "G0890"
        )
        # fallback to path parsing when grid_id absent
        self.assertEqual(grid_for_path({0: {}}, "/r/run/G0816/predictions_metric.gpkg"), "G0816")


def _stage1_row(grid, pid, *, pv_present, label, quality_flag, path="/p/Gx/predictions_metric.gpkg"):
    """A stage-1-only multiscale row: pv_present/label but NO production fields."""
    return {
        "candidate_id": f"{grid}_pred{pid:06d}",
        "grid_id": grid,
        "pred_id": pid,
        "predictions_path": path,
        "pv_present": pv_present,
        "label": label,
        "quality_flag": quality_flag,
    }


class SynthesizeStage1Tests(unittest.TestCase):
    def test_not_pv_becomes_drop(self):
        rec = synthesize_stage1_decision(
            {"pv_present": False, "label": "not_pv", "quality_flag": "usable"}
        )
        self.assertEqual(rec["production_action"], "drop")
        self.assertIs(rec["auto_drop"], True)
        self.assertIs(rec["pv_present"], False)
        self.assertIs(rec["requires_human_review"], False)

    def test_pv_becomes_keep(self):
        rec = synthesize_stage1_decision(
            {"pv_present": True, "label": "pv", "quality_flag": "usable"}
        )
        self.assertEqual(rec["production_action"], "keep")
        self.assertIs(rec["auto_drop"], False)
        self.assertIs(rec["pv_present"], True)

    def test_abstain_never_drops(self):
        # The _as_bool(None)->False trap: a null pv_present must route to review, NOT drop.
        rec = synthesize_stage1_decision(
            {"pv_present": None, "label": "", "quality_flag": "unusable"}
        )
        self.assertEqual(rec["production_action"], "review")
        self.assertIs(rec["auto_drop"], False)
        self.assertIs(rec["requires_human_review"], True)
        self.assertIsNone(rec["pv_present"])

    def test_inconsistent_row_routes_to_review(self):
        # usable but label/pv_present disagree -> fail-closed to review, not drop
        rec = synthesize_stage1_decision(
            {"pv_present": None, "label": "not_pv", "quality_flag": "usable"}
        )
        self.assertEqual(rec["production_action"], "review")
        self.assertIs(rec["auto_drop"], False)


class Stage1AsDropsLoadTests(unittest.TestCase):
    def test_synthesizes_and_passes_failclosed(self):
        p = _write_jsonl([
            _stage1_row("G1", 0, pv_present=False, label="not_pv", quality_flag="usable"),
            _stage1_row("G1", 1, pv_present=True, label="pv", quality_flag="usable"),
            _stage1_row("G1", 2, pv_present=None, label="", quality_flag="unusable"),
        ])
        decisions, conflicts, violations = load_decisions([p], stage1_as_drops=True)
        self.assertEqual(len(decisions), 3)
        self.assertEqual(violations, [])  # synthesized rows satisfy the fail-closed gate
        drops = [r for r in decisions.values() if r["auto_drop"] is True]
        self.assertEqual(len(drops), 1)  # only the not_pv row

    def test_without_flag_stage1_rows_are_not_dropped(self):
        # Without --stage1-as-drops, a stage-1 row has no auto_drop -> never dropped.
        p = _write_jsonl([
            _stage1_row("G1", 0, pv_present=False, label="not_pv", quality_flag="usable"),
        ])
        decisions, _conflicts, violations = load_decisions([p])
        (rec,) = decisions.values()
        self.assertFalse(bool(rec.get("auto_drop")))
        self.assertEqual(violations, [])

    def test_malformed_existing_action_caught_by_gate(self):
        # A row that already has production fields but violates fail-closed must be flagged.
        p = _write_jsonl([
            _row("G1", 0, auto_drop=True, action="keep"),  # auto_drop+keep is illegal
        ])
        _decisions, _conflicts, violations = load_decisions([p], stage1_as_drops=True)
        self.assertTrue(violations)


if __name__ == "__main__":
    unittest.main()
