"""Data-level fail-closed gate test for two-stage Gemini FP-review artifacts.

Imports the stdlib-only checker (no repo modules) and asserts 0 violations on
the two real prelaunch artifacts. Also exercises the negative-control path so a
regression that silently makes validate_row() permissive is caught.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "analysis" / "check_two_stage_failclosed.py"

REAL_ARTIFACTS = [
    REPO_ROOT
    / "data/analysis/gemini_review_calib/prelaunch/two_stage_failclosed.jsonl",
    REPO_ROOT / "data/analysis/gemini_review_calib/prelaunch/two_stage_jhb.jsonl",
]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_two_stage_failclosed", CHECKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


@pytest.mark.parametrize("artifact", REAL_ARTIFACTS, ids=lambda p: p.name)
def test_real_artifact_has_no_violations(artifact):
    assert artifact.exists(), "missing real artifact: %s" % artifact
    n_rows, violations = checker.validate_file(str(artifact))
    assert n_rows > 0, "artifact had no rows: %s" % artifact
    assert violations == [], "fail-closed violations in %s: %s" % (
        artifact.name,
        violations[:20],
    )


def test_negative_control_row_is_caught():
    # auto_drop=True paired with production_action='review' must be rejected.
    bad_row = {
        "candidate_id": "NEGCTL_pred000001",
        "production_action": "review",
        "auto_drop": True,
        "requires_human_review": True,
        "pv_present": None,
    }
    reasons = checker.validate_row(bad_row)
    assert reasons, "negative-control row was not flagged as a violation"


def test_required_fields_enforced():
    reasons = checker.validate_row({"candidate_id": "x"})
    assert any("missing required field" in r for r in reasons)
