"""Pure-function tests for the v2b -> v2c mask patch.

These lock the two rules the patch is allowed to apply: which grids count as
labelled for solar-thermal collectors, and which value-2 pixels become
background. They do not read the trainset.
"""
import numpy as np
import pytest

from scripts.training.patch_ct_seg_trainset_v2c import remap_mask, swh_labelled_grids


def test_swh_labelled_grid_id_is_parent_of_review_not_the_filename():
    # The gpkg is named for a different grid. The id is the directory that
    # contains ``review``, because that is how the FP-review tree is laid out.
    paths = ["/data/cape_town/v3c/G1855/review/G9999_reviewed.gpkg"]
    out = swh_labelled_grids(paths, collector_grids=[], all_grids=["G1855", "G9999", "G1"])
    assert out["G1855"] == {"swh_labelled": True, "reason": "fp_review"}
    assert out["G9999"]["swh_labelled"] is False
    assert out["G9999"]["reason"] == "unlabelled"
    assert out["G1"]["reason"] == "unlabelled"


def test_swh_labelled_uses_the_review_directory_closest_to_the_file():
    paths = ["/tmp/review/G7/extra/review/ZZ_reviewed.gpkg"]
    out = swh_labelled_grids(paths, [], ["extra", "G7"])
    assert out["extra"]["swh_labelled"] is True
    assert out["G7"]["swh_labelled"] is False


def test_swh_labelled_collectors_csv_any_mask_value_and_both_sources():
    # The caller passes every collectors.csv grid, including mask_value 255.
    # Presence is enough; this function does not re-read the table.
    only = swh_labelled_grids([], collector_grids=["G10"], all_grids=["G10", "G11"])
    assert only["G10"] == {"swh_labelled": True, "reason": "collectors"}
    assert only["G11"]["swh_labelled"] is False

    both = swh_labelled_grids(
        ["/x/G10/review/G10_reviewed.gpkg"],
        collector_grids=["G10"],
        all_grids=["G10"],
    )
    assert both["G10"] == {"swh_labelled": True, "reason": "fp_review+collectors"}


def test_swh_labelled_ignores_grids_outside_the_trainset_and_duplicate_paths():
    paths = [
        "/x/G2/review/a_reviewed.gpkg",
        "/x/G2/review/b_reviewed.gpkg",
        "/x/G77/review/a_reviewed.gpkg",
        "/x/G2/notreview/c_reviewed.gpkg",
    ]
    out = swh_labelled_grids(paths, collector_grids=["G88"], all_grids=["G2", "G2", "G1"])
    assert list(out) == ["G2", "G1"]
    assert out["G2"] == {"swh_labelled": True, "reason": "fp_review"}
    assert out["G1"]["swh_labelled"] is False


def test_remap_mask_demotes_only_non_collector_value_2():
    mask = np.array([
        [2, 2, 1],
        [255, 0, 2],
        [1, 2, 255],
    ], dtype=np.uint8)
    collector = np.array([
        [True, False, True],
        [True, False, False],
        [False, True, True],
    ])
    new, n_remapped = remap_mask(mask, collector)
    assert n_remapped == 2
    assert new.dtype == np.uint8
    assert new.tolist() == [
        [2, 0, 1],
        [255, 0, 0],
        [1, 2, 255],
    ]
    # The source array is not edited, and pixels inside a collector stay put
    # even when they are PV or ignore. No new 2 is created.
    assert mask[0, 1] == 2
    assert np.array_equal(new[collector], mask[collector])
    assert not ((new == 2) & (mask != 2)).any()
    assert int((new == 2).sum()) <= int((mask == 2).sum())
    assert int((mask == 2).sum()) - int((new == 2).sum()) == n_remapped


def test_remap_mask_accepts_zero_one_collector_and_leaves_other_values():
    mask = np.array([2, 2, 1, 255, 7], dtype=np.uint8)
    new, n_remapped = remap_mask(mask, np.array([1, 0, 1, 1, 0]))
    assert n_remapped == 1
    assert new.tolist() == [2, 0, 1, 255, 7]
    assert mask.tolist() == [2, 2, 1, 255, 7]


def test_remap_mask_without_a_collector_demotes_every_value_2():
    mask = np.array([2, 1, 0, 255, 2], dtype=np.uint8)
    new, n_remapped = remap_mask(mask, np.zeros(5, dtype=bool))
    assert n_remapped == 2
    assert new.tolist() == [0, 1, 0, 255, 0]
    assert np.array_equal(new[mask != 2], mask[mask != 2])


def test_remap_mask_does_not_paint_2_onto_a_collector_of_another_class():
    mask = np.array([1, 255, 0], dtype=np.uint8)
    new, n_remapped = remap_mask(mask, np.array([True, True, True]))
    assert n_remapped == 0
    assert new.tolist() == [1, 255, 0]


def test_remap_mask_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        remap_mask(np.zeros((2, 2), np.uint8), np.zeros((2, 3), dtype=bool))
