"""Spatial IoU matching primitives — the F1 main-judge logic.

Pure functions, no GPU, no module-level side effects (no matplotlib backend
switch, no `set_grid_context`). Extracted verbatim from the inline logic in
`detect_and_evaluate.py` (the historical home of `compute_iou` /
`iou_matching`) so that the 5+ analysis scripts that `by-name` import this
matcher can do so without dragging in the heavyweight pipeline import and its
module-level side effects.

`detect_and_evaluate.py` re-exports these via a one-line import shim, so every
existing `from detect_and_evaluate import iou_matching` keeps working with no
behavioural change. There is exactly ONE implementation — the shim never holds
a second copy (cf. the `spatial_nms` lesson where a duplicate implementation
was left behind on extraction).

Behaviour parity is pinned by `tests/eval/test_iou_matching.py`. The signatures,
defaults (`iou_threshold=0.3`, `merge_preds=True`), greedy IoU-descending match
order, and the `installation`-profile pred-side many-to-one merge semantics are
preserved exactly.
"""
from __future__ import annotations

import geopandas as gpd
from shapely.ops import unary_union


# ════════════════════════════════════════════════════════════════════════
# IoU 匹配与评估
# ════════════════════════════════════════════════════════════════════════
def compute_iou(geom_a, geom_b) -> float:
    """计算两个几何体的交并比 (IoU)"""
    try:
        if geom_a.is_empty or geom_b.is_empty:
            return 0.0
        intersection = geom_a.intersection(geom_b).area
        union = geom_a.area + geom_b.area - intersection
        if union == 0:
            return 0.0
        return intersection / union
    except Exception:
        return 0.0


def iou_matching(gt: gpd.GeoDataFrame,
                 pred: gpd.GeoDataFrame,
                 iou_threshold: float = 0.3,
                 merge_preds: bool = True,
                 return_match_details: bool = False,
                 ) -> dict:
    """
    基于空间 IoU 的匹配，支持两种模式：

    merge_preds=False（严格一对一模式）:
      对每个 GT 多边形，找到 IoU 最高的单个预测多边形匹配。

    merge_preds=True（多对一合并模式，默认）:
      对每个 GT 多边形，将所有与之相交的预测多边形 union 合并后再计算 IoU。
      适用于标注者用一个大多边形覆盖屋顶上多组面板，而检测器将其拆分为
      多个小多边形的情况。

    return_match_details=True 时额外返回 "match_details" 列表，每个元素为:
      {"gt_idx", "pred_indices", "iou", "intersection_area", "gt_area", "pred_area"}

    返回:
      {
        "tp": int, "fp": int, "fn": int,
        "precision": float, "recall": float, "f1": float,
        "matched_pred_indices": set,
        "matched_gt_indices": set,
        "iou_scores": list  # 每个 TP 对应的 IoU 值
      }
    """
    pred_sindex = pred.sindex  # 空间索引
    matched_pred = set()
    matched_gt = set()
    iou_scores = []
    match_details = [] if return_match_details else None

    if merge_preds:
        # ── 多对一合并模式 ────────────────────────────────────────────
        # 对每个 GT，找出所有与之相交的 pred，合并后计算 IoU
        gt_match_results = []  # (gt_idx, merged_iou, pred_indices_set, gt_geom, merged_pred_geom)

        for gt_idx, gt_row in gt.iterrows():
            gt_geom = gt_row.geometry
            # 空间索引粗筛
            candidate_idxs = list(pred_sindex.intersection(gt_geom.bounds))

            # 精筛：只保留真正相交的
            intersecting_idxs = []
            for pidx in candidate_idxs:
                pred_geom = pred.iloc[pidx].geometry
                try:
                    if gt_geom.intersects(pred_geom):
                        intersecting_idxs.append(pidx)
                except Exception:
                    continue

            if not intersecting_idxs:
                continue

            # 合并所有相交的预测多边形
            merged_pred_geom = unary_union(
                [pred.iloc[pidx].geometry for pidx in intersecting_idxs]
            )

            iou_val = compute_iou(gt_geom, merged_pred_geom)
            if iou_val >= iou_threshold:
                gt_match_results.append(
                    (gt_idx, iou_val, set(intersecting_idxs), gt_geom, merged_pred_geom)
                )

        # 按 IoU 降序处理（贪心），避免 pred 被重复分配
        gt_match_results.sort(key=lambda x: x[1], reverse=True)

        for gt_idx, iou_val, pidx_set, gt_geom, merged_pred_geom in gt_match_results:
            if gt_idx in matched_gt:
                continue
            # 检查是否有至少一个 pred 尚未被分配
            available = pidx_set - matched_pred
            if not available:
                continue
            matched_gt.add(gt_idx)
            matched_pred.update(pidx_set)  # 所有参与合并的 pred 都标记为已匹配
            iou_scores.append(iou_val)
            if return_match_details:
                try:
                    inter_area = gt_geom.intersection(merged_pred_geom).area
                except Exception:
                    inter_area = 0.0
                match_details.append({
                    "gt_idx": gt_idx,
                    "pred_indices": pidx_set,
                    "iou": iou_val,
                    "intersection_area": inter_area,
                    "gt_area": gt_geom.area,
                    "pred_area": merged_pred_geom.area,
                })

    else:
        # ── 严格一对一模式 ────────────────────────────────────────────
        candidate_pairs = []

        for gt_idx, gt_row in gt.iterrows():
            gt_geom = gt_row.geometry
            candidate_idxs = list(pred_sindex.intersection(gt_geom.bounds))
            for pred_idx in candidate_idxs:
                pred_geom = pred.iloc[pred_idx].geometry
                iou_val = compute_iou(gt_geom, pred_geom)
                if iou_val >= iou_threshold:
                    candidate_pairs.append((gt_idx, pred_idx, iou_val))

        candidate_pairs.sort(key=lambda x: x[2], reverse=True)

        for gt_idx, pred_idx, iou_val in candidate_pairs:
            if gt_idx not in matched_gt and pred_idx not in matched_pred:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                iou_scores.append(iou_val)
                if return_match_details:
                    gt_geom = gt.loc[gt_idx].geometry
                    pred_geom = pred.iloc[pred_idx].geometry
                    try:
                        inter_area = gt_geom.intersection(pred_geom).area
                    except Exception:
                        inter_area = 0.0
                    match_details.append({
                        "gt_idx": gt_idx,
                        "pred_indices": {pred_idx},
                        "iou": iou_val,
                        "intersection_area": inter_area,
                        "gt_area": gt_geom.area,
                        "pred_area": pred_geom.area,
                    })

    tp = len(matched_gt)
    fn = len(gt) - tp
    fp = len(pred) - len(matched_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_pred_indices": matched_pred,
        "matched_gt_indices": matched_gt,
        "iou_scores": iou_scores,
    }
    if return_match_details:
        result["match_details"] = match_details
    return result
