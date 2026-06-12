"""Area-adaptive boundary ignore band (C-3(b) recipe lever).

Standalone module — deliberately **not** folded into
``core.training.boundary_aware_mask`` (Phase A proved that packing several
levers into one patch makes failures un-attributable; see
``docs/plans/2026-05-09-training-supervision-layering.md`` action 5, which names
this module path explicitly).

What this replaces
------------------
``train.py``'s ``_boundary_pixel_weights(mask, label_source, band_iters=2)``
builds a **fixed-width** boundary ring (``band_iters=2`` → ~4 px) for every
instance regardless of size, and assigns the band the source's ``boundary_w``
(R/S = 0.0 ignore, H = 1.0 supervise). This module keeps the same per-pixel
weight-map contract (1.0 outside the band; source-dependent inside) but makes
the band **width adapt to the instance area** per the action-5 spec:

    | target size          | band half-width |
    |----------------------|-----------------|
    | small                | 1 px            |
    | medium               | 2 px            |
    | large / S (sam_*)     | 3 px            |
    | R (reviewed_prediction) | band ignored, core still supervised |

The width is the number of 3x3 dilate/erode iterations, so the ring spans
roughly ``2 * width`` px straddling the polygon edge (same construction as the
legacy fixed-band code, just size-dependent ``iterations``).

Rationale (from the supervision plan): at GSD ~6.7 cm a 1–3 px ignore band
corresponds to ~0.07–0.20 m of real spatial slack — wide enough to absorb the
SAM/reviewed-prediction edge noise on big installations (whose absolute boundary
error grows with size) while not erasing the entire mask of a tiny sub-array.

R-class semantics ("band-ignore-core-supervised")
--------------------------------------------------
For R-type sources (``reviewed_prediction`` and the R-like batch/Gemini/SAM
sources whose ``boundary_w`` is 0.0) the band pixels get weight 0.0 (ignored)
while the **core foreground and background stay weight 1.0** — i.e. the model is
still told "there is / isn't a panel here", it is just not penalised for the
exact edge placement. This is the single un-ablated orthogonal lever Phase A's
post-mortem left open: Phase A's ignore only reached the *mask* loss while the
boundary band still contributed box + cls gradients; here the ignore is purely a
per-pixel **mask-BCE** weight, applied via the existing
``boundary_aware_mask.patched_maskrcnn_loss`` ``mask_pixel_weights`` channel — it
does NOT touch box or cls loss (those keep the full polygon).

Retrain gate (DO NOT MISJUDGE — read before scoring tomorrow's run)
-------------------------------------------------------------------
Per the plan (line 202-203) C-3(b) is a single-lever single-retrain change. Its
success criteria are **bulk_ratio / σ_Bw / area_F1** vs the unified_A baseline on
the locked JHB CBD25 clean_gt, scored in BOTH merge modes (pixel-or and
per-detection) with ``scripts/analysis/area_aggregate_eval.py`` + polygon-conf
sweep. It must **NOT book any polygon-F1 / polygon-recall gain** as the win — a
boundary ignore band is a boundary-quality lever, not a recall lever; if polygon
recall moves, that is noise or a confound, not the deliverable. Gate = area-side
improvement (lower σ_Bw / RMSE, bulk in [0.5, 2.0]) with no area_F1 regression.

This module is CPU-importable (numpy + cv2 only, no torch) so the band geometry
is unit-testable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.boundary_trust import boundary_w_map


# Default area tiers, expressed in **mask-pixel area** (pixels inside the
# polygon, i.e. ``mask.sum()``). These are intentionally conservative round
# numbers; the pod operator can override via --boundary-ignore-band-thresholds.
#
# At the 400 px chip resolution this band is built on (GSD ~6.7 cm), 1 px ≈
# 0.067 m, so a 30x30 px panel (~900 px) ≈ 2 m × 2 m. The split below puts
# typical single sub-arrays in "medium" and large installations in "large".
DEFAULT_SMALL_MAX_AREA_PX = 400.0    # < 400 px  → small  → 1 px band
DEFAULT_MEDIUM_MAX_AREA_PX = 2500.0  # < 2500 px → medium → 2 px band
#                                      >= 2500 px → large  → 3 px band

# Band half-widths (3x3 dilate/erode iteration counts) per tier.
SMALL_BAND_ITERS = 1
MEDIUM_BAND_ITERS = 2
LARGE_BAND_ITERS = 3

# S-type sources always get the widest band (their masks are SAM-derived and
# carry the most edge noise irrespective of size). Mirrors the action-5 row
# "large target / S class → 3 px".
_S_TYPE_SOURCES = frozenset({
    "sam_refined_review",
    "sam_added_true_fn",
})


@dataclass(frozen=True)
class BandConfig:
    """Area-adaptive band configuration (all widths in 3x3 iteration counts)."""

    small_max_area_px: float = DEFAULT_SMALL_MAX_AREA_PX
    medium_max_area_px: float = DEFAULT_MEDIUM_MAX_AREA_PX
    small_iters: int = SMALL_BAND_ITERS
    medium_iters: int = MEDIUM_BAND_ITERS
    large_iters: int = LARGE_BAND_ITERS

    def tier_iters(self, area_px: float, label_source: str | None) -> int:
        """Return the band half-width (iterations) for one instance.

        S-type sources are forced to the large/widest band regardless of area
        (their edges are the noisiest). Otherwise the width is chosen by area:
        small < ``small_max_area_px`` ≤ medium < ``medium_max_area_px`` ≤ large.
        """
        if label_source in _S_TYPE_SOURCES:
            return self.large_iters
        if area_px < self.small_max_area_px:
            return self.small_iters
        if area_px < self.medium_max_area_px:
            return self.medium_iters
        return self.large_iters


def parse_band_thresholds(spec: str | None) -> BandConfig:
    """Parse a ``"small_max,medium_max"`` px-area threshold spec into a BandConfig.

    ``None`` → default thresholds. Example ``"400,2500"`` reproduces the
    defaults. Iteration widths (1/2/3) are fixed by the action-5 spec and are
    not part of the CLI spec.
    """
    if spec is None:
        return BandConfig()
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"--boundary-ignore-band-thresholds expects 'small_max,medium_max', "
            f"got {spec!r}"
        )
    small_max, medium_max = float(parts[0]), float(parts[1])
    if not (0 < small_max < medium_max):
        raise ValueError(
            f"thresholds must satisfy 0 < small_max < medium_max; got "
            f"small_max={small_max} medium_max={medium_max}"
        )
    return BandConfig(small_max_area_px=small_max, medium_max_area_px=medium_max)


def adaptive_boundary_pixel_weights(
    mask_np: np.ndarray,
    label_source: str | None,
    *,
    area_px: float | None = None,
    config: BandConfig | None = None,
    boundary_w_lookup: dict[str, float] | None = None,
) -> np.ndarray:
    """Per-pixel BCE weight map with an **area-adaptive** boundary ignore band.

    Drop-in replacement for ``train._boundary_pixel_weights`` with a size-aware
    band width. Same contract: returns a float32 map the shape of ``mask_np``,
    1.0 everywhere except inside the polygon-edge band, where the value is the
    source's ``boundary_w`` (H = 1.0 → no change; R/S = 0.0 → ignored).

    Args:
        mask_np: binary uint8 mask (H, W) for one instance.
        label_source: the annotation's ``label_source`` enum (None → full weight).
        area_px: foreground pixel area; if None it is computed from the mask.
        config: BandConfig (defaults if None).
        boundary_w_lookup: label_source → boundary_w (defaults to the project
            ``boundary_trust_rules.yaml`` map).

    R-class semantics: when ``boundary_w == 0.0`` (R/S sources) the band pixels
    are zeroed (ignored) and core fg/bg pixels stay 1.0 — boundary ignored, core
    supervised. When ``boundary_w == 1.0`` (H sources) the map is all-ones (early
    return; no band carved), so trusted edges keep full supervision.
    """
    lookup = boundary_w_lookup if boundary_w_lookup is not None else boundary_w_map()
    bw = lookup.get(label_source, 1.0)
    if bw == 1.0:
        # Trusted (H) or unknown-as-full source: no band carved, full supervision.
        return np.ones_like(mask_np, dtype=np.float32)

    cfg = config if config is not None else BandConfig()
    if area_px is None:
        area_px = float(mask_np.sum())
    iters = cfg.tier_iters(area_px, label_source)

    import cv2
    kernel = np.ones((3, 3), dtype=np.uint8)
    dil = cv2.dilate(mask_np, kernel, iterations=iters)
    ero = cv2.erode(mask_np, kernel, iterations=iters)
    band = (dil.astype(np.int8) ^ ero.astype(np.int8)) > 0
    out = np.ones_like(mask_np, dtype=np.float32)
    out[band] = bw
    return out
