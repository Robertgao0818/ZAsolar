# Handoff — extract `core/polygon_validation.py` (deepening candidate #1)

- **Created**: 2026-06-19
- **Type**: architecture deepening (continuation of the ADR-0001 track)
- **Source**: 2026-06-19 architecture review (6-explorer + adversarial-verify workflow). This candidate was **1 of 2 survivors out of 26**; adversarial verdict `real=true, deletion_test_holds=true, overlaps_adr=none`, strength downgraded Strong→**Worth exploring**, and named **Top recommendation**.
- **For**: a fresh session. This thread is context-heavy; start clean and work from this file.
- **Status**: not started. Interface **not yet locked** — design happens first (grill or design-an-interface), then implement.

---

## 1. Goal in one line

Concentrate the **polygon geometry-validity + area-cap filtering pipeline** — currently re-derived across 8+ analysis scripts — behind one deep module `core/polygon_validation.py`, **byte-for-byte equivalent** to today's behaviour, so the appendix can cite one canonical definition of "valid polygon" at a commit SHA.

## 2. Why this one (and why it's safe)

- **Deletion test holds across *multiple real callers*** — unlike the 24 rejected candidates, which were single-caller (extraction just moves complexity) or already tracked in ADR-0001.
- **Zero locked-semantics risk** *if* done as a behaviour-preserving move: the geometry-filtering pipeline is **not** part of `docs/evaluation_protocol.md`'s locked scoring (that covers Tier-1 metrics, merge-mode, IoU caliber, working points). Filtering is upstream plumbing.
- **Paper / Data-appendix payoff (high)**: a reviewer's first question about an inventory number is *"exactly which polygons were included or excluded, and by what rule?"* Right now the answer is spread across 8 files with subtle drift (see §4). One module = one auditable, testable, citable answer.

## 3. The friction is bigger than the review reported

Grep on 2026-06-19 (`scripts/analysis/`) confirms the duplication is worse than the explorer's sample:

- **The magic constant `20_000.0 m²` is independently redefined in 8 files** under 5 different names: `_MAX_PLAUSIBLE_POLY_M2`, `MAX_PLAUSIBLE_POLY_M2`, `MAX_POLY_M2`, `MAX_PLAUSIBLE_M2`, `MAX_PLAUSIBLE`. (area_aggregate_eval, build_model_area_coverage_tracker, eval_xdomain60, installation_sym_eval, per_grid_dispersion_audit, gtnoise_t1_score, sseg_kw_calibration, + filter_sam_inventory's policy default.)
- **Two confusingly-named siblings exist**: `polygon_conf_sweep.py` *and* `poly_conf_sweep.py` — both import `_geometry_finite` / `_MAX_PLAUSIBLE_POLY_M2` from area_aggregate_eval. Untangle / document which is canonical (this is itself a provenance smell for the paper).
- A docstring hardcodes CT's CRS — `polygon_conf_sweep.py:4` says *"EPSG:32734 metric reproject"* — which won't generalise to other cities (violates rule 06: CRS must be looked up via `get_metric_crs`).

## 4. The exact contract to preserve (byte-equivalence target)

Canonical home today: `scripts/analysis/area_aggregate_eval.py`.

**`_geometry_finite(geom)`** (`area_aggregate_eval.py:62`):
```python
# rejects NaN/inf coords and |coord| > 1e18 via geom.bounds; bare-except → False
```

**Filtering order (must stay identical):**
1. `gdf.geometry.notna() & gdf.geometry.is_valid`
2. `gdf.geometry.apply(_geometry_finite)`
3. reproject **only if** `gdf.crs is None or str(gdf.crs) != metric_crs` → `to_crs(metric_crs)`
4. area cap `areas <= _MAX_PLAUSIBLE_POLY_M2`

**⚠ Known latent divergence — reconcile consciously, do not "clean up" silently:**
The two reference functions in the *same* file already disagree on zero-area polygons:
- `_sum_area_m2` (line 74): `keep_mask = areas <= MAX` — **keeps** `area == 0` polygons in the count.
- `_read_polys_geom` (line 108): `kept_mask = keep_mask & (areas > 0)` — **drops** `area == 0`.

The canonical module must either expose both behaviours explicitly or pick one and prove every migrated caller is unaffected. This is the single highest-risk point of the whole task.

**Current return tuples (callers depend on positional unpacking):**
- `_sum_area_m2` → `(n_kept, total_area_m2, max_poly_m2, n_dropped)`
- `_read_polys_geom` → `(n_kept, sum_area_m2, max_poly_m2, n_dropped, union_geom_or_None)`

## 5. Scope boundary — IN vs OUT (load-bearing)

**IN (the shared thing to extract):** geometry **validity** filtering — load gpkg (layer fallback to first), notna+valid, finite, reproject to metric CRS, area∈(0|≥0, 20000] cap, optional `unary_union`.

**OUT — must NOT be absorbed into the canonical module:**
- **Score / policy filters.** `filter_sam_inventory.py` mixes validity with *policy*: `refined_area_m2_max` (tunable), confidence min, SAM-score min. The adversarial verifier explicitly flagged this — `_filter_gdf` is policy-orchestration, not validity. Callers layer their own policy *on top of* canonical validity; the module owns validity only.
- **Tier-1 statistic formulas** (bootstrap CI, OLS R², σ_Bw, RMSE). Those are **ADR-0001 step 3** (already extracted to `core/area_metrics.py`) and **step 4** (pending: route `per_grid_dispersion_audit` + `poly_conf_sweep._agg` to `core.area_metrics`). Don't touch the stats — only the geometry loader feeding them.

## 6. Guardrails (read before writing code)

1. **Byte-equivalence gate (ADR-0001 全程铁律).** Every migrated call site must produce byte-identical output on a real run. Snapshot before → move → snapshot after → diff. No numeric-caliber change, ever.
2. **CRS policy** (`project_crs_policy` memory + rule 06): 4326 vector / native raster / UTM metric. `metric_crs` flows in from `get_metric_crs(grid_id, region=)` — **never hardcode** `EPSG:32734/5`. The new module takes `metric_crs` as a parameter; it does not resolve it.
3. **Move + import shim, never a second implementation** (ADR-0001 **D3**). Extract to `core/`, leave a re-export shim in `area_aggregate_eval.py` so existing imports (`from scripts.analysis.area_aggregate_eval import _read_polys_geom`) keep working — note `validate_checkpoint.py:62` and `lock_operating_point.py:170` import it, and `validate_checkpoint` is eval-protocol-sensitive.
4. **Don't collide with pending ADR-0001 step 4.** Step 4 edits `per_grid_dispersion_audit.py` and `poly_conf_sweep.py` for the *stats* path; this task edits them for the *geometry* path. Either sequence after step 4 or keep the file-level diffs disjoint. Flag in the commit.
5. **Doc sync (rule 03):** `docs/architecture.md` module table updated in the **same commit** as the file move. Update ADR-0001 with a new step/side-item.

## 7. Proposed interface — *starting point, settle in design*

Do **not** treat these as final. `build_model_area_coverage_tracker._clean_metric_gdf(gdf, *, assumed_crs, metric_crs) -> (gdf, n_dropped)` already has a good shape worth adopting.

```python
# core/polygon_validation.py  (PROPOSED)

def geometry_finite(geom) -> bool: ...

def validate_and_reproject_gdf(
    gdf, *, assumed_crs: str, metric_crs: str, max_area_m2: float = 20_000.0,
    drop_zero_area: bool,          # ← forces the §4 divergence to be explicit
) -> tuple[GeoDataFrame, int]:     # (cleaned, n_dropped)
    ...

def read_polygons(
    gpkg_path, *, metric_crs: str, layer: str | None = None,
    max_area_m2: float = 20_000.0, drop_zero_area: bool, with_union: bool,
) -> PolyReadResult:               # n_kept, sum_area_m2, max_poly_m2, n_dropped, union|None
    ...
```

## 8. Open design decisions (the grilling questions)

1. **`drop_zero_area`**: one flag, or two functions, or a frozen default? Whichever, every caller must be audited (§4).
2. **CRS reprojection — in or out of the module?** The rejected "mask-to-polygon" candidate argued CRS belongs to the caller; here `_read_polys_geom` reprojects internally. Pick one and apply consistently.
3. **Does `max_area_m2` stay a module constant or become a parameter?** filter_sam_inventory wants it tunable (`refined_area_m2_max`); the eval path wants it locked at 20000. Likely: module default 20000, overridable — but the eval callers must pass nothing (keep the lock).
4. **Layer-selection fallback** (`layer if present else first layer`) — part of this module, or caller's job?
5. **`polygon_conf_sweep.py` vs `poly_conf_sweep.py`** — consolidate, or just document? (Out of strict scope, but discovered here.)
6. **Return type** — keep the positional 5-tuple (callers unpack it) or introduce a small dataclass and adapt callers? Dataclass is cleaner but widens the diff and the byte-equivalence surface.

## 9. Call-site inventory (audit target)

| File | Today | Action |
|---|---|---|
| `area_aggregate_eval.py` | **defines** `_geometry_finite`, `_sum_area_m2`, `_read_polys_geom`, `_MAX_PLAUSIBLE_POLY_M2` | move to core + leave shim |
| `validate_checkpoint.py` | imports `_read_polys_geom` (eval-protocol-sensitive) | repoint via shim; byte-equiv gate |
| `lock_operating_point.py` | imports `_read_polys_geom` | repoint |
| `polygon_conf_sweep.py` | imports `_geometry_finite`, `_read_polys_geom`, `_MAX_PLAUSIBLE_POLY_M2` | repoint |
| `poly_conf_sweep.py` | imports same | repoint (+ untangle vs polygon_conf_sweep) |
| `li_count_recall_sweep.py` | imports same | repoint |
| `eval_xdomain60.py` | **reimplements** `_clean_polys` + own 20000 const | replace with canonical |
| `build_model_area_coverage_tracker.py` | **reimplements** `_geometry_finite` + `_clean_metric_gdf` + own const | replace with canonical |
| `filter_sam_inventory.py` | **reimplements** `_finite_bounds`/`_clean_metric`; **also has policy filters** | extract validity only; keep policy layer in script |
| `installation_sym_eval.py`, `per_grid_dispersion_audit.py`, `gtnoise_t1_score.py`, `sseg_kw_calibration.py` | own 20000 const + ad-hoc validity drops | fold to canonical (coordinate w/ step 4 for the first two) |

## 10. Acceptance criteria

- [ ] `core/polygon_validation.py` exists; `area_aggregate_eval.py` keeps a re-export shim (D3).
- [ ] Byte-identical outputs on a real run for **every** repointed caller (snapshot diff; at minimum `area_aggregate_eval` + `validate_checkpoint` on a known grid set).
- [ ] §4 zero-area divergence resolved **explicitly** and documented.
- [ ] New CPU-only unit tests: invalid geom, non-finite bounds, area==0, area at/over 20000, empty gpkg, missing layer, CRS already-metric vs needs-reproject.
- [ ] No policy/score filter logic absorbed into the module.
- [ ] No hardcoded EPSG; `metric_crs` is a parameter.
- [ ] `docs/architecture.md` module table + ADR-0001 updated in the same commit; full test suite green.

## 11. First moves for the fresh session

1. Read the 4 canonical funcs in `area_aggregate_eval.py:55-145` and the 3 local reimplementations (`build_model_area_coverage_tracker._clean_metric_gdf`, `filter_sam_inventory._finite_bounds/_clean_metric`, `eval_xdomain60._clean_polys`). **Diff them** — find every real divergence (the §4 area==0 split is one; look for more).
2. Decide the interface (grill §8, or `/design-an-interface`).
3. Move + shim; repoint callers one at a time, each behind a byte-equiv snapshot.
4. Tests + doc/ADR sync.

## 12. Pointers

- Survivor #2 (deferred): `MaskSupervisionPatch` lifecycle — `boundary_aware_mask.py` exports `clear_batch_supervision()` that the prod train loop never calls. Separate handoff if/when picked.
- Full review (ephemeral): `/tmp/architecture-review-20260619-002918.html`.
- `docs/adr/0001-codebase-optimization-2026-06.md` — the deepening track + 全程铁律 + D3 + step 3/4.
- `docs/evaluation_protocol.md` — what is locked (and what isn't).
- Rules: `06-multi-city.md` (CRS lookup), `03-doc-sync.md`, `02-evaluation-semantics.md`.
- Memory: `project_crs_policy`, `feedback_tier1_metric_system`.
