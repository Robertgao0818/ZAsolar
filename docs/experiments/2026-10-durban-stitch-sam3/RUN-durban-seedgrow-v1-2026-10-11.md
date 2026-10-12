# Durban pilot XL+L: SAM3 seed growth v1

Date: 2026-10-11. The GPU run and the report finished just after midnight on 2026-10-12.
Work dir: `~/zasolar_data/durban_stitch_sam3_20261009/seedgrow_v1/` (= `~/data/ZAsolar/...`).
Local RTX 4070 only. Cached Vexcel extracts only. No download, no RunPod, no Dropbox, no git commit.
Nothing in `B2_pilot200/`, `B2_pilot200/v3/`, `C_recall/`, or the census inventory was modified.
Script: `seedgrow_v1/seedgrow_v1.py`. Cache key `sg1-sam-20261011a`.

## What this compares

The seed is the original census polygon. SAM3 may add area around it. Growth never removes original area.
The comparison arms are the v2 box final (BOX) and the v3 text-plus-box final (TBX), which can still remove area.
Scope: the 80 pilot-200 blocks with stratum XL (30) or L (50), sample seed 20261009. Sum of original area: 138,079 m2 (XL 68,974, L 69,105).

## Method

The search region is the original polygon's axis-aligned bbox expanded by 6 m, then clipped to the cached extract. `growth_zone_clipped` is false on every row. The 6 m zone was inside the extract for all 80 blocks.

TXT runs `Sam3Model` + `Sam3Processor` (fp16, one 1008 px window at a time, stride 880, 64 px edge drop) with the text "solar panel" and no boxes. The instance threshold is 0.3. An instance is kept only from the window whose non-overlapping keep region contains the mask centroid. The full mask is kept, not the slice inside the keep region. Specks under 0.5 m2 are dropped.

TBX_nc is the same text plus the v3 piece boxes as positive exemplars, and it keeps every instance. There is no clip-to-box step. A window with no piece box reuses the TXT instances and is not run again. `seconds_sam` counts only forwards that actually ran.

TXT_ds2 resamples that same region to 0.5x and runs TXT again. It is XL only.

An instance is connected when its intersection with the original is at least max(1 m2, 10% of the instance area). The chain then adds an instance within 0.3 m of the connected set, or overlapping that set by the same area rule. Unconnected instances are discarded. Grown = original union the connected instances, clipped to `original.buffer(3 m)` or `original.buffer(5 m)`. Additions are Grown minus original. Components under 1 m2 are dropped. The delivered polygon does not keep those crumbs, and it does not drop any of the original.

Arms: G3_TXT, G5_TXT, G3_TBXnc, G5_TBXnc, and on XL only G3_TXTds2 and G5_TXTds2. BOXplusG5 is the v2 final polygon union the G5_TXT additions, so BOX can still remove area inside the seed.

`area_added_sam` is the full area of the delivered polygon outside the original, including slivers under 1 m2. `area_tightened` is original area missing from the final. They are never folded into one number. On the pure growth arms the two added-area definitions match: G5_TXT is 11,008 m2 either way. On BOX, `area_added_sam` is 3,048 m2 and the components of at least 1 m2 sum to 2,758 m2. The 290 m2 gap is exterior slivers under 1 m2. Use `area_added_sam` as the block total.

Flags: `touches_neighbour` means an added component intersects, or comes within 0.5 m of, a different census block. `capped` means the cap removed more than 20% of that component's pre-cap area. `lowconf_add` means the instances that made the component have a mean score under 0.5. `big_growth` means the block's added area is over 25% of the original.

F3 overlap uses the 24 F3_rescue near-array pieces from experiment C. Fifteen of these 80 blocks sit in the five C cells (DBN0716, DBN0733, DBN0734, DBN0030, DBN0016). Overlap is 0 m2 on every block and every arm.

The first smoke test failed in the sheet renderer: a growth dict had been stored in place of the BOX and TBX polygons. That was fixed without rerunning SAM. The second smoke wrote `sheets/sheet_smoke.jpg`. The full run then cached the other 77 blocks. Failures: 0. Those 77 per-block wall times sum to 317 s. Report assemble wall was 541 s. Median GPU seconds are in the table.

## Totals, all 80 blocks

Added and tightened are separate. TXT_ds2 rows are the 30 XL blocks only. Median GPU seconds is the median of `seconds_sam`.

| arm | n | added m2 | tightened m2 | median added % | with addition | neighbour | capped | lowconf | big_growth | gapfill m2 | net m2 | median GPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL | 80 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | |
| BOX | 80 | 3,048 | 12,153 | 0.8 | 69 | 5 | 0 | 0 | 0 | 559.1 | −9,105 | 1.28 |
| TBX | 80 | 2,290 | 14,767 | 0.6 | 73 | 2 | 0 | 0 | 0 | 360.0 | −12,478 | 1.20 |
| G3_TXT | 80 | 7,241 | 0 | 1.6 | 63 | 32 | 35 | 42 | 2 | 1,746.8 | 7,241 | 0.61 |
| G5_TXT | 80 | 11,008 | 0 | 2.9 | 65 | 34 | 30 | 44 | 10 | 2,247.4 | 11,008 | 0.61 |
| G3_TBXnc | 80 | 6,693 | 0 | 1.0 | 64 | 20 | 29 | 14 | 2 | 2,566.5 | 6,693 | 0.62 |
| G5_TBXnc | 80 | 10,124 | 0 | 1.7 | 65 | 22 | 23 | 14 | 7 | 3,384.7 | 10,124 | 0.62 |
| G3_TXTds2 | 30 | 2,291 | 0 | 1.3 | 24 | 11 | 13 | 11 | 1 | 786.5 | 2,291 | 0.30 |
| G5_TXTds2 | 30 | 3,857 | 0 | 1.5 | 24 | 12 | 12 | 12 | 2 | 983.2 | 3,857 | 0.30 |
| BOXplusG5 | 80 | 12,728 | 12,153 | 5.2 | 73 | 35 | 30 | 43 | 10 | 2,546.6 | 575 | 0.61 |

G5_TXT by stratum: XL adds 3,496 m2 (median block 2.1%, neighbour 12, big-growth 1). L adds 7,512 m2 (median 3.0%, neighbour 22, big-growth 9). The full stratum table is in `summary_seedgrow.md`.

`sheet_largest_additions.jpg` and `sheet_flagged.jpg` are the same 12 blocks. The tail that makes the sum 8.0% of the seed, while the median block is only +2.9%, is the flagged set.

## Recommendation

Do not write seed growth back onto the census automatically.

G5_TXT adds 11,008 m2. Fifty of the 80 blocks carry a neighbour, cap, low-confidence, or big-growth flag, and those blocks hold 10,664 m2 of the added area. Components that touch another census block sum to 7,192 m2. Applying this arm unfiltered would treat neighbouring arrays, walkways, and roof as new panel area.

The 3 m text cap does not fix that. G3_TXT adds 7,241 m2. Big-growth falls from 10 blocks to 2, neighbour stays at 32 against 34, and capped rises from 30 to 35 because the tighter cap cuts more of the same masks. Of the extra 3,767 m2 that the 5 m cap keeps, 2,347 m2 is on L blocks and 1,420 m2 is on XL blocks.

G5_TBXnc is the cleaner growth arm on the counts. It adds 10,124 m2, with neighbour blocks 22 against 34, low-confidence blocks 14 against 44, and 3,385 m2 overlapping a v2 gap-fill candidate against 2,247 m2 for G5_TXT. The contact sheets draw G5_TXT only, so this comparison was not checked block by block on imagery.

If a later pass keeps a review queue, use the 5 m exemplar arm, G5_TBXnc, and keep only components that are not a neighbour, not capped, and not low-confidence, on blocks that are not big-growth. That queue is 2,717 m2 (241 components on 50 blocks).

The 3 m cap is the wrong sieve for an interior hole. On `g0.25_013743` the addition is 166 m2, the farthest point is 4.9 m from the original outline, and the cyan lies on the panel rows. G5_TBXnc keeps that hole (166 m2, score 0.99, no flags). G3_TBXnc marks the block capped, and the same component filter keeps 5 m2 of it. The 5 m exemplar arm adds 3,430 m2 more than G3_TBXnc and raises big-growth from 2 blocks to 7, so those blocks still have to be dropped from the queue.

A flag filter is not ground truth. `g0.25_027878` adds 65 m2 on G5_TXT with no flags, and the cyan lies on the white walkways between panel blocks. The same block stays inside both caps (62 m2 on G5_TBXnc, no flags). It is not one of the six example files. The queue is for review. It is not a new census polygon.

Dropping a whole block that has any of the four flags leaves less: G5_TXT keeps 344 m2 on 30 blocks, and G5_TBXnc keeps 2,108 m2 on 50 blocks. The component filter above is the larger review list, because a clean component can sit on a block that also has a bad one.

Do not use TXT_ds2. On the 30 XL blocks G5_TXTds2 adds 3,857 m2 against 3,496 m2 for G5_TXT. On `g0.25_023868` one downsampled instance at score 0.33 adds 666 m2. G5_TXT on that block adds 28 m2. The example image draws the 28 m2 result. It does not draw the downsampled mask.

Do not read the BOXplusG5 net of 575 m2 as a small change. That arm still removes 12,153 m2 inside the seed, the same tightened area as BOX, and it adds 12,728 m2 outside the seed. The addition does not put back the area BOX removed. G5 additions and the BOX exterior overlap, which is why 3,048 + 11,008 is larger than 12,728.

## Examples

Same 4-panel layout as the sheets: original yellow, BOX red, TBX magenta, G5_TXT with additions in cyan and discarded instances in grey. The convex-hull sort was not used. It had picked `g0.25_022676` (312 m2, capped, lowconf) and `g0.25_026846` (lowconf). Both spill onto roof or a gap. They were replaced after looking at the images.

- `examples/01_good_fill_g0.25_013743.png` — rectangular hole in the census polygon. Cyan sits on the same panel rows. 166 m2, score 0.55, no flags.
- `examples/02_good_fill_g0.25_013853.png` — cyan sits on panel modules at the seam between array sections. 21 m2, score 0.97, no flags. A bare-roof blob at the south end is already in the original polygon and was not grown.
- `examples/03_neighbour_or_roof_g0.25_026513.png` — the original polygon is the centre walkway. Cyan covers the panel rows on both sides. 740 m2. Flags: neighbour, capped, lowconf, big_growth.
- `examples/04_neighbour_or_roof_g0.25_002428.png` — the original polygon is the two outer rows. Cyan fills the white roof and the panel rows between them. 664 m2. Flags: neighbour, capped, lowconf, big_growth.
- `examples/05_ds2_differs_g0.25_023868.png` — the picture is G5_TXT only (28 m2, lowconf). G5_TXTds2 on this block adds 666 m2 from 1 instance at score 0.33, with neighbour, capped, lowconf, and big_growth.
- `examples/06_no_change_g0.25_020163.png` — G5 addition is 0.0 m2, and 94 text instances were still found. The polygon covers one panel row and leaves the next row out. Growth does not add that row, because an instance joins only when it meets the seed or comes within 0.3 m of an instance that already has.

## Caveats

There is no ground truth for the additions. The check is the six example images, one extra walkway block (`g0.25_027878`), and the three contact sheets.

The TBX_nc arms were not drawn as their own panels. The ds2 spill on `g0.25_023868` is in the metrics, not in the fourth panel.

Seed growth misses a neighbouring row that does not touch the seed. Example 06 is that case.

F3 overlap is 0. This pilot does not measure whether growth recovers the pieces experiment C found.

## Outputs

`~/zasolar_data/durban_stitch_sam3_20261009/seedgrow_v1/`

- `seedgrow_v1.py`, `run.log`, `blocks/<id>.json` (80 caches)
- `blocks_seedgrow.csv` (700 rows, block x arm), `added_components.csv` (3,355 rows)
- `seedgrow_pilot_XL_L_v1.gpkg` (EPSG:32736): `original` 80; `grown_*` and `added_*` for G3_TXT (428 added), G5_TXT (451), G3_TBXnc (355), G5_TBXnc (371), G3_TXTds2 (248 added, 30 grown), G5_TXTds2 (255 added, 30 grown), BOXplusG5 (556 added); `instances_txt` 4,356
- `summary_seedgrow.md`, `summary_seedgrow.json`
- `sheets/sheet_largest_additions.jpg`, `sheets/sheet_flagged.jpg`, `sheets/sheet_random.jpg` (seed 20261009), plus `sheets/sheet_smoke.jpg`
- `examples/` the six PNGs listed above
