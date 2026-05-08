# Li GT Archive

Archived on 2026-05-08.

These files are historical audit material only. They are no longer active GT
sources for training, benchmark reporting, Channel 2 work package generation, or
area coverage tracking.

Reason:

- The Joburg CBD historical annotation set was drawn against a different imagery
  vintage than current evaluation imagery, so it can misclassify install-date
  differences as model recall failures.
- The Cape Town alternate grid scheme is no longer registered as an active
  annotation scheme in `configs/datasets/regions.yaml`.
- V1.4 validation now centers on reviewed/clean GT and grid-level aggregate
  inventory quality.

Kept here for traceability:

- `scripts/`: old Li-specific dataset and imagery helpers.
- `data/annotations/`: archived Li-derived annotation manifests/summaries and
  ignored local GPKGs when present in the workspace.
- `data/cape_town_grid_Li.*`: archived alternate Cape Town grid files.
- `docs/experiments/exp_li_gt_audit.md`: historical audit write-up.

Do not import or call these paths from active pipeline code. If a future audit
needs them, copy the relevant material into a new dated experiment folder and
document the imagery vintage and validation frame explicitly.
