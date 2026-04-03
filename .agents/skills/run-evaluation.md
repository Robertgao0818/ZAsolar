# run-evaluation Skill

Run end-to-end detection + evaluation and summarize results.

## When to use

After model training, parameter changes, or when evaluating a new grid.

## Steps

1. Confirm GPU availability: `./scripts/check_env.sh`
2. Run detection + evaluation:
   ```bash
   python detect_and_evaluate.py \
     --model-path checkpoints/best_model.pth \
     --evaluation-profile installation \
     --force
   ```
3. Read `results/<GridID>/presence_metrics.csv` for F1/precision/recall.
4. Read `results/<GridID>/footprint_metrics.csv` for IoU/Dice distribution.
5. Read `results/<GridID>/area_error_metrics.csv` for size-stratified errors.
6. Compare against baseline thresholds from `configs/datasets/regions.yaml`.
7. Summarize: pass/fail for each acceptance criterion.

## Benchmark-First Workflow

For standard model-vs-model comparisons, prefer the benchmark runner over single-grid evaluation:

```bash
# Benchmark across all suites (primary output: summary.json)
python3 scripts/analysis/run_benchmark.py --checkpoint checkpoints/best_model.pth --tag my_exp

# Results at: results/benchmark/<run_name>_<timestamp>/summary.json
```

Single-grid evaluation (above) is still useful for quick checks and debugging.

## Constraints

- Always use `--evaluation-profile installation` unless user requests legacy mode.
- Always use `--force` when evaluating after parameter or model changes.
- Report all three metric layers (presence, footprint, area error).
- For ranking decisions, use `summary.json` from benchmark runs, not single-grid results.
