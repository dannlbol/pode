# Verifying agreement between compute engines (`verify-parity`)

Use `verify-parity` to prove the exact, CPU-batched, and GPU engines produce the
same signature — and, in particular, that the CPU and GPU paths are numerically
identical. It runs the full workflow under each engine, compares every output
artifact, prints a PASS/FAIL table, and exits non-zero on failure.

```bash
metacpg-py verify-parity \
  --config analysis.yml --feature-count 30 \
  --engines exact,batched,gpu --workers 20 \
  --output parity_report.json
```

- The first engine listed is the reference (default `exact`).
- `--tolerance` (default `1e-6`) bounds the max absolute coefficient/z/risk-score
  difference versus the reference; selected features must be identical. CPU vs GPU
  are additionally held to a strict `1e-9`.
- Search-curve AUC/C-index rows are marked advisory (`*`) — they are rank-based
  step functions that can shift at a non-selected count without changing the
  final signature, so they are reported but do not gate the verdict.

A full exact reference run over a large matrix takes hours, so the
exact-vs-accelerated comparison is usually run on a CpG subset; the
`--engines batched,gpu` comparison runs quickly on the full matrix and is the
direct CPU-vs-GPU equivalence check.

# Comparing a run against a reference result

`compare` checks a new run against a reference result directory (for example, a
previous release or the other engine's output):

```bash
pode signatures compare \
  --engine python \
  --reference results/reference_run \
  --candidate results/example/python \
  --output results/python_comparison.json
```

Each comparison creates JSON, Markdown, and a risk-score agreement PNG. Inspect a report with:

```bash
jq '.metrics, .thresholds, .checks, .passed' results/python_comparison.json
```

Acceptance thresholds are:

| Engine | CpG overlap | Risk-score Spearman | Maximum AUC/C-index difference |
|---|---:|---:|---:|
| R | ≥ 0.95 | ≥ 0.99 | ≤ 0.01 |
| Python | ≥ 0.80 | ≥ 0.95 | ≤ 0.03 |

## Run the package tests

```bash
Rscript -e 'testthat::test_local("R/metacpg", reporter="summary")'
python -m pytest tests/signatures -q
```

## Inspect methodology and refit accounting

```bash
jq '.methodology, .cox_fit_counts, .scaling, .performance' \
  results/example/r/result.json
```

`methodology` should be `legacy_exact`. Search and extraction have separate task provenance and Cox-fit counts; extraction should not report resumed search tasks.

