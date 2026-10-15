# Architecture

Both packages implement the same stages:

1. Resolve configuration-relative paths and validate data.
2. Align matrix columns to phenotype IDs and convert beta values to M-values.
3. Fit per-CpG Cox models for each endpoint, applying configured strata/adjustment terms.
4. For paired endpoints, combine log hazard ratios with an empirical-Bayes random-effects model.
5. For each leave-one-out fold, refit all CpG Cox models once and rank features by absolute ordinary Cox z, then reuse that ranking across every candidate count to compute each count's fold median/IQR-normalized score. The ranking does not depend on the candidate count, so computing it once per fold is exactly equivalent to the original per-count recomputation (see "Per-fold ranking reuse" in the README).
6. Report the search curve and use the manually configured count by default; maximum AUC is opt-in.
7. Run a separate fresh leave-one-out extraction at the chosen count and select stable CpGs by inclusion frequency.
8. Refit full-data weights, apply legacy median/IQR scaling, and write portable results and task provenance.

The local and SLURM backends invoke the same literal task definitions. Parallelization changes task placement only. The per-fold ranking is computed once and reused across candidate counts (it is count-independent), but estimates are never reused between the search and extraction phases. Scheduler-specific rendering is isolated in `submit_slurm()`.

The per-CpG Cox scan in stages 3 and 5 can run under different **compute engines** (exact scipy, vectorized CPU, or CUDA) selected with `--engine`; all produce the same result to numerical tolerance. See "Compute engines" in the README.
