# Federated learning in PODE

`pode federated` lets several centres build things together without pooling
patient rows. It covers two kinds of output:
- synthetic-data generators (TVAE);
- prognostic CpG signatures (MetaCPG).

It answers the questions raised in `federated_learning.md` with four capabilities and
an evaluation harness:

| Capability | Command | What leaves a site |
|---|---|---|
| Federated TVAE (FedAvg / FedProx, personalisation) | `pode federated tvae train\|sample` | column summaries once; model weights each round (masked) |
| DP-SGD for local or federated TVAE | `--dp --target-epsilon E` | the same, with each update differentially private |
| Local TVAEs + privacy-gated exchange | `pode federated exchange` | a verdict; synthetic rows only if they pass the gate |
| Federated MetaCPG signatures | `pode federated signature run\|verify\|benchmark` | per-CpG score/information, moments, counts |
| Benchmark of all generator options | `pode federated benchmark` / `report` | (evaluation harness) |

Today the sites are **simulated in one process**: one table is split by a site
column, or a manifest lists per-site CSVs. Every protocol is written so that a site
answers only *site operations*, through `Site.run`. That function is the seam where a
networked client (e.g. Flower) would plug in, without any change to the coordinator
code. See [Path to a real deployment](#path-to-a-real-deployment).

Install the optional extra for DP accounting:

```bash
pip install -e '.[federated]'
```

This adds Opacus. Everything except DP works without it.

## What crosses a site boundary

Every value a site returns is recorded in a **disclosure ledger**, saved as
`disclosure_ledger.csv` and summarised in every report. The ledger records:
- the operation;
- the site;
- the bytes and shape;
- whether the value is **per patient** (one value per patient) or an aggregate.

Read the ledger before trusting a privacy argument. Federated learning reduces data
transfer; it does not make leakage impossible. Model updates and summaries can still
leak, which is why DP-SGD and the privacy gate exist.

## 1. Federated TVAE

The architecture is the same ctgan TVAE that PODE synthesis uses. Its sizes come from
the PODE config's `synthesizer` section, so tuned values carry over. Training is
spread across sites.

1. **Shared column encoder from summaries.** Every site has to encode its rows the
   same way: the same category lists and the same Gaussian-mixture modes per numeric
   column.
   - Each site sends, per column:
     - category counts (a category seen in fewer than `min_cell` patients at that site
       is pooled into `__rare__`; this is small-cell suppression);
     - missing counts;
     - two robust quantiles;
     - a histogram on the agreed bin edges.
   - The coordinator fits the encoder on a pseudo-sample drawn independently from
     those marginals. No row ever leaves the site, and no joint information does
     either.
2. **Rounds.**
   - The coordinator broadcasts the weights.
   - Each site runs `local_epochs` of Adam on its own rows. Its optimiser state and
     random stream stay at the site.
   - **FedProx** adds `mu/2 ||w - w_global||^2`, which limits drift when sites differ
     (non-IID data).
3. **Aggregation.**
   - Sites pre-scale their weights by their aggregation weight:
     - `n`: FedAvg;
     - `sqrt_n` or `uniform`: flatter;
     - `capped`: no site above `cap_fraction`, so one big centre cannot dominate.
   - With `secure_aggregation`, each site adds pairwise masks that cancel in the sum,
     so the coordinator sees only the weighted average. This is the Bonawitz protocol,
     simulated in fixed point.
4. **Personalisation (optional).** After federation, each site fine-tunes its own copy
   for `personalise_epochs`. That generator stays at the site and samples there.
5. **Sampling** applies PODE's survival post-processing and the configured
   plausibility `constraints`.

Exactness checks (in the tests):
- one site trained for R × L epochs equals a plain TVAE fit;
- FedAvg over identical sites equals one site;
- FedProx with `mu = 0` equals FedAvg;
- secure and plain aggregation agree to fixed-point precision (about 1e-6).

```bash
pode federated tvae train --data cohort.csv --site-column Cohort --config pode_config.yaml \
    --federated-config configs/federated/benchmark.example.yaml --output-dir fed_tvae/ --num-samples 500
pode federated tvae sample --model fed_tvae/ --config pode_config.yaml --num-samples 1000 --output-dir out/
```

The global generator is shared with every party, which is the point of a federated
generator. Rows sampled from it are subject to the same release gate
([PRIVACY.md](../synthesis/PRIVACY.md)) as any other synthetic data.

## 2. DP-SGD

With `dp.enabled` (or `--dp --target-epsilon E`), each training step does three
things:
- it draws a Poisson batch (each patient included with probability `q = dp.batch_size / n_site`);
- it clips each patient's gradient to norm `C`;
- it adds Gaussian noise.

The decoder's free `sigma` parameter defeats layer hooks, so per-patient gradients
come from `torch.func`. Opacus does the accounting: each site's ε is composed over all
its steps in every round. A `target_epsilon` is solved into a noise multiplier before
training. The encoder summaries get Laplace noise from `dp.transformer_epsilon`. That
amount is added to each site's total, and both parts are reported.

**Reading ε:**

| ε | Guarantee |
|---|---|
| about 1 | strong |
| 3–8 | common for useful deep models |
| above 10 | weak |

ε bounds how much any one patient can change what the site's updates reveal. Some
encoder inputs are data-derived (numeric ranges and category lists) and DP does not
cover them; the report lists them. To close that gap, set public
`transformer.bounds` and `transformer.categories`.

## 3. Local TVAEs + gated exchange

This is the lightest federation: there are no rounds. At each site:
1. Split the site's patients into train and holdout.
2. Fit a local TVAE: `fed_tvae` (optionally DP) or PODE's `sdv_tvae`.
3. Sample.
4. Run the full `pode.privacy` assessment.
5. **Release only on an allowed verdict** (default: PASS).

All four steps happen at the site. The release card records which sites were released
and which were withheld, ε, and the intended and prohibited uses. The pooled data mixes
site-specific distributions and omits withheld sites. Say so wherever it is used.

```bash
pode federated exchange --data cohort.csv --site-column Cohort --config pode_config.yaml \
    --privacy-config configs/privacy/small_cohort.example.yaml --output-dir exchange/
```

## 4. Federated MetaCPG signatures

MetaCPG decomposes over sites, so the federated version **reproduces the centralised
analysis** while sharing only statistics:

- **Pooled designs** (one endpoint whose strata nest within sites, e.g. cohort × age
  strata).
  - The Efron score vector and information matrix are sums over strata, so they are
    also sums over sites.
  - The coordinator runs MetaCPG's own batched Newton–Raphson. Each iteration sums
    the sites' score and information, computed at the broadcast coefficients.
  - The method refuses strata that span sites, or an unstratified pooled risk set:
    neither decomposes. The fix is to add the site column as a stratum.
- **Meta-analysis designs** (`group_column`). Each site fits its own Cox models and
  the coordinator pools them with MetaCPG's random-effects meta-analysis.
- **Weights** need the pooled SD per CpG. Sites send (n, mean, M2), and the
  coordinator merges them exactly.
- **Score scaling** (the median and IQR of training raw scores) comes from an exact
  count-only protocol.
  - Sites first report their minimum and maximum per target.
  - They then answer "how many of your values are ≤ t?" for a grid of thresholds.
  - Finally, each site sends the few values inside the last tiny bracket.
  - The only individual values revealed are order statistics, the same numbers the
    final model publishes.
- **Leave-one-out.** Holding out patient *i* changes only their own site's
  contribution, so `--cv exact_loo` reproduces MetaCPG's search and extraction.
  - `--metrics pooled` sends the held-out LOO scores, times and statuses to the
    coordinator (per patient, as logged) and gives MetaCPG's exact metrics.
  - `--metrics within_site` keeps them at the site and shares pair counts; the C-index
    is then site-stratified.
- **`--cv loso`** is the deployment-realistic option. The signature is built from
  k−1 sites and scored at the held-out site, which reports only aggregates (pair
  counts, AUC, the score's HR).

```bash
# Prove equivalence on your config (and against a frozen model):
pode federated signature verify --config metacpg.yml --site-column os.cohort.merged \
    --feature-count 35 --extraction-only --reference-model frozen/model.json --output-dir verify/
# Leave-one-site-out: is the federated signature better than single-site ones at an unseen site?
pode federated signature benchmark --config metacpg.yml --site-column os.cohort.merged \
    --feature-count 35 --output-dir loso/
# Federated search + extraction writing a normal model.json (usable by `pode signatures predict`):
pode federated signature run --config metacpg.yml --site-column os.cohort.merged --output-dir run/
```

The federated fits use the batched Newton engine. Against a central run with
`engine: batched`, the agreement is about 1e-15, which is summation order. Against the
default scipy `exact` engine it is about 1e-8, from a different optimiser converging to
the same estimate. Selections are identical in both cases.

## 5. Benchmark

`pode federated benchmark` asks whether the federated generators are useful and safe.

**Design.** For each repeat, every site is split into train and test. Generators see
site *train* rows only. At every evaluation site, measured on that site's **real test
patients**, the benchmark reports:
- **Utility (TSTR).** A downstream model (augmentation's `cox_ph`, `coxnet`,
  `metacpg_fixed`, …) is trained on the synthetic rows and scored. The report gives
  Harrell and Uno C, AUC, Brier, IPA and calibration at the horizon.
- **References:**
  - `real_local`: the site's own real rows;
  - `real_pooled`: the "if pooling were permitted" case;
  - `real_federated_cox`: a site-stratified ridge Cox fitted from per-site score and
    information. This is the note's federated real-data reference model.
- **Paired ΔC** against `real_local`, the local TVAE and the pooled TVAE, with
  corrected CIs over sites and repeats.
- **Fidelity:** marginal gaps, correlation gap, real-vs-synthetic discriminator AUC,
  KM gap and missing-rate gap.
- **Privacy:** the `pode.privacy` suite per site, with members = its training
  patients and non-members = its test patients.
- **Cost:** seconds, MB communicated and ε.

**Leave-one-site-out arm (`sites.loso`).** The held-out site contributes nothing to
any generator or model, and all of its patients are test patients.

**Conditions:**

| Condition | What it is |
|---|---|
| `local` | each site's own TVAE |
| `exchange` | pooled synthetic data that passed the gate |
| `fedavg`, `fedprox` | federated TVAE |
| `fed_personalised` | federated TVAE plus local fine-tuning |
| `fed_dp_eps<E>` | federated TVAE with DP at ε = E |
| `central_fed_tvae` | same architecture, pooled |
| `central_sdv_tvae` | PODE's TVAE, pooled |
| `central_copula` | non-neural baseline |

**Verdicts (first model):**

| Verdict | Meaning |
|---|---|
| **UNSAFE** | the privacy gate FAILed at some site in some repeat |
| **RECOMMENDED** | non-inferior to the pooled TVAE *and* not worse than each site's local TVAE, with acceptable fidelity |
| **NON-INFERIOR** | non-inferior to pooled only |
| **NOT RECOMMENDED** | otherwise |
| **BASELINE** | a comparator |

The UNSAFE rule is deliberately strict: one FAIL anywhere is enough. Small sites with
15–20 test patients produce noisy privacy checks, so read the FAIL rate and the
failing checks in the privacy table alongside the verdict.

```bash
pode federated benchmark --data cohort.csv --config pode_config.yaml \
    --federated-config configs/federated/benchmark.example.yaml --dry-run   # what it resolves to
pode federated benchmark ... --quick                                        # smoke test
pode federated benchmark ... --output-dir run/                              # full; resumes from checkpoints
pode federated report --output-dir run/                                     # re-render
```

Long runs: launch detached (`setsid nohup … &`) from a git worktree.

## Configuration

[`configs/federated/benchmark.example.yaml`](../../configs/federated/benchmark.example.yaml)
lists every option with its default. A new study needs:
- a table with a site column (or a site manifest);
- `sites.column`;
- `generator.exclude_columns` (ids, post-outcome fields);
- the evaluation endpoint and features.

The TVAE architecture comes from `--config` (a PODE config, e.g. a promoted
`pode_config_final.yaml`). `n_jobs` defaults to 4. CPU threads and torch threads are
capped accordingly.

## Limitations

- **Simulation.** Secure aggregation is simulated. Real deployments need key
  agreement, dropout recovery, authenticated sites and audit logs.
- **Federation does not protect updates.** Model updates can leak through
  gradient-inversion and membership attacks on updates. DP-SGD is the protection that
  covers them; the release gate tests only the synthetic rows.
- **The schema is an agreed data model.** In simulation it is detected on the pooled
  table; in a deployment it is supplied (`--schema`).
- **Not survival-aware.** The TVAE treats time and event as ordinary columns. For
  whole trajectories, use a survival-aware or longitudinal generator.
- **The signature-benchmark C-index is site-stratified** (concordant pairs within
  each held-out site), not pooled.
- **Small sites.** Wide CIs, noisy privacy checks and the largest DP utility cost all
  concentrate in small sites, which are also where federation is supposed to help
  most.

## Path to a real deployment

1. Wrap each `site_*` operation in a Flower `NumPyClient`. `Site.run` becomes a
   remote call; the coordinator code is unchanged.
2. Replace `SecureSum` with Flower's SecAgg+ (or equivalent).
3. Distribute the schema, the encoder bounds and vocabularies, and the configuration
   as versioned public artefacts.
4. Keep the per-site privacy gate, the disclosure ledger and the model and data cards
   as release requirements.
