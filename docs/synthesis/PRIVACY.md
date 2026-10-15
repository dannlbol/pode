# Privacy assessment of synthetic data

`pode synthesis privacy` tests whether a TVAE synthetic dataset protects the
children whose records trained it. A standard TVAE is **not privacy-preserving
by default**. It can generalise well and still memorise rare or distinctive
patients. Paediatric oncology raises that risk: cohorts are small, molecular
subtypes are rare, treatment pathways are distinctive, and quasi-identifiers
such as age, centre and diagnosis year are informative.

The assessment is a **release gate**. It runs ten checks against a stated threat
model and gives each one PASS, REVIEW or FAIL. It also lists the governance
items that cannot be automated. A PASS is evidence, not a guarantee. Call the
data *synthetic*, not "anonymous".

Code: `pode/privacy/`. Example config:
`configs/privacy/small_cohort.example.yaml`.

## Two modes

```bash
# 1. Assess an existing synthetic table.
pode synthesis privacy assess \
    --train real_training.csv \        # the patients the generator was trained on
    --holdout real_unseen.csv \        # same population, never seen by the generator
    --synthetic synthetic_data.csv \
    --config pode_config.yaml --privacy-config privacy.yaml \
    --output-dir privacy_out --n-jobs 4

# 2. Audit the TVAE configuration end to end:
#    split -> fit on the training share -> generate -> assess.
pode synthesis privacy audit \
    --data all_real_patients.csv \
    --config pode_config.yaml --privacy-config privacy.yaml \
    --holdout-frac 0.3 --repeats 3 --output-dir privacy_audit --n-jobs 4
```

From Python:

```python
from pode.privacy import PrivacyConfig, assess_privacy

report = assess_privacy(train, synthetic, holdout, schema=schema,
                        config=PrivacyConfig.from_yaml("privacy.yaml"))
report.verdict   # "PASS" | "REVIEW" | "FAIL"
report.gate      # one row per check
report.save("privacy_out")
```

**Membership and disclosure need patients the generator never saw.** A
synthetic table generated from *all* real patients (the usual
`pode synthesis generate` output) has no non-members left, so membership
inference cannot be measured. Use one of these instead:

* keep a holdout aside before training and pass it to `assess`, or
* run `audit`, which measures the *model and configuration*. A release
  generated from all patients with the same configuration carries at least
  this risk. More patients per model usually lowers it; more epochs raise it.

Without a holdout, `assess` still runs. Distances are then referenced to each
patient's nearest real neighbour (leave-one-out). Membership, the attribute
member gap and linkage are NOT_ASSESSED, and the verdict is at best REVIEW.

The holdout must come from the **same population** as the training patients. An
external cohort differs systematically, so it makes synthetic rows look
spuriously close to the training patients. In practice, using an external
cohort as the reference can push membership AUC far above what internal splits
show.

`assess` applies the PODE config's data preparation to the real tables, so they
match what the generator saw. That covers identifier dropping, quality filters,
time scaling and censoring. The identifier check runs on the raw tables.

## Threat model

Privacy numbers only mean something relative to what an intruder could know.
The privacy config states this, and the report repeats it.

| Key | Meaning | Default when empty |
|---|---|---|
| `quasi_identifiers` | Attacker tiers: `{name: [columns]}`, e.g. `basic` (sex, age band, broad class), `linkage` (+ age, subtype, centre, trial), `high_risk` (+ treatment, rare events) | one `categorical` tier of all non-sensitive categorical columns (a warning is printed) |
| `sensitive` | Attributes harmful to disclose: relapse, survival status, germline flag, molecular status, toxicity | the schema's survival event columns |
| `strata` | Rare groups checked separately: diagnosis, subtype, age band, outcome | none |
| `weights` | Gower weight per column (globs), so identifying columns are not drowned out, e.g. by hundreds of CpGs | 1 |
| `id_column` | Labels case-review rows; never a feature | none |

Numeric quasi-identifiers form keys by training-quantile bins
(`numeric_bins`, default 5). Exact dates, postcodes and centre IDs should be
coarsened **before training**. A generator cannot make direct identifiers safe.

## The checks

Distances use a weighted, NaN-aware **Gower** distance. For numbers it is
|a−b| / training range, capped at 1; for categories it is a 0/1 mismatch, with
1, 1.0 and "1" treated as equal. Each distance is compared with a **reference**:
how close the held-out patients are to the training patients. The near-copy
band is the reference's closest 1% or 5%.

| # | Check | What it measures | Gate |
|---|---|---|---|
| 1 | **identifiers** | Identifier columns (schema role or unique-per-row text), date-like and free-text columns in the synthetic table, and real ID values appearing in any synthetic cell | identifier = FAIL; date/free text = REVIEW |
| 2 | **exact_copies** | Training patients reproduced as a whole record, after rounding and normalising. Compared with the rate at which synthetic rows coincidentally match *holdout* patients, which happens in coarse, mostly categorical tables | above `max_exact_copies` (0) and above the coincidence rate = FAIL; at the coincidence rate = REVIEW |
| 3 | **near_copies** | `dcr_p5_ratio`: 5th percentile of synthetic distance-to-closest-record / the reference's. `near_copy_excess`: share of synthetic rows inside the near-copy band ÷ the reference's share. `isolated_near_copy_excess`: near-copies sitting on one patient (nearest / second-nearest < 0.5) | ratio < 0.8 = FAIL; excess > 2 and significant = FAIL |
| 4 | **strata** | The same near-copy rate (and membership AUC, where n ≥ `min_group`) within each stratum level. Also flags levels with ≤ `rare_k` training patients that the synthetic data reproduces | any flagged level = REVIEW (`strata.csv`) |
| 5 | **membership** | Distance attack. Each real patient is scored by closeness to the nearest synthetic row, and the AUC measures how well that tells training patients from holdout patients. Run on all columns and on each attacker tier's columns, with a bootstrap 95% CI, advantage (max TPR−FPR) and TPR at 10% FPR | AUC > 0.60 = FAIL; CI upper > 0.60 = REVIEW |
| 6 | **attribute_inference** | An attacker who knows a child's quasi-identifiers finds the nearest 1 or 5 synthetic records and reads off each sensitive value. Reports accuracy for training patients, for unseen patients, and for a no-knowledge baseline | **member gap** > 0.05 with CI above 0 = FAIL; > 0.05 = REVIEW |
| 7 | **linkage** | *Singling-out*: a patient's quasi-identifier key is unique and exactly one synthetic row matches it (training vs holdout, in equal-sized cohorts). *Disclosive link*: that row is also a near-copy of the child's full record, which amounts to re-identification | singling-out excess > 0.02 and significant = FAIL; disclosive links > 2× the holdout rate and significant = FAIL |
| 8 | **rare_combinations** | Synthetic rows that reproduce a key seen in ≤ `rare_k` training patients **and** are near-copies, against the same event for holdout patients | above `max_rare_combo_leaks` (0) and > 2× chance, significant = FAIL |
| 9 | **outliers** | Unusual training patients: the top 5% by distance to the nearest real patient, or members of a small stratum. One is *replicated* when a synthetic row sits in the near-copy band or at under half the distance of the nearest real patient. Holdout outliers give the chance rate | replication significantly above chance = REVIEW (case review) |
| 10 | **missingness** | Synthetic rows whose missing-data pattern is rare in training. Unusual testing or follow-up patterns can identify a patient or centre | > 2× holdout rate = REVIEW |

How to read the main numbers:

* **Membership AUC**: 0.5 means no signal. 0.6 is worrying. 1.0 means membership
  is fully revealed. In small cohorts the CI is wide. A REVIEW caused by a wide CI
  calls for a larger holdout or more `--repeats`. It is not a reason to ignore
  the check.
* **Attribute member gap**: accuracy that is just as high for *unseen* patients
  is the data doing its job (population-level inference, `gain_over_baseline`).
  Only the *gap* between training and unseen patients is disclosure caused by
  being in the training data.
* **`dcr_p5_ratio`**: about 1 means the closest synthetic rows are no closer to
  training patients than new patients are.

**Chance and multiple testing.** Excess-type checks (3, 7, 8, 9, 10) compare
counts with the holdout rate using a one-sided binomial or two-proportion test
at `gates.alpha` (default 0.01, strict because about ten tests run together).
A large excess that chance could still explain is REVIEW, not FAIL. A near-copy
band holding fewer than 5 reference patients is not gated, because the band
itself is too noisy. This is why a holdout of at least 50–100 patients is worth
having.

The overall verdict is the worst automated status. A required check that could
not run (membership without a holdout) counts as REVIEW.

### Manual checklist

The report always lists these as unchecked boxes:

- data minimisation before training;
- a release tier that matches residual risk;
- utility checked separately (train on synthetic, test on real);
- independent review;
- governance, ethics and information-governance sign-off;
- whether a formal guarantee (differentially private training) is required.

## Outputs

| File | Contents |
|---|---|
| `report.md` | Verdict, gate table, manual checklist, threat model, how-to-read guide, per-check details, case-review tables, remediation options |
| `gate.csv` | One row per check: metric, value, threshold, status, reason |
| `privacy_summary.json` | All summary metrics plus the threat model actually used |
| `membership.csv`, `attribute_inference.csv`, `linkage.csv`, `rare_combinations.csv`, `strata.csv`, `identifiers.csv` | Per-attack / per-tier / per-level detail |
| `outliers.csv`, `closest_pairs.csv` | Case review: row numbers, patient IDs (with `id_column`) and distances, **never real values**. Inspect the pairs side by side with the data custodian |
| `privacy_config_used.yaml` | The resolved config |

`audit` writes one `repeat_<i>/` directory per repeat, each containing the above
plus `split.csv` and `synthetic.csv`. Alongside them it writes:

- a summary `report.md`;
- `repeats.csv` and `repeats_summary.csv` (mean/min/max);
- `gate_by_repeat.csv`.

The audit verdict is the worst repeat.

## Remediation when a check fails

* Coarsen quasi-identifiers (age and diagnosis-year bands, region not centre,
  broader subtypes). Drop free text, exact dates and IDs before training.
* Suppress or pool strata with very few patients. Do not generate "synthetic
  children" from 1–5 real cases.
* Reduce memorisation (fewer epochs, a smaller network), then re-audit.
* For a defensible formal claim, train with differential privacy (DP-SGD) and
  report ε, δ, the clipping norm, the noise multiplier and the accountant. PODE's
  DP-SGD TVAE is in `pode federated` (it also works with a single site), e.g.
  `pode federated exchange --dp --target-epsilon 3`; see
  [../federated/FEDERATED.md](../federated/FEDERATED.md). Expect a utility cost:
  at a few dozen patients per site, a strong privacy budget can remove the
  membership signal and most of the data's usefulness with it.
* Release higher-fidelity data only through controlled access: a trusted
  research environment, a data-access agreement and audit logging.

## Limitations

* The attacks are distance-based (nearest record, k-NN, key matching). Stronger
  attacks, such as shadow models, likelihood-based attacks or access to the model
  file itself, can do better. Releasing the trained `model.pkl` is a larger
  attack surface than releasing a fixed synthetic table.
* The checks are empirical, not a formal guarantee. Small holdouts give wide
  uncertainty, so treat that as a reason for conservative governance.
* Longitudinal trajectories should be assessed as a whole. A table with one row
  per patient is assessed row by row.
* Performance: distance matrices are computed in chunks on `n_jobs` threads. A
  2000 × 2000-row, 42-column assessment takes about 5 s on 4 cores. For wide omics
  tables, assess the clinical columns plus latents, or down-weight matrix columns
  with `weights`.
