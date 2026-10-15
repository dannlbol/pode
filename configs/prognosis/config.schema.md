# prognostk configuration schema

Every prognostk function takes its defaults from a YAML config loaded with
`load_config()` (R) / `prognostk.load_config()` (Python). The *same* YAML works in both
languages. Any value can be overridden by passing it directly to a function; the config
just supplies the defaults.

The loader **validates** the config on load and raises a clear error if a required key is
missing or malformed. Only `columns.time` and `columns.status` are strictly required; every
other section has a sensible built-in default and may be omitted.

---

## `columns` — map your data's columns to prognostk roles

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `id` | string | no | Unique sample/patient identifier column. Defaults to row index if absent. |
| `time` | string | **yes** | Survival / follow-up time (numeric, > 0). |
| `status` | string | **yes** | Event indicator: `1` = event, `0` = censored. |
| `score` | string | no | Continuous biomarker / prognostic score column. |
| `group` | string | no | Pre-computed risk-tier factor column. |
| `covariates` | list[string] | no | Cox model terms. Prefix a term with `strata:` to enter it as a stratum, e.g. `"strata:subgroup"`. Defaults to `[score]` if a score is set. |
| `categorical` | list[string] | no | Categorical columns used as oncoplot tracks / distribution facets. |

## `survival`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `horizon` | number | `5` | Time at which survival is predicted / monitored. Same unit as `columns.time`. |
| `time_unit` | string | `"years"` | Axis label only. |
| `landmark_offset` | number \| null | `null` | If set (e.g. `5`), analyses condition on survival past this time and shift the clock by it (`time - offset`). |
| `bootstrap.n` | int | `1000` | Bootstrap resamples for CIs / bias correction. |
| `bootstrap.conf` | number | `0.95` | Confidence level in (0, 1). |
| `bootstrap.seed` | int | `42` | RNG seed. |

## `risk_groups`

| Key | Type | Rule |
|-----|------|------|
| `thresholds` | list[number] | Strictly **descending**, each in `(0, 1)`. Applied to predicted survival. |
| `labels` | list[string] | Length must be exactly `length(thresholds) + 1`. Highest survival tier first. |

A prediction `p` is assigned the label at the first threshold it exceeds; predictions below the
last threshold get the final label.

## `palettes`

| Key | Type | Meaning |
|-----|------|---------|
| `risk` | map{label → hex} | One colour per `risk_groups.labels` entry. |
| `cohort` | map{name → hex} | Colours for cohorts in two-cohort plots (typically `discovery`, `validation`). |
| `heatmap` | list[hex] (len 2–3) | Continuous ramp `(low, [mid,] high)` for heatmaps / score gradients. |
| `categorical` | map{column → map{level → hex}} | Per-variable colour maps for oncoplot categorical tracks. A track whose column is absent here gets an auto-generated palette. |

## `oncoplot.tracks`

Ordered list of annotation tracks drawn above the heatmap body. Each track is a map:

| Key | Type | Applies to | Meaning |
|-----|------|-----------|---------|
| `name` | string | all | Track label shown on the plot. |
| `column` | string | all | Column in the data providing the track values. |
| `type` | `points`\|`categorical`\|`line` | all | Annotation kind. |
| `baseline` | number | points | Optional reference line drawn across the track. |
| `ylim` | list[number] (len 2) | points/line | Optional y-axis limits. |
| `status_col` | string | points | Optional event column; points are styled/coloured by event status (used for the OS track). |

## `figure`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `page.width_in` | number | `7.09` | Output width (inches). |
| `page.height_in` | number | `10.51` | Output height (inches). |
| `page.base_size` | number | `8` | Base font size (pt). |
| `device` | `pdf`\|`cairo_pdf`\|`png` | `pdf` | Output device. `cairo_pdf` renders italic gene names correctly. |
| `dpi` | int | `300` | Raster resolution (png only). |
| `outdir` | string | `"./figures"` | Directory figures are written to (created if absent). |
