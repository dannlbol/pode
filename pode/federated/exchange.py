"""Local TVAEs with privacy-gated synthetic exchange.

The lightest-weight federation: no iterative training rounds at all. Each site

1. splits its patients into a training share and a holdout share;
2. fits its own generator on the training share (``exchange.generator``:
   this package's TVAE, optionally under DP-SGD, or PODE's SDV TVAE);
3. samples a release candidate and runs the full :mod:`pode.privacy`
   assessment against its own training and holdout patients;
4. releases the synthetic rows **only** when the verdict is in
   ``exchange.release_verdicts`` (default: PASS only).

All four steps run at the site (:func:`site_local_release`); only the verdict,
the gate table, aggregate privacy metrics and (if released) the synthetic rows
leave it. Released rows are pooled with a ``_site`` tag. Synthetic cohorts from
local generators reflect *site-specific* distributions, and a site whose
release fails the gate simply contributes nothing; the release card records
both, so users of the pooled data know which sites it represents.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import SynthesizerConfig
from ..privacy.config import PrivacyConfig
from .config import FederatedConfig
from .sites import Site

INTENDED_USE = ("development environment: code and pipeline development, phenotype definitions, feasibility "
                "counts, teaching; findings must be re-run on real data (locally or federated)")
PROHIBITED_USE = ("clinical inference, effect estimates or biomarker claims from synthetic rows alone; "
                  "re-identification attempts; linkage with other data sources")


def site_local_release(data: pd.DataFrame, columns: list[str], schema, synth: SynthesizerConfig,
                       fcfg: FederatedConfig, pcfg: PrivacyConfig, seed: int, censor_threshold=None,
                       split: tuple[np.ndarray, np.ndarray] | None = None, site_name: str = "site",
                       generator: str | None = None, return_candidate: bool = False) -> dict:
    """Fit, sample and privacy-assess a local release at the site. Returns what the site publishes."""
    from ..privacy.assess import assess_privacy
    from ..privacy.audit import split_holdout
    from .generators import fit_generator

    data = data.reset_index(drop=True)
    if split is None:
        p = copy.deepcopy(pcfg)
        p.audit.holdout_frac = fcfg.exchange.holdout_frac
        split = split_holdout(data, schema, p, seed)
    tr, ho = split
    train, holdout = data.iloc[tr].reset_index(drop=True), data.iloc[ho].reset_index(drop=True)
    kind = generator or fcfg.exchange.generator
    gen = fit_generator(kind, train, columns, schema, synth, fcfg, seed, censor_threshold, site_name,
                        dp=fcfg.dp.enabled)
    n = fcfg.exchange.num_samples or len(train)
    synthetic = gen.sample(n, seed=seed)
    rcfg = copy.deepcopy(pcfg)
    rcfg.seed = seed
    report = assess_privacy(train[columns], synthetic, holdout[columns], schema, rcfg, log=lambda m: None)
    released = report.verdict in fcfg.exchange.release_verdicts
    return {"verdict": report.verdict, "released": released, "synthetic": synthetic if released else None,
            "gate": report.gate, "summary": {k: v for k, v in report.summary.items()
                                             if isinstance(v, (int, float, np.number, str, bool))},
            "n_train": len(train), "n_holdout": len(holdout), "privacy": gen.privacy, "report": report,
            "generator": kind, "fit_seconds": gen.fit_seconds,
            # The unreleased candidate never leaves a real site; the benchmark (as referee)
            # uses it for the 'local' condition, which is evaluated at the site itself.
            "candidate": synthetic if return_candidate else None}


@dataclass
class ExchangeResult:
    released: pd.DataFrame
    sites: pd.DataFrame
    card: dict
    outcomes: dict = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> Path:
        from ..privacy.assess import jsonable

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.released.to_csv(out / "released_synthetic.csv", index=False)
        self.sites.to_csv(out / "release_sites.csv", index=False)
        (out / "release_card.json").write_text(json.dumps(jsonable(self.card), indent=2))
        for name, o in self.outcomes.items():
            o["report"].save(out / "sites" / _safe(name), title=f"Local release privacy - site {name}")
        return out


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def local_exchange(sites: list[Site], columns: list[str], schema, synth: SynthesizerConfig,
                   fcfg: FederatedConfig, pcfg: PrivacyConfig | None = None, censor_threshold=None,
                   splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                   generator: str | None = None, log: Callable[[str], None] | None = print,
                   return_candidates: bool = False) -> ExchangeResult:
    """Run :func:`site_local_release` at every site and pool what was released."""
    pcfg = pcfg or PrivacyConfig()
    pcfg.n_jobs = fcfg.n_jobs
    outcomes, rows, parts = {}, [], []
    for i, s in enumerate(sites):
        split = (splits or {}).get(s.name)
        o = s.run(site_local_release, columns, schema, synth, fcfg, pcfg, fcfg.seed + i, censor_threshold,
                  split, s.name, generator, return_candidates, label="local_release")
        outcomes[s.name] = o
        eps = o["privacy"].get("epsilon_total") if o["privacy"].get("dp") else None
        rows.append({"site": s.name, "n_train": o["n_train"], "n_holdout": o["n_holdout"], "verdict": o["verdict"],
                     "released": o["released"], "rows_released": 0 if o["synthetic"] is None else len(o["synthetic"]),
                     "membership_auc": o["summary"].get("membership_auc"),
                     "dcr_p5_ratio": o["summary"].get("dcr_p5_ratio"), "epsilon": eps,
                     "generator": o["generator"]})
        if o["synthetic"] is not None:
            parts.append(o["synthetic"].assign(_site=s.name))
        if log:
            log(f"  site {s.name}: {o['verdict']} -> {'released' if o['released'] else 'withheld'}")
    table = pd.DataFrame(rows)
    released = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*columns, "_site"])
    card = {
        "sites_released": table.loc[table["released"], "site"].tolist(),
        "sites_withheld": table.loc[~table["released"], "site"].tolist(),
        "release_verdicts": fcfg.exchange.release_verdicts,
        "generator": generator or fcfg.exchange.generator,
        "dp": bool(fcfg.dp.enabled),
        "epsilon_by_site": {r["site"]: r["epsilon"] for r in rows if r["epsilon"] is not None},
        "rows": int(len(released)),
        "columns": list(columns),
        "intended_use": INTENDED_USE,
        "prohibited_use": PROHIBITED_USE,
        "caveat": ("Each site's rows come from its own generator: the pool mixes site-specific distributions "
                   "and omits withheld sites. A PASS is evidence against the tested attacks, not a guarantee."),
    }
    return ExchangeResult(released, table, card, outcomes)
