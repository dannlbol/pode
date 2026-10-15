"""``pode federated signature run | verify | benchmark``: federated MetaCPG."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cli import _log


def _cfg(args) -> dict:
    from metacpg.core import read_config

    cfg = read_config(args.config)
    if args.engine:
        cfg.setdefault("execution", {})["engine"] = args.engine
    return cfg


def _write(cfg: dict, result: dict, out: Path, ledger) -> Path:
    from metacpg.core import _write_results

    c = copy.deepcopy(cfg)
    c["output_dir"] = str(out)
    path = _write_results(c, result, float(result.get("elapsed_seconds", 0.0)))
    ledger.save(out / "disclosure_ledger.csv")
    ledger.summary().to_csv(out / "disclosure_summary.csv", index=False)
    return path


def cmd_run(args) -> int:
    from .ledger import DisclosureLedger
    from .signatures import federated_loso, federated_metacpg, loso_final

    cfg, ledger, out = _cfg(args), DisclosureLedger(), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.cv == "loso":
        res = federated_loso(cfg, args.site_column, ledger, engine=args.engine, log=_log)
        res["curve"].to_csv(out / "loso_curve.csv", index=False)
        res["per_site"].to_csv(out / "loso_per_site.csv", index=False)
        n = args.feature_count or int(res["curve"].sort_values(["c_index_stratified", "feature_count"],
                                                               ascending=[False, True]).iloc[0]["feature_count"])
        result = loso_final(res["fed"], n, res["curve"])
    else:
        result = federated_metacpg(cfg, args.site_column, ledger, args.metrics, args.feature_count,
                                   args.extraction_only, args.engine, log=_log)
    path = _write(cfg, result, out, ledger)
    print(f"Federated MetaCPG ({args.cv}, {result.get('design')}): {result['selected_count']} CpGs; "
          f"performance {json.dumps({k: round(v, 4) for k, v in result['performance'].items() if isinstance(v, float)})}")
    print(f"Wrote {path}")
    return 0


def compare_results(central: dict, fed: dict) -> dict:
    """Numerical agreement between a centralised and a federated MetaCPG result."""
    out: dict = {}
    if isinstance(central.get("ranking"), pd.DataFrame) and isinstance(fed.get("ranking"), pd.DataFrame):
        a, b = central["ranking"], fed["ranking"]
        out["ranking_order_identical"] = a["feature"].tolist() == b["feature"].tolist()
        top = a["feature"].head(100)
        out["ranking_top100_identical"] = top.tolist() == b["feature"].head(100).tolist()
        out["coef_max_abs_diff"] = float((a.set_index("feature")["coef"]
                                          - b.set_index("feature")["coef"]).abs().max())
    for key in ("scores", "extraction_scores"):
        if isinstance(central.get(key), pd.DataFrame) and isinstance(fed.get(key), pd.DataFrame) \
                and "risk_score" in fed[key]:
            s1 = central[key].set_index(["sample_id", "feature_count"])["risk_score"]
            s2 = fed[key].set_index(["sample_id", "feature_count"])["risk_score"].reindex(s1.index)
            out[f"{key}_max_abs_diff"] = float((s1 - s2).abs().max())
            out[f"{key}_rows"] = int(len(s1))
    if isinstance(central.get("metrics"), pd.DataFrame) and isinstance(fed.get("metrics"), pd.DataFrame):
        m1 = central["metrics"].set_index("feature_count")
        m2 = fed["metrics"].set_index("feature_count").reindex(m1.index)
        cols = [c for c in m1.columns if c in m2.columns]
        out["metrics_max_abs_diff"] = float((m1[cols] - m2[cols]).abs().max().max())
    f1 = central["selected_features"]["feature"].tolist()
    f2 = fed["selected_features"]["feature"].tolist()
    out["selected_identical"] = f1 == f2
    out["selected_overlap"] = len(set(f1) & set(f2))
    out["selected_count"] = len(f1)
    w1 = central["selected_features"].set_index("feature")["standardized_coef"]
    w2 = fed["selected_features"].set_index("feature")["standardized_coef"].reindex(w1.index)
    out["weight_max_abs_diff"] = float((w1 - w2).abs().max())
    for k in ("center", "scale"):
        out[f"scaling_{k}_abs_diff"] = abs(float(central["scaling"][k]) - float(fed["scaling"][k]))
    perf = {k: (central["performance"].get(k), fed["performance"].get(k)) for k in central["performance"]}
    out["performance"] = perf
    return out


def compare_reference(fed: dict, model_path: str) -> dict:
    ref = json.loads(Path(model_path).read_text())
    rw = {x["feature"]: x["weight"] for x in ref["signature"]}
    fw = dict(zip(fed["selected_features"]["feature"], fed["selected_features"]["standardized_coef"]))
    common = set(rw) & set(fw)
    return {"reference_model": str(model_path), "reference_features": len(rw),
            "same_features": set(rw) == set(fw), "same_order": list(rw) == list(fw), "overlap": len(common),
            "weight_max_abs_diff": max((abs(rw[k] - fw[k]) for k in common), default=float("nan")),
            "center_abs_diff": abs(ref["scaling"]["center"] - fed["scaling"]["center"]),
            "scale_abs_diff": abs(ref["scaling"]["scale"] - fed["scaling"]["scale"])}


def cmd_verify(args) -> int:
    import time

    from metacpg.core import extract_signature, fit_metacpg

    from ..privacy.assess import jsonable
    from .ledger import DisclosureLedger
    from .signatures import federated_metacpg

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    report: dict = {"config": str(args.config), "site_column": args.site_column}
    if not args.skip_central:
        ccfg = copy.deepcopy(cfg)
        ccfg.setdefault("execution", {})["engine"] = args.central_engine
        ccfg["output_dir"] = str(out / "central")
        _log(f"central MetaCPG (engine {args.central_engine}) ...")
        t0 = time.perf_counter()
        if args.extraction_only:
            central = extract_signature(ccfg, feature_count=args.feature_count)
        else:
            central = fit_metacpg(ccfg, feature_count=args.feature_count)
        report["central_seconds"] = time.perf_counter() - t0
    ledger = DisclosureLedger()
    _log("federated MetaCPG ...")
    t0 = time.perf_counter()
    fed = federated_metacpg(copy.deepcopy(cfg), args.site_column, ledger, "pooled", args.feature_count,
                            args.extraction_only, args.engine, log=_log)
    report["federated_seconds"] = time.perf_counter() - t0
    _write(cfg, fed, out / "federated", ledger)
    if not args.skip_central:
        report["agreement"] = compare_results(central, fed)
    if args.reference_model:
        report["reference"] = compare_reference(fed, args.reference_model)
    report["disclosure"] = ledger.summary().to_dict(orient="records")
    (out / "verify_report.json").write_text(json.dumps(jsonable(report), indent=2))
    (out / "verify_report.md").write_text(render_verify(report))
    ok = report.get("agreement", {}).get("selected_identical", True) and \
        report.get("reference", {}).get("same_features", True)
    print(render_verify(report))
    return 0 if ok else 4


def render_verify(r: dict) -> str:
    lines = ["# Federated MetaCPG verification", "",
             f"Config: `{r['config']}`; sites from `{r.get('site_column') or 'group_column'}`.", ""]
    a = r.get("agreement")
    if a:
        verdict = "IDENTICAL SELECTION" if a["selected_identical"] else "SELECTION DIFFERS"
        lines += [f"**Central vs federated: {verdict}** ({a['selected_overlap']}/{a['selected_count']} CpGs shared).",
                  "", "| quantity | max abs difference |", "|---|---|"]
        for k, v in a.items():
            if k.endswith("diff"):
                lines.append(f"| {k} | {v:.3g} |")
        lines += ["", "Differences around 1e-15 are floating-point summation order (the federated sum adds site "
                  "contributions in a different order); around 1e-8 when the central run used the scipy `exact` "
                  "engine (a different optimiser converging to the same estimate).", ""]
        lines += ["| performance | central | federated |", "|---|---|---|"]
        for k, (c, f) in a["performance"].items():
            if isinstance(c, (int, float)):
                lines.append(f"| {k} | {c:.6g} | {f:.6g} |" if isinstance(f, (int, float)) else f"| {k} | {c} | {f} |")
        lines.append("")
    ref = r.get("reference")
    if ref:
        lines += ["## Against the frozen reference model", "",
                  f"- model: `{ref['reference_model']}`",
                  f"- same CpG set: **{ref['same_features']}** (overlap {ref['overlap']}/{ref['reference_features']}); "
                  f"same order: {ref['same_order']}",
                  f"- max weight difference: {ref['weight_max_abs_diff']:.3g}; centre difference "
                  f"{ref['center_abs_diff']:.3g}; scale difference {ref['scale_abs_diff']:.3g}", ""]
    lines += ["## What crossed site boundaries", "", "| operation | calls | bytes | per patient |", "|---|---|---|---|"]
    for d in r.get("disclosure", []):
        lines.append(f"| {d['op']} | {d['calls']} | {d['bytes']:,} | {'yes' if d['per_patient'] else 'no'} |")
    lines += ["", "`loo_heldout_score` (pooled metrics) and `final_risk_scores_site_output` are per-patient; in a "
              "deployment the latter stay at their site and the former can be replaced by `--metrics within_site`.",
              f"", f"Timing: federated {r.get('federated_seconds', 0):.0f}s"
              + (f", central {r['central_seconds']:.0f}s" if "central_seconds" in r else "") + "."]
    return "\n".join(lines) + "\n"


def cmd_benchmark(args) -> int:
    from .ledger import DisclosureLedger
    from .signatures import federated_loso, site_local_signature_transfer

    cfg, ledger, out = _cfg(args), DisclosureLedger(), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = None
    if args.counts:
        counts = [int(x) for x in args.counts.split(",")]
    res = federated_loso(cfg, args.site_column, ledger, counts=counts, engine=args.engine, log=_log)
    fed = res["fed"]
    counts = sorted(res["curve"]["feature_count"].unique().tolist())
    transfer = site_local_signature_transfer(fed, counts)
    n = args.feature_count or cfg.get("selection", {}).get("feature_count") or \
        int(res["curve"].sort_values("c_index_stratified", ascending=False).iloc[0]["feature_count"])
    res["curve"].to_csv(out / "loso_curve.csv", index=False)
    res["per_site"].to_csv(out / "loso_per_site.csv", index=False)
    transfer.to_csv(out / "site_local_transfer.csv", index=False)
    ledger.save(out / "disclosure_ledger.csv")
    table = benchmark_table(res["per_site"], transfer, int(n))
    table.to_csv(out / "benchmark_at_count.csv", index=False)
    (out / "report.md").write_text(render_benchmark(res, transfer, table, int(n), fed, ledger))
    print(table.to_string(index=False))
    print(f"See {out / 'report.md'}")
    return 0


def benchmark_table(per_site: pd.DataFrame, transfer: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for h, g in per_site[per_site["feature_count"] == n].groupby("heldout_site"):
        r = g.iloc[0]
        loc = transfer[(transfer["heldout_site"] == h) & (transfer["feature_count"] == n)]
        rows.append({"heldout_site": h, "n": r["n"], "events": r["events"],
                     "federated_c": r["c_index"], "federated_auc": r["auc"], "federated_hr_p": r["hr_p"],
                     "best_single_site_c": loc["c_index"].max() if len(loc) else np.nan,
                     "mean_single_site_c": loc["c_index"].mean() if len(loc) else np.nan,
                     "worst_single_site_c": loc["c_index"].min() if len(loc) else np.nan})
    return pd.DataFrame(rows)


def render_benchmark(res, transfer, table, n, fed, ledger) -> str:
    c = res["curve"]
    best = c.sort_values("c_index_stratified", ascending=False).iloc[0]
    lines = ["# Federated MetaCPG: leave-one-site-out benchmark", "",
             f"Sites: {', '.join(f'{s.name} (n={fed.n[s.name]})' for s in fed.sites)}; design `{fed.design}`.", "",
             "Each site in turn is held out entirely. The signature is built from the other sites' statistics "
             "only and scored at the held-out site, which reports aggregates (pair counts, AUC, the score's "
             "hazard ratio), never patient rows. **C-index reading:** 0.5 = no discrimination, 0.6 modest, "
             "0.7 good. The stratified C pools concordant pairs within each held-out site.", "",
             f"Best LOSO count: **{int(best['feature_count'])}** CpGs (stratified C {best['c_index_stratified']:.3f}).",
             "", "## LOSO curve", "", "| CpGs | stratified C | mean AUC | worst-site C | selection Jaccard |",
             "|---|---|---|---|---|"]
    for _, r in c.iterrows():
        lines.append(f"| {int(r['feature_count'])} | {r['c_index_stratified']:.3f} | {r['auc_mean']:.3f} | "
                     f"{r['c_index_worst_site']:.3f} | {r['mean_jaccard']:.2f} |")
    lines += ["", "Selection Jaccard: overlap of the top-n CpG sets built without each site (1 = the same CpGs "
              "whichever site is left out).", "",
              f"## Federated vs single-site signatures at {n} CpGs", "",
              "Single-site: a signature built from one other site alone, scored at the held-out site. Federating "
              "pays off when the federated C beats the typical single-site C.", "",
              "| held-out site | n | events | federated C | AUC | HR p | best single-site C | mean | worst |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['heldout_site']} | {int(r['n'])} | {int(r['events'])} | {r['federated_c']:.3f} | "
                     f"{r['federated_auc']:.3f} | {r['federated_hr_p']:.3g} | {r['best_single_site_c']:.3f} | "
                     f"{r['mean_single_site_c']:.3f} | {r['worst_single_site_c']:.3f} |")
    s = ledger.summary()
    lines += ["", "## What crossed site boundaries", "", "| operation | calls | bytes | per patient |",
              "|---|---|---|---|"]
    for _, d in s.iterrows():
        lines.append(f"| {d['op']} | {d['calls']} | {d['bytes']:,} | {'yes' if d['per_patient'] else 'no'} |")
    return "\n".join(lines) + "\n"


def register(subparsers) -> None:
    p = subparsers.add_parser("signature", help="Federated MetaCPG signatures: run / verify / benchmark.")
    sub = p.add_subparsers(dest="signature_command", required=True)

    def common(q):
        q.add_argument("--config", required=True, help="MetaCPG config (schema_version 1, legacy_exact).")
        q.add_argument("--site-column", default=None,
                       help="Phenotype column naming each sample's site (pooled designs; meta designs use "
                            "their group_column).")
        q.add_argument("--engine", default=None, choices=["batched", "gpu", "auto"],
                       help="Batched Cox backend for the federated fits (default: from config; exact -> batched).")
        q.add_argument("--feature-count", type=int, default=None)
        q.add_argument("--output-dir", required=True)

    r = sub.add_parser("run", help="Federated MetaCPG search + extraction (writes a normal model.json).")
    common(r)
    r.add_argument("--cv", default="exact_loo", choices=["exact_loo", "loso"])
    r.add_argument("--metrics", default="pooled", choices=["pooled", "within_site"])
    r.add_argument("--extraction-only", action="store_true")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verify", help="Run central and federated MetaCPG and compare them.")
    common(v)
    v.add_argument("--central-engine", default="batched", choices=["batched", "exact", "gpu"])
    v.add_argument("--extraction-only", action="store_true")
    v.add_argument("--skip-central", action="store_true", help="Only compare with --reference-model.")
    v.add_argument("--reference-model", default=None, help="A frozen model.json to compare against.")
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("benchmark", help="Leave-one-site-out: federated vs single-site signatures.")
    common(b)
    b.add_argument("--counts", default=None, help="Comma-separated feature counts (default: the search grid).")
    b.set_defaults(func=cmd_benchmark)
