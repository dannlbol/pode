from __future__ import annotations

import importlib.util
from typing import Any

from .base import CohortUnsupportedError, MethodSpec
from .models import (
    CoxBoostModel,
    DeepCoxModel,
    FrailtyCoxModel,
    HierarchicalCoxModel,
    LinearCoxModel,
    PLSCoxModel,
    SksurvModel,
    StabilityBoostModel,
    graph_edges,
)


def _linear(ctx: dict[str, Any], cfg: dict[str, Any]):
    return LinearCoxModel(
        l1=cfg.get("l1", .002), l2=cfg.get("l2", .01),
        max_iter=cfg.get("max_iter", 500), learning_rate=cfg.get("learning_rate", .2),
        n_omics=ctx["n_omics"],
    )


def _boost(ctx, cfg):
    return CoxBoostModel(steps=cfg.get("steps", 100), learning_rate=cfg.get("learning_rate", .05),
                         n_omics=ctx["n_omics"])


def _stability(ctx, cfg):
    return StabilityBoostModel(
        repetitions=cfg.get("repetitions", 20), threshold=cfg.get("threshold", .6),
        steps=cfg.get("steps", 60), learning_rate=cfg.get("learning_rate", .05),
        seed=ctx["seed"], n_omics=ctx["n_omics"],
    )


def _pls(ctx, cfg):
    return PLSCoxModel(components=cfg.get("components", 3), n_omics=ctx["n_omics"])


def _net(ctx, cfg):
    edges = graph_edges(cfg.get("graph_path"), ctx["feature_names"])
    return LinearCoxModel(
        l1=cfg.get("l1", .001), l2=cfg.get("l2", .01), n_omics=ctx["n_omics"],
        graph_edges=edges, graph_strength=cfg.get("graph_strength", .1),
        max_iter=cfg.get("max_iter", 500),
    )


def _sksurv(name):
    return lambda ctx, cfg: SksurvModel(
        name, n_omics=ctx["n_omics"], seed=ctx["seed"], **cfg.get("params", {})
    )


def _deep(name):
    return lambda ctx, cfg: DeepCoxModel(
        name, n_omics=ctx["n_omics"], block_slices=ctx["block_slices"],
        n_tasks=ctx.get("n_tasks", 1), task_index=ctx.get("task_index", 0),
        epochs=cfg.get("epochs", 80), hidden=cfg.get("hidden", 32),
        learning_rate=cfg.get("learning_rate", 1e-3), seed=ctx["seed"],
        pathway_path=cfg.get("pathway_path"), cohort_aware=name == "maml_cox",
    )


def _hierarchical(ctx, cfg):
    return HierarchicalCoxModel(n_omics=ctx["n_omics"], **cfg)


def _frailty(ctx, cfg):
    return FrailtyCoxModel(n_omics=ctx["n_omics"], **cfg)


_SPECS = [
    MethodSpec("elastic_net_cox", "Elastic-net Cox", "stable", "penalised", _linear,
               description="Sparse Cox partial-likelihood model with unpenalised clinical covariates."),
    MethodSpec("coxboost", "Component-wise CoxBoost", "stable", "boosting", _boost,
               description="PODE component-wise Cox boosting with mandatory covariate updates."),
    MethodSpec("stability_coxboost", "Stability selection + CoxBoost", "stable", "stability", _stability,
               description="Complementary half-sample selection frequencies around CoxBoost."),
    MethodSpec("pls_cox_multiblock", "Multiblock PLS-Cox", "stable", "latent", _pls,
               description="PODE PLS-Cox adaptation for correlated multi-omics blocks."),
    MethodSpec("random_survival_forest", "Random survival forest", "stable", "ensemble",
               _sksurv("random_survival_forest"), optional_dependency="scikit-survival",
               cohort_support="none", cohort_stratifies=False),
    MethodSpec("gradient_boosted_survival", "Gradient-boosted survival", "stable", "ensemble",
               _sksurv("gradient_boosted_survival"), optional_dependency="scikit-survival",
               cohort_support="none", cohort_stratifies=False),
    MethodSpec("survival_svm", "Survival SVM", "stable", "kernel",
               _sksurv("survival_svm"), optional_dependency="scikit-survival",
               cohort_support="none", cohort_stratifies=False),
    MethodSpec("deepsurv", "DeepSurv", "stable", "deep", _deep("deepsurv")),
    MethodSpec("autoencoder_cox", "Multi-omics autoencoder-Cox", "stable", "deep", _deep("autoencoder_cox")),
    MethodSpec("net_cox", "Graph-regularised Net-Cox", "experimental", "penalised", _net,
               requires_graph=True, description="Sparse Cox model with a graph Laplacian penalty."),
    MethodSpec("cox_nnet", "Pathway-masked Cox-nnet", "experimental", "deep", _deep("cox_nnet"),
               requires_pathways=True),
    MethodSpec("surv_transformer", "SurvTRACE-style transformer", "experimental", "deep",
               _deep("surv_transformer")),
    MethodSpec("prototype_contrastive_survival", "PCLSurv-style prototypical survival", "experimental", "deep",
               _deep("prototype_contrastive_survival")),
    MethodSpec("moli_survival", "MOLI-style late-integration survival", "experimental", "deep",
               _deep("moli_survival")),
    # Models per-cohort coefficient deviations around a shared effect but keeps one shared
    # baseline hazard, so cohort is additionally added to the strata.
    MethodSpec("hierarchical_cox", "Hierarchical multi-study Cox", "experimental", "multi-study",
               _hierarchical, requires_cohort=True, cohort_support="explicit"),
    # Partitions its meta-learning inner loop by cohort already.
    MethodSpec("maml_cox", "MAML-style Cox meta-learning", "experimental", "meta-learning",
               _deep("maml_cox"), requires_cohort=True,
               cohort_support="explicit", cohort_stratifies=False),
    # Puts cohort indicators on the linear predictor; a stratified baseline would absorb them.
    MethodSpec("frailty_cox", "Penalised shared-frailty Cox", "experimental", "multi-study",
               _frailty, requires_cohort=True,
               cohort_support="explicit", cohort_stratifies=False),
]

REGISTRY = {spec.name: spec for spec in _SPECS}


def method_catalog() -> list[dict[str, Any]]:
    rows = []
    for spec in _SPECS:
        available = True
        reason = None
        if spec.optional_dependency == "scikit-survival" and importlib.util.find_spec("sksurv") is None:
            available = False
            reason = "install pode[signatures-advanced]"
        rows.append({
            "name": spec.name, "title": spec.title, "maturity": spec.maturity,
            "family": spec.family, "available": available, "unavailable_reason": reason,
            "requires_cohort": spec.requires_cohort, "requires_graph": spec.requires_graph,
            "requires_pathways": spec.requires_pathways,
            "cohort_support": spec.cohort_support,
            "supports_missing_blocks": spec.supports_missing_blocks,
            "description": spec.description,
        })
    return rows


#: Human-readable explanation of each ``MethodSpec.cohort_support`` level.
COHORT_SUPPORT_NOTES = {
    "stratified": "cohort enters the Cox partial likelihood as an extra stratum "
                  "(shared coefficients, one baseline hazard per cohort)",
    "explicit": "the model represents cohort structure directly",
    "none": "the method cannot represent cohort structure; samples are pooled and "
            "differences in baseline survival between cohorts are not accounted for",
}


def cohort_report(name: str, has_cohort: bool,
                  cohort_validation: str = "stratified_kfold") -> dict[str, Any]:
    """Describe how ``name`` treats ``phenotype.cohort_column`` for the current run.

    Returns ``cohort_handling`` (``none``/``stratified``/``explicit``/``ignored``) and a
    ``cohort_warning`` that is non-null exactly when the configured cohort structure is not
    being honoured, so the caveat travels with the results instead of being silently lost.
    """
    if not has_cohort:
        return {"cohort_handling": "none", "cohort_warning": None,
                "cohort_validation": cohort_validation}
    support = REGISTRY[name].cohort_support
    warning = None
    if support == "none":
        warning = (f"{name} ignores phenotype.cohort_column: {COHORT_SUPPORT_NOTES['none']}. "
                   "Its scores are not comparable with the cohort-aware methods; set "
                   "phenotype.cohort_handling: skip to exclude it instead.")
    elif cohort_validation == "leave_one_cohort_out":
        warning = (f"{name} is being validated leave-one-cohort-out, which measures "
                   "cross-cohort transfer rather than performance on the pooled training "
                   "population; unset evaluation.cohort_validation for the default.")
    return {"cohort_handling": "ignored" if support == "none" else support,
            "cohort_warning": warning, "cohort_validation": cohort_validation}


def build_estimator(name: str, context: dict[str, Any], config: dict[str, Any]):
    if name not in REGISTRY:
        choices = ", ".join(REGISTRY)
        raise ValueError(f"unknown signature method {name!r}; choose one of: {choices}")
    spec = REGISTRY[name]
    if spec.requires_cohort:
        cohort = context.get("cohort")
        if cohort is None or len(set(cohort)) < 2:
            raise ValueError(f"{name} requires phenotype.cohort_column with at least two cohorts")
    if (spec.cohort_support == "none" and context.get("has_cohort")
            and context.get("cohort_handling") == "skip"):
        raise CohortUnsupportedError(
            f"{name} cannot honour phenotype.cohort_column ({COHORT_SUPPORT_NOTES['none']}); "
            "skipped because phenotype.cohort_handling is 'skip'"
        )
    if spec.optional_dependency == "scikit-survival" and context.get("strata") is not None:
        raise ValueError(f"{name} does not support covariate mode: strata (no stratified Cox baseline hazard)")
    if spec.requires_graph and not config.get("graph_path"):
        raise ValueError(f"{name} requires methods.config.{name}.graph_path")
    if spec.requires_pathways and not config.get("pathway_path"):
        raise ValueError(f"{name} requires methods.config.{name}.pathway_path")
    return spec.factory(context, dict(config))
