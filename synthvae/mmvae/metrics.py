"""Feature-space fidelity metrics for multimodal synthetic cohorts.

Every fitted representation is learned from real data only.  UMAP is deliberately
absent: these metrics quantify the structures that a visualization can reveal
without allowing the visualization to become the tuning target.
"""
from __future__ import annotations

import warnings
import weakref

import numpy as np


_REAL_PREPARATION_CACHE = {}


def _array(frame):
    return frame.to_numpy(dtype=float) if hasattr(frame, "to_numpy") else np.asarray(frame, dtype=float)


def _safe_corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _finite_extreme(values, fn, fallback=float("nan")):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(fn(values)) if len(values) else float(fallback)


def _top_standardized(real, synthetic, max_features=512):
    r, s = _array(real), _array(synthetic)
    if r.ndim != 2 or s.ndim != 2 or r.shape[1] != s.shape[1]:
        raise ValueError("real and synthetic modality feature counts differ")
    variances = np.nanvar(r, axis=0)
    idx = np.argsort(variances)[-min(max_features, r.shape[1]):]
    mean, std = np.nanmean(r[:, idx], axis=0), np.nanstd(r[:, idx], axis=0)
    keep = np.isfinite(std) & (std > 1e-10)
    if not np.any(keep):
        raise ValueError("real modality has no variable finite features")
    idx = idx[keep]
    mean, std = mean[keep], std[keep]
    return (r[:, idx] - mean) / std, (s[:, idx] - mean) / std, idx


def _prepare_real(real, max_features):
    """Cache real-only selection/scaling/PCA inputs across sweep candidates."""
    key = (id(real), int(max_features))
    cached = _REAL_PREPARATION_CACHE.get(key)
    if cached is not None and cached["reference"]() is real:
        return cached
    raw = _array(real)
    variances = np.nanvar(raw, axis=0)
    idx = np.argsort(variances)[-min(max_features, raw.shape[1]):]
    mean, std = np.nanmean(raw[:, idx], axis=0), np.nanstd(raw[:, idx], axis=0)
    keep = np.isfinite(std) & (std > 1e-10)
    if not np.any(keep):
        raise ValueError("real modality has no variable finite features")
    idx, mean, std = idx[keep], mean[keep], std[keep]
    prepared = {
        "reference": weakref.ref(real), "raw": raw, "idx": idx,
        "mean": mean, "std": std, "standardized": (raw[:, idx] - mean) / std,
    }
    _REAL_PREPARATION_CACHE[key] = prepared
    return prepared


def _transform_prepared(synthetic, prepared):
    raw = _array(synthetic)
    return raw, (raw[:, prepared["idx"]] - prepared["mean"]) / prepared["std"]


def _mean_nearest(a, b, self_match=False):
    from sklearn.neighbors import NearestNeighbors
    k = 2 if self_match and len(b) > 1 else 1
    distances = NearestNeighbors(n_neighbors=k).fit(b).kneighbors(a, return_distance=True)[0]
    return float(np.mean(distances[:, -1]))


def _source_neighbour_metrics(r, s, k=15, prepared=None):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import NearestNeighbors

    # Fifty components better approximates the neighbour graph used by the
    # fixed-real high-dimensional UMAP while remaining cheap enough for tuning.
    n_components = max(1, min(50, r.shape[1], len(r) - 1))
    pca = prepared.get("pca") if prepared is not None else None
    rp = prepared.get("projection") if prepared is not None else None
    if pca is None or rp is None:
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0).fit(r)
        rp = pca.transform(r)
        if prepared is not None:
            prepared["pca"], prepared["projection"] = pca, rp
    sp = pca.transform(s)
    x = np.vstack([rp, sp])
    y = np.r_[np.zeros(len(r), dtype=int), np.ones(len(s), dtype=int)]
    kk = min(k, len(x) - 1)
    neighbours = NearestNeighbors(n_neighbors=kk + 1).fit(x).kneighbors(
        x, return_distance=False)[:, 1:]
    mixing = float(np.mean(y[neighbours] != y[:, None]))
    expected = 2.0 * y.mean() * (1.0 - y.mean())
    mixing_normalized = mixing / max(expected, 1e-8)

    nearest_real = NearestNeighbors(n_neighbors=min(2, len(r))).fit(r)
    s_to_r = nearest_real.kneighbors(s, return_distance=True)
    coverage = len(np.unique(s_to_r[1][:, 0])) / max(len(r), 1)
    dcr = float(np.mean(s_to_r[0][:, 0]))
    if s_to_r[0].shape[1] > 1:
        nndr = float(np.mean(s_to_r[0][:, 0] / (s_to_r[0][:, 1] + 1e-8)))
    else:
        nndr = float("nan")

    auc = float("nan")
    minority = int(min(np.bincount(y)))
    if minority >= 3:
        folds = min(5, minority)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc = float(np.mean(cross_val_score(
                    LogisticRegression(max_iter=1000), x, y,
                    cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                    scoring="roc_auc",
                )))
        except Exception:
            pass
    return {
        "knn_mixing": mixing,
        "knn_mixing_normalized": mixing_normalized,
        "real_coverage": float(coverage),
        "source_auc": auc,
        "distance_to_closest_real": dcr,
        "nearest_neighbour_distance_ratio": nndr,
    }, rp, sp, pca


def _manifold_precision_recall(r, s, k=5):
    """PRDC-style support overlap in a real-fitted representation."""
    from scipy.spatial.distance import cdist
    from sklearn.neighbors import NearestNeighbors

    if min(len(r), len(s)) < 2:
        return float("nan"), float("nan")
    kr, ks = min(k + 1, len(r)), min(k + 1, len(s))
    real_radius = NearestNeighbors(n_neighbors=kr).fit(r).kneighbors(
        r, return_distance=True)[0][:, -1]
    synth_radius = NearestNeighbors(n_neighbors=ks).fit(s).kneighbors(
        s, return_distance=True)[0][:, -1]
    distances = cdist(s, r)
    precision = np.mean(np.any(distances <= real_radius[None, :], axis=1))
    recall = np.mean(np.any(distances <= synth_radius[:, None], axis=0))
    return float(precision), float(recall)


def _conditional_source_overlap(rp, sp, real_groups=None, synthetic_groups=None,
                                k=15):
    """Source overlap within biological groups, including boundary-shell tests.

    The original conditional metrics are weighted subgroup averages.  They are
    retained for compatibility, but averages can hide a source shell in a small
    subgroup.  The ``*_min``, ``*_max`` and ``*_worst`` values below are hard
    worst-subgroup diagnostics used by the calibration workflow and promotion
    gates.
    """
    from scipy.stats import wasserstein_distance
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import NearestNeighbors

    precision, recall = _manifold_precision_recall(rp, sp)
    result = {
        "subgroup_knn_mixing_normalized": float("nan"),
        "subgroup_source_auc": float("nan"),
        "subgroup_radial_wasserstein": float("nan"),
        "subgroup_knn_mixing_min": float("nan"),
        "subgroup_source_auc_max": float("nan"),
        "subgroup_radial_wasserstein_max": float("nan"),
        "subgroup_radial_median_ratio_worst": float("nan"),
        "subgroup_radial_p90_ratio_worst": float("nan"),
        "subgroup_local_density_ratio_worst": float("nan"),
        "subgroup_local_density_wasserstein_max": float("nan"),
        "subgroup_manifold_precision_min": float("nan"),
        "subgroup_manifold_recall_min": float("nan"),
        "manifold_precision": precision,
        "manifold_recall": recall,
    }
    if real_groups is None or synthetic_groups is None:
        return result
    rg = np.asarray(real_groups).astype(str)
    sg = np.asarray(synthetic_groups).astype(str)
    groups = sorted(set(rg) & set(sg))
    mix_values, auc_values, radial_values, weights = [], [], [], []
    worst_mix_values, worst_auc_values, worst_radial_values = [], [], []
    radial_median_ratios, radial_p90_ratios = [], []
    density_ratios, density_wassersteins = [], []
    group_precisions, group_recalls = [], []

    def symmetric_ratio(left, right):
        ratio = float(left) / max(float(right), 1e-8)
        return max(ratio, 1.0 / max(ratio, 1e-8))

    for group in groups:
        r, s = rp[rg == group], sp[sg == group]
        if min(len(r), len(s)) < 3:
            continue
        x = np.vstack([r, s])
        y = np.r_[np.zeros(len(r), dtype=int), np.ones(len(s), dtype=int)]
        kk = min(k, len(x) - 1)
        neighbours = NearestNeighbors(n_neighbors=kk + 1).fit(x).kneighbors(
            x, return_distance=False)[:, 1:]
        mixing = float(np.mean(y[neighbours] != y[:, None]))
        expected = 2.0 * y.mean() * (1.0 - y.mean())
        mix_values.append(mixing / max(expected, 1e-8))

        auc = float("nan")
        minority = min(len(r), len(s))
        if minority >= 3:
            try:
                folds = min(5, minority)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auc = float(np.mean(cross_val_score(
                        LogisticRegression(max_iter=1000), x, y,
                        cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                        scoring="roc_auc")))
                # Source direction is arbitrary; 0.5 is ideal and 1.0 is fully
                # separable, whichever label the classifier happens to invert.
                auc = 0.5 + abs(auc - 0.5)
            except Exception:
                pass
        auc_values.append(auc)

        centre = r.mean(0)
        rr = np.linalg.norm(r - centre, axis=1)
        sr = np.linalg.norm(s - centre, axis=1)
        scale = np.median(rr)
        radial_values.append(float(wasserstein_distance(rr, sr) / max(scale, 1e-8)))

        # Compare local support density using distances to the same real
        # subgroup manifold.  Real distances are leave-one-out; synthetic
        # distances use the corresponding kth real neighbour.
        local_k = min(5, len(r) - 1)
        real_local = NearestNeighbors(n_neighbors=local_k + 1).fit(r).kneighbors(
            r, return_distance=True)[0][:, -1]
        synth_local = NearestNeighbors(n_neighbors=local_k).fit(r).kneighbors(
            s, return_distance=True)[0][:, -1]
        # Worst-subgroup extrema are unstable with only three or four held-out
        # patients. Keep those groups in the weighted compatibility metrics but
        # require at least eight real and synthetic observations for hard
        # extrema used during three-fold tuning.
        if min(len(r), len(s)) >= 8:
            worst_mix_values.append(mix_values[-1])
            worst_auc_values.append(auc_values[-1])
            worst_radial_values.append(radial_values[-1])
            radial_median_ratios.append(symmetric_ratio(
                np.median(sr), np.median(rr)))
            radial_p90_ratios.append(symmetric_ratio(
                np.quantile(sr, 0.9), np.quantile(rr, 0.9)))
            density_ratios.append(symmetric_ratio(
                np.median(synth_local), np.median(real_local)))
            density_wassersteins.append(float(wasserstein_distance(
                np.log(real_local + 1e-8), np.log(synth_local + 1e-8))))
            group_precision, group_recall = _manifold_precision_recall(r, s)
            group_precisions.append(group_precision)
            group_recalls.append(group_recall)
        weights.append(len(r) + len(s))

    if weights:
        w = np.asarray(weights, dtype=float)
        result["subgroup_knn_mixing_normalized"] = float(np.average(mix_values, weights=w))
        ok = np.isfinite(auc_values)
        if np.any(ok):
            result["subgroup_source_auc"] = float(
                np.average(np.asarray(auc_values)[ok], weights=w[ok]))
        result["subgroup_radial_wasserstein"] = float(
            np.average(radial_values, weights=w))
        result["subgroup_knn_mixing_min"] = _finite_extreme(worst_mix_values, np.min)
        result["subgroup_source_auc_max"] = _finite_extreme(worst_auc_values, np.max)
        result["subgroup_radial_wasserstein_max"] = _finite_extreme(
            worst_radial_values, np.max)
        result["subgroup_radial_median_ratio_worst"] = _finite_extreme(
            radial_median_ratios, np.max)
        result["subgroup_radial_p90_ratio_worst"] = _finite_extreme(
            radial_p90_ratios, np.max)
        result["subgroup_local_density_ratio_worst"] = _finite_extreme(
            density_ratios, np.max)
        result["subgroup_local_density_wasserstein_max"] = _finite_extreme(
            density_wassersteins, np.max)
        result["subgroup_manifold_precision_min"] = _finite_extreme(
            group_precisions, np.min)
        result["subgroup_manifold_recall_min"] = _finite_extreme(
            group_recalls, np.min)
    return result


def _real_pca_projection(r, s, max_components=50, prepared=None):
    from sklearn.decomposition import PCA
    n_components = max(1, min(max_components, r.shape[1], len(r) - 1))
    pca = prepared.get("pca") if prepared is not None else None
    rp = prepared.get("projection") if prepared is not None else None
    if pca is None or rp is None:
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0).fit(r)
        rp = pca.transform(r)
        if prepared is not None:
            prepared["pca"], prepared["projection"] = pca, rp
    return rp, pca.transform(s)


def _subgroup_metrics(r, s, real_groups, synthetic_groups, top_k=100):
    rg = np.asarray(real_groups).astype(str)
    sg = np.asarray(synthetic_groups).astype(str)
    groups = sorted(set(rg) & set(sg))
    dispersions, real_centres, synth_centres = [], [], []
    effects, signs, overlaps = [], [], []
    for group in groups:
        r1, r0 = r[rg == group], r[rg != group]
        s1, s0 = s[sg == group], s[sg != group]
        if min(len(r1), len(r0), len(s1), len(s0)) < 2:
            continue
        rc, sc = r1.mean(0), s1.mean(0)
        real_centres.append(rc); synth_centres.append(sc)
        dr = np.mean(np.linalg.norm(r1 - rc, axis=1))
        ds = np.mean(np.linalg.norm(s1 - sc, axis=1))
        dispersions.append(ds / (dr + 1e-8))
        er, es = r1.mean(0) - r0.mean(0), s1.mean(0) - s0.mean(0)
        effects.append(_safe_corr(er, es))
        informative = np.abs(er) > 0.1
        signs.append(float(np.mean(np.sign(er[informative]) == np.sign(es[informative])))
                     if informative.any() else float("nan"))
        k = min(top_k, len(er))
        ri = set(np.argpartition(np.abs(er), -k)[-k:])
        si = set(np.argpartition(np.abs(es), -k)[-k:])
        overlaps.append(len(ri & si) / max(k, 1))
    centroid_corr = float("nan")
    if len(real_centres) >= 3:
        from scipy.spatial.distance import pdist
        centroid_corr = _safe_corr(pdist(np.asarray(real_centres)), pdist(np.asarray(synth_centres)))
    return {
        "subgroup_dispersion_ratio": float(np.nanmedian(dispersions)) if dispersions else float("nan"),
        "subgroup_centroid_distance_corr": centroid_corr,
        "subgroup_effect_corr": float(np.nanmedian(effects)) if effects else float("nan"),
        "subgroup_effect_sign_agreement": float(np.nanmean(signs)) if signs else float("nan"),
        "subgroup_top_feature_overlap": float(np.nanmean(overlaps)) if overlaps else float("nan"),
    }


def modality_geometry(real, synthetic, kind="continuous", real_groups=None,
                      synthetic_groups=None, max_features=512):
    """Return interpretable per-head fidelity metrics and a ranking penalty."""
    prepared = _prepare_real(real, max_features)
    raw_r, idx, r = prepared["raw"], prepared["idx"], prepared["standardized"]
    raw_s, s = _transform_prepared(synthetic, prepared)
    neighbour, rp, sp, pca = _source_neighbour_metrics(r, s, prepared=prepared)
    overlap = _conditional_source_overlap(
        rp, sp, real_groups=real_groups, synthetic_groups=synthetic_groups)
    # UMAP diagnostics use up to 5000 variable features. Repeat the source-only
    # metrics at that resolution without constructing a 5000x5000 covariance.
    broad_prepared = _prepare_real(real, 5000)
    broad_r = broad_prepared["standardized"]
    _, broad_s = _transform_prepared(synthetic, broad_prepared)
    if broad_r.shape[1] == r.shape[1]:
        broad_overlap = dict(overlap)
    else:
        broad_rp, broad_sp = _real_pca_projection(
            broad_r, broad_s, prepared=broad_prepared)
        broad_overlap = _conditional_source_overlap(
            broad_rp, broad_sp, real_groups=real_groups,
            synthetic_groups=synthetic_groups)
    source_overlap = dict(overlap)
    source_overlap.update({f"broad_{key}": value for key, value in broad_overlap.items()})
    boundary_components = []
    for values in (overlap, broad_overlap):
        boundary_components.extend([
            values["subgroup_radial_wasserstein_max"],
            np.log(max(values["subgroup_radial_median_ratio_worst"], 1.0)),
            np.log(max(values["subgroup_radial_p90_ratio_worst"], 1.0)),
            np.log(max(values["subgroup_local_density_ratio_worst"], 1.0)),
            values["subgroup_local_density_wasserstein_max"],
        ])
    source_overlap.update({
        "source_overlap_mixing_worst": _finite_extreme([
            overlap["subgroup_knn_mixing_normalized"],
            broad_overlap["subgroup_knn_mixing_normalized"],
        ], np.min, neighbour["knn_mixing_normalized"]),
        "source_overlap_auc_worst": _finite_extreme([
            overlap["subgroup_source_auc"], broad_overlap["subgroup_source_auc"],
        ], np.max, 0.5),
        "source_overlap_radial_worst": _finite_extreme([
            overlap["subgroup_radial_wasserstein"],
            broad_overlap["subgroup_radial_wasserstein"],
        ], np.max, 0.0),
        "source_overlap_precision_worst": _finite_extreme([
            overlap["manifold_precision"], broad_overlap["manifold_precision"],
        ], np.min, 0.0),
        "source_overlap_recall_worst": _finite_extreme([
            overlap["manifold_recall"], broad_overlap["manifold_recall"],
        ], np.min, 0.0),
        "source_overlap_mixing_min_worst": _finite_extreme([
            overlap["subgroup_knn_mixing_min"],
            broad_overlap["subgroup_knn_mixing_min"],
        ], np.min, neighbour["knn_mixing_normalized"]),
        "source_overlap_auc_max_worst": _finite_extreme([
            overlap["subgroup_source_auc_max"],
            broad_overlap["subgroup_source_auc_max"],
        ], np.max, 0.5),
        "source_overlap_radial_group_worst": _finite_extreme([
            overlap["subgroup_radial_wasserstein_max"],
            broad_overlap["subgroup_radial_wasserstein_max"],
        ], np.max, 0.0),
        "source_overlap_radial_median_ratio_worst": _finite_extreme([
            overlap["subgroup_radial_median_ratio_worst"],
            broad_overlap["subgroup_radial_median_ratio_worst"],
        ], np.max, 1.0),
        "source_overlap_radial_p90_ratio_worst": _finite_extreme([
            overlap["subgroup_radial_p90_ratio_worst"],
            broad_overlap["subgroup_radial_p90_ratio_worst"],
        ], np.max, 1.0),
        "source_overlap_local_density_ratio_worst": _finite_extreme([
            overlap["subgroup_local_density_ratio_worst"],
            broad_overlap["subgroup_local_density_ratio_worst"],
        ], np.max, 1.0),
        "source_overlap_local_density_wasserstein_worst": _finite_extreme([
            overlap["subgroup_local_density_wasserstein_max"],
            broad_overlap["subgroup_local_density_wasserstein_max"],
        ], np.max, 0.0),
        "source_overlap_group_precision_worst": _finite_extreme([
            overlap["subgroup_manifold_precision_min"],
            broad_overlap["subgroup_manifold_precision_min"],
        ], np.min, 1.0),
        "source_overlap_group_recall_worst": _finite_extreme([
            overlap["subgroup_manifold_recall_min"],
            broad_overlap["subgroup_manifold_recall_min"],
        ], np.min, 1.0),
        "source_overlap_boundary_penalty_worst": _finite_extreme(
            boundary_components, np.max, 0.0),
    })
    exact_duplicates = 0.0
    if len(raw_s):
        # Hashing rounded rows avoids constructing a large n_real*n_synth tensor.
        real_rows = {np.ascontiguousarray(row).tobytes() for row in np.round(raw_r[:, idx], 10)}
        exact_duplicates = float(np.mean([
            np.ascontiguousarray(row).tobytes() in real_rows for row in np.round(raw_s[:, idx], 10)
        ]))

    if kind == "binary":
        prevalence = float(np.mean(np.abs(raw_r[:, idx].mean(0) - raw_s[:, idx].mean(0))))
        cov_r, cov_s = np.cov(raw_r[:, idx], rowvar=False), np.cov(raw_s[:, idx], rowvar=False)
        association = float(np.linalg.norm(cov_r - cov_s, "fro") /
                            (np.linalg.norm(cov_r, "fro") + 1e-8))
        penalty = prevalence + association + abs(1.0 - neighbour["knn_mixing_normalized"])
        return {
            "prevalence_gap": prevalence,
            "association_gap": association,
            **neighbour,
            **source_overlap,
            "exact_duplicate_fraction": exact_duplicates,
            "geometry_penalty": float(penalty),
        }

    from sklearn.covariance import LedoitWolf
    cov_r = prepared.get("covariance")
    if cov_r is None:
        cov_r = LedoitWolf().fit(r).covariance_
        prepared["covariance"] = cov_r
    cov_s = LedoitWolf().fit(s).covariance_
    cov_gap = float(np.linalg.norm(cov_r - cov_s, "fro") /
                    (np.linalg.norm(cov_r, "fro") + 1e-8))
    tri = np.triu_indices_from(cov_r, 1)
    covariance_corr = _safe_corr(cov_r[tri], cov_s[tri])
    variances = np.var(s, axis=0)
    variance_mean = float(np.mean(variances))
    variance_median = float(np.median(variances))
    eig_real = pca.explained_variance_
    eig_synth = np.var(sp, axis=0, ddof=1)
    eig_gap = float(np.linalg.norm(eig_real - eig_synth) /
                    (np.linalg.norm(eig_real) + 1e-8))
    mean_corr = _safe_corr(raw_r[:, idx].mean(0), raw_s[:, idx].mean(0))
    coverage_ratio = _mean_nearest(s, r) / (_mean_nearest(r, r, self_match=True) + 1e-8)
    penalty = (
        abs(np.log(max(variance_median, 1e-8)))
        + cov_gap
        + eig_gap
        + abs(np.log(max(coverage_ratio, 1e-8)))
        + abs(1.0 - neighbour["knn_mixing_normalized"])
        + abs(1.0 - source_overlap["source_overlap_mixing_worst"])
        + 2.0 * abs(source_overlap["source_overlap_auc_worst"] - 0.5)
        + source_overlap["source_overlap_radial_worst"]
        + source_overlap["source_overlap_boundary_penalty_worst"]
        + (1.0 - source_overlap["source_overlap_precision_worst"])
        + (1.0 - source_overlap["source_overlap_recall_worst"])
        + (1.0 - source_overlap["source_overlap_group_precision_worst"])
        + (1.0 - source_overlap["source_overlap_group_recall_worst"])
    )
    result = {
        "mean_correlation": mean_corr,
        "variance_ratio": variance_mean,
        "variance_ratio_median": variance_median,
        "variance_ratio_p10": float(np.quantile(variances, 0.1)),
        "variance_ratio_p90": float(np.quantile(variances, 0.9)),
        "covariance_gap": cov_gap,
        "covariance_correlation": covariance_corr,
        "pca_eigenspectrum_gap": eig_gap,
        "coverage_ratio": float(coverage_ratio),
        **neighbour,
        **source_overlap,
        "exact_duplicate_fraction": exact_duplicates,
        "geometry_penalty": float(penalty),
    }
    if real_groups is not None and synthetic_groups is not None:
        subgroup = _subgroup_metrics(r, s, real_groups, synthetic_groups)
        result.update(subgroup)
        for key in ("subgroup_dispersion_ratio",):
            if np.isfinite(result[key]):
                result["geometry_penalty"] += abs(np.log(max(result[key], 1e-8)))
        for key in ("subgroup_centroid_distance_corr", "subgroup_effect_corr",
                    "subgroup_effect_sign_agreement", "subgroup_top_feature_overlap"):
            if np.isfinite(result[key]):
                result["geometry_penalty"] += 1.0 - result[key]
    return result


def cross_omic_fidelity(real_a, synth_a, real_b, synth_b, max_features=64):
    """Compare real-fitted cross-modal PCA associations and canonical spectra."""
    ra, sa, _ = _top_standardized(real_a, synth_a, max_features=max_features)
    rb, sb, _ = _top_standardized(real_b, synth_b, max_features=max_features)
    from sklearn.decomposition import PCA
    ka = max(1, min(10, ra.shape[1], len(ra) - 1))
    kb = max(1, min(10, rb.shape[1], len(rb) - 1))
    pa, pb = PCA(ka, random_state=0).fit(ra), PCA(kb, random_state=0).fit(rb)
    rpa, spa = pa.transform(ra), pa.transform(sa)
    rpb, spb = pb.transform(rb), pb.transform(sb)
    real_cross = np.cov(rpa.T, rpb.T)[:ka, ka:]
    synth_cross = np.cov(spa.T, spb.T)[:ka, ka:]
    covariance_gap = float(np.linalg.norm(real_cross - synth_cross, "fro") /
                           (np.linalg.norm(real_cross, "fro") + 1e-8))
    real_corr = np.corrcoef(rpa.T, rpb.T)[:ka, ka:]
    synth_corr = np.corrcoef(spa.T, spb.T)[:ka, ka:]
    correlation_gap = float(np.linalg.norm(real_corr - synth_corr, "fro") /
                            (np.linalg.norm(real_corr, "fro") + 1e-8))
    canonical_real = np.linalg.svd(np.nan_to_num(real_corr), compute_uv=False)
    canonical_synth = np.linalg.svd(np.nan_to_num(synth_corr), compute_uv=False)
    canonical_gap = float(np.linalg.norm(canonical_real - canonical_synth) /
                          (np.linalg.norm(canonical_real) + 1e-8))
    return {
        "covariance_gap": covariance_gap,
        "correlation_gap": correlation_gap,
        "canonical_spectrum_gap": canonical_gap,
        "geometry_penalty": covariance_gap + correlation_gap + canonical_gap,
    }


def cross_omic_geometry(real_a, synth_a, real_b, synth_b):
    """Backward-compatible scalar cross-modal covariance discrepancy."""
    return cross_omic_fidelity(real_a, synth_a, real_b, synth_b)["covariance_gap"]


def quality_guard_failures(metrics, prefixes=()):
    """Return hard failures that prevent white-noise or collapsed promotion."""
    failures = []
    prefixes = list(prefixes) or sorted({
        key.rsplit("_variance_ratio_median", 1)[0]
        for key in metrics if key.endswith("_variance_ratio_median")
    })
    for prefix in prefixes:
        median = metrics.get(f"{prefix}_variance_ratio_median", np.nan)
        p10 = metrics.get(f"{prefix}_variance_ratio_p10", np.nan)
        mixing = metrics.get(f"{prefix}_knn_mixing_normalized", np.nan)
        structural = metrics.get(f"{prefix}_covariance_correlation", np.nan)
        boundary = metrics.get(
            f"{prefix}_source_overlap_boundary_penalty_worst", np.nan)
        subgroup_auc = metrics.get(f"{prefix}_source_overlap_auc_max_worst", np.nan)
        subgroup_precision = metrics.get(
            f"{prefix}_source_overlap_group_precision_worst", np.nan)
        subgroup_recall = metrics.get(
            f"{prefix}_source_overlap_group_recall_worst", np.nan)
        if not np.isfinite(median) or not 0.5 <= median <= 1.5:
            failures.append(f"{prefix}:median_variance")
        if np.isfinite(p10) and p10 < 0.1:
            failures.append(f"{prefix}:feature_variance")
        if not np.isfinite(mixing) or mixing < 0.5:
            failures.append(f"{prefix}:neighbour_mixing")
        if not np.isfinite(structural) or structural < 0.1:
            failures.append(f"{prefix}:covariance_structure")
        # These are deliberately *severe* hard limits. Boundary/local-density
        # scores are already part of the rank and are compared with immutable
        # references at final promotion. Applying tight absolute limits to a
        # 150-epoch held-out fold rejects every calibration candidate before
        # the robust finalist stage can distinguish them.
        if np.isfinite(boundary) and boundary > 1.5:
            failures.append(f"{prefix}:subgroup_boundary")
        if np.isfinite(subgroup_auc) and subgroup_auc > 0.95:
            failures.append(f"{prefix}:subgroup_source_separation")
        if np.isfinite(subgroup_precision) and subgroup_precision < 0.25:
            failures.append(f"{prefix}:subgroup_manifold_precision")
    return failures
