"""Discrimination and independent-sample comparison helpers."""

import math
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import t
from sklearn.metrics import roc_auc_score


def as_binary(y: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(y), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("The outcome must be a finite one-dimensional array.")
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0.0, 1.0}) or len(unique) != 2:
        raise ValueError("The outcome must contain both binary classes 0 and 1.")
    return values.astype(int)


def auc_summary(y_true: Iterable[float], y_score: Iterable[float]) -> Dict[str, float]:
    y = as_binary(y_true)
    score = np.asarray(list(y_score), dtype=float)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("Scores must be finite and have the same shape as outcomes.")
    return {
        "n": int(y.size),
        "positive": int(y.sum()),
        "negative": int(y.size - y.sum()),
        "auc": float(roc_auc_score(y, score)),
    }


def auc_bootstrap_summary(
    y_true: Iterable[float], y_score: Iterable[float], n_bootstraps: int = 1000, seed: int = 42
) -> Dict[str, float]:
    """Reproduce the study percentile-index AUC interval.

    The integer-index convention is retained to match the manuscript notebooks;
    it is recorded explicitly because other quantile conventions can differ in
    the final decimal place.
    """
    result = auc_summary(y_true, y_score)
    y = as_binary(y_true)
    score = np.asarray(list(y_score), dtype=float)
    rng = np.random.RandomState(seed)
    bootstrap = []
    for _ in range(n_bootstraps):
        index = rng.randint(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            bootstrap.append(roc_auc_score(y[index], score[index]))
    ordered = np.sort(np.asarray(bootstrap, dtype=float))
    result.update(
        {
            "auc_ci_lower": float(ordered[int(0.025 * len(ordered))]),
            "auc_ci_upper": float(ordered[int(0.975 * len(ordered))]),
            "bootstrap_resamples": int(len(ordered)),
            "bootstrap_seed": int(seed),
            "ci_convention": "sorted integer indices floor(0.025*n), floor(0.975*n)",
        }
    )
    return result


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and ordered[right] == ordered[left]:
            right += 1
        ranks[left:right] = 0.5 * (left + right - 1) + 1.0
        left = right
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def delong_auc_variance(
    y_true: Iterable[float], y_score: Iterable[float]
) -> Tuple[float, float]:
    """Return one-sample DeLong AUC and variance.

    This is kept dependency-light so the independent subgroup comparison can
    be calculated without R. It follows the same midrank calculation as the
    study ``roc_pvalue_python.py`` file.
    """
    y = as_binary(y_true)
    score = np.asarray(list(y_score), dtype=float)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("Scores must be finite and have the same shape as outcomes.")
    order = np.argsort(-y)
    ordered = score[order]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    positive_scores = ordered[:positives]
    negative_scores = ordered[positives:]
    tx = _midrank(positive_scores)
    ty = _midrank(negative_scores)
    tz = _midrank(ordered)
    auc = float(tz[:positives].sum() / positives / negatives)
    auc -= float(positives + 1.0) / 2.0 / negatives
    v01 = (tz[:positives] - tx) / negatives
    v10 = 1.0 - (tz[positives:] - ty) / positives
    variance = float(np.var(v01, ddof=1) / positives + np.var(v10, ddof=1) / negatives)
    return auc, variance


def independent_delong_test(
    y_true_a: Iterable[float],
    y_score_a: Iterable[float],
    y_true_b: Iterable[float],
    y_score_b: Iterable[float],
) -> Dict[str, float]:
    """Reproduce pROC's two-sided unpaired DeLong comparison.

    pROC uses a Welch/Satterthwaite t approximation for two independent ROC
    curves. This is the comparison used for the manuscript subgroup panels.
    """
    y_a = as_binary(y_true_a)
    y_b = as_binary(y_true_b)
    auc_a, variance_a = delong_auc_variance(y_a, y_score_a)
    auc_b, variance_b = delong_auc_variance(y_b, y_score_b)
    total_variance = variance_a + variance_b
    statistic = (auc_a - auc_b) / math.sqrt(total_variance)
    degrees_freedom = total_variance ** 2 / (
        variance_a ** 2 / (len(y_a) - 1) + variance_b ** 2 / (len(y_b) - 1)
    )
    p_value = 2.0 * t.sf(abs(statistic), degrees_freedom)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "variance_a": variance_a,
        "variance_b": variance_b,
        "t": float(statistic),
        "degrees_freedom": float(degrees_freedom),
        "p_value": float(p_value),
        "method": "unpaired DeLong with Welch/Satterthwaite t approximation",
    }
