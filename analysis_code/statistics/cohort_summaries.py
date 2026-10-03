"""Descriptive cohort summaries, missingness, and standardized mean differences."""
import math
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ranksums


def _category_summary(values: pd.Series, positive: float = 1.0) -> Dict[str, float]:
    observed = values.dropna()
    count = int(observed.eq(positive).sum())
    return {
        "observed_n": int(len(observed)),
        "missing_n": int(values.isna().sum()),
        "positive_n": count,
        "positive_fraction_of_observed": float(count / len(observed)),
    }

def _numeric_summary(values: pd.Series) -> Dict[str, float]:
    observed = values.dropna().to_numpy(dtype=float)
    return {
        "observed_n": int(len(observed)),
        "missing_n": int(values.isna().sum()),
        "mean": float(np.mean(observed)),
        "sd_population": float(np.std(observed, ddof=0)),
        "sd_sample": float(np.std(observed, ddof=1)),
        "median": float(np.median(observed)),
        "q1": float(np.quantile(observed, 0.25)),
        "q3": float(np.quantile(observed, 0.75)),
    }

def _comparison(a: pd.Series, b: pd.Series, categorical: bool) -> Dict[str, float]:
    if categorical:
        first = a.dropna()
        second = b.dropna()
        categories = sorted(set(first.unique()).union(second.unique()))
        table = np.asarray(
            [[int(first.eq(value).sum()), int(second.eq(value).sum())] for value in categories]
        )
        statistic, p_value, degrees_freedom, _ = chi2_contingency(table, correction=False)
        return {
            "statistic": float(statistic),
            "degrees_freedom": int(degrees_freedom),
            "p_value": float(p_value),
            "method": "Pearson chi-square without continuity correction",
        }
    statistic, p_value = ranksums(a.dropna(), b.dropna())
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "method": "Wilcoxon rank-sum as implemented by scipy.stats.ranksums",
    }

def _binary_smd(a: pd.Series, b: pd.Series) -> float:
    pa = float(a.dropna().eq(1).mean())
    pb = float(b.dropna().eq(1).mean())
    return abs(pa - pb) / math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2.0)

def _continuous_smd(a: pd.Series, b: pd.Series) -> float:
    first = a.dropna().to_numpy(dtype=float)
    second = b.dropna().to_numpy(dtype=float)
    return float(
        abs(first.mean() - second.mean())
        / math.sqrt((first.var(ddof=1) + second.var(ddof=1)) / 2.0)
    )
