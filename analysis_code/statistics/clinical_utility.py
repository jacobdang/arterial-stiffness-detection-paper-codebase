"""Hosmer-Lemeshow grouping and decision-curve net-benefit calculations."""
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import chi2


def _hosmer_lemeshow(y: np.ndarray, probability: np.ndarray, groups: int) -> Dict[str, float]:
    categories = pd.cut(
        probability,
        np.percentile(probability, np.linspace(0, 100, groups + 1)),
        labels=False,
        include_lowest=True,
    )
    statistic = 0.0
    actual_groups = 0
    for group in range(groups):
        selected = categories == group
        if not np.any(selected):
            continue
        actual_groups += 1
        n = int(selected.sum())
        observed_event = float(y[selected].sum())
        expected_event = float(n * probability[selected].mean())
        observed_nonevent = n - observed_event
        expected_nonevent = n - expected_event
        statistic += (observed_event - expected_event) ** 2 / expected_event
        statistic += (observed_nonevent - expected_nonevent) ** 2 / expected_nonevent
    degrees_freedom = groups - 2
    return {
        "requested_groups": int(groups),
        "actual_nonempty_groups": int(actual_groups),
        "degrees_freedom": int(degrees_freedom),
        "chi_square": float(statistic),
        "p_value": float(chi2.sf(statistic, degrees_freedom)),
    }

def _standard_net_benefit(
    y: np.ndarray, probability: np.ndarray, threshold: float
) -> Dict[str, object]:
    """Calculate binary decision-curve net benefit at the specified threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("Decision threshold must lie strictly between zero and one")
    predicted = probability >= threshold
    event = y == 1
    tp = int(np.sum(predicted & event))
    fp = int(np.sum(predicted & ~event))
    n = int(len(y))
    odds = threshold / (1.0 - threshold)
    model = tp / n - fp / n * odds
    prevalence = float(np.mean(event))
    treat_all = prevalence - (1.0 - prevalence) * odds
    return {
        "threshold": float(threshold),
        "formula": "TP/N - FP/N * threshold/(1-threshold)",
        "n": n,
        "events": int(np.sum(event)),
        "tp": tp,
        "fp": fp,
        "fusion_model": float(model),
        "treat_all": float(treat_all),
        "treat_none": 0.0,
        "fusion_minus_treat_all": float(model - treat_all),
        "fusion_minus_treat_none": float(model),
    }
