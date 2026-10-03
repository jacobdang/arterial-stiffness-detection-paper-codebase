"""Participant-level retinal-vessel associations with bootstrap intervals and BH correction."""
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURE_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Diameter / Calibre",
        ("artery_width_median_px", "vein_width_median_px", "AVR"),
    ),
    ("Central Light Reflex", ("artery_clr", "vein_clr")),
    (
        "Morphology",
        (
            "artery_tortuosity_median_branch",
            "vein_tortuosity_median_branch",
            "artery_branch_count",
            "vein_branch_count",
        ),
    ),
)

FEATURES = tuple(feature for _, features in FEATURE_GROUPS for feature in features)

EXPECTED_ROIS = ("disc", "full", "macula_dd2")

PRIMARY_BOOTSTRAP_RESAMPLES = 2000

PRIMARY_BOOTSTRAP_SEED = 20260826

def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> Tuple[float, float, int, int]:
    if n_resamples < 100:
        raise ValueError("At least 100 participant bootstrap resamples are required")
    if len(x) != len(y) or len(x) < 30:
        raise ValueError("Bootstrap inputs must have the same participant count >= 30")
    rng = np.random.RandomState(seed)
    accepted: List[float] = []
    rejected = 0
    for _ in range(n_resamples):
        index = rng.randint(0, len(x), len(x))
        coefficient = spearmanr(x[index], y[index]).correlation
        if np.isfinite(coefficient):
            accepted.append(float(coefficient))
        else:
            rejected += 1
    if len(accepted) < max(100, int(0.95 * n_resamples)):
        raise ValueError("Too many invalid participant bootstrap resamples")
    lower, upper = np.quantile(np.asarray(accepted), [0.025, 0.975])
    return float(lower), float(upper), len(accepted), rejected

def _primary_participant_correlations(
    frame: pd.DataFrame,
    *,
    n_resamples: int = PRIMARY_BOOTSTRAP_RESAMPLES,
    seed: int = PRIMARY_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Primary 27-test participant analysis with bootstrap CIs and global BH."""
    rows: List[Dict[str, object]] = []
    test_index = 0
    for roi in EXPECTED_ROIS:
        roi_frame = frame.loc[frame["roi"] == roi]
        for group, features in FEATURE_GROUPS:
            for feature in features:
                complete = roi_frame[[feature, "pwv"]].dropna()
                if len(complete) < 30:
                    raise ValueError(
                        "Primary participant analysis has fewer than 30 complete rows: "
                        f"{roi}/{feature}"
                    )
                coefficient, p_value = spearmanr(complete[feature], complete["pwv"])
                row_seed = int(seed + test_index)
                lower, upper, accepted, rejected = _bootstrap_spearman_ci(
                    complete[feature].to_numpy(dtype=float),
                    complete["pwv"].to_numpy(dtype=float),
                    n_resamples=n_resamples,
                    seed=row_seed,
                )
                rows.append(
                    {
                        "analysis_unit": "final_test_table_participant_mean_primary",
                        "roi": roi,
                        "group": group,
                        "feature": feature,
                        "target": "param35_continuous_pwv",
                        "spearman_r": float(coefficient),
                        "ci_lower_participant_bootstrap_95": lower,
                        "ci_upper_participant_bootstrap_95": upper,
                        "p_value_raw": float(p_value),
                        "n": int(len(complete)),
                        "bootstrap_resamples_requested": int(n_resamples),
                        "bootstrap_resamples_accepted": int(accepted),
                        "bootstrap_resamples_rejected": int(rejected),
                        "bootstrap_seed_for_test": row_seed,
                    }
                )
                test_index += 1
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_ROIS) * len(FEATURES):
        raise AssertionError("Primary vessel family must contain exactly 27 tests")
    result["p_fdr_bh_all_27"] = _benjamini_hochberg(result["p_value_raw"])
    result["significant_fdr_0_05"] = result["p_fdr_bh_all_27"] < 0.05
    return result

def _benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("FDR input must be a finite one-dimensional array")
    count = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted
