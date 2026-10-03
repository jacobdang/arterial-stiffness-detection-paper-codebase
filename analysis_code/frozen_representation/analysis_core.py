"""Statistical core for the frozen-feature probe and incremental analysis.

The functions in this module are deliberately independent of the PWV file layout so
that their leakage properties and numerical behavior can be unit tested.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stable_seed(base_seed: int, label: str) -> int:
    """Derive a stable 32-bit seed without depending on Python's hash salt."""

    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted q values while preserving NaN positions."""

    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    values = p[finite]
    if values.size == 0:
        return q
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("P values must lie in [0, 1].")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[finite] = restored
    return q


def _training_fold_transform(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, int]]:
    """Median-impute, center, and scale using training data only."""

    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Predictor arrays must be two-dimensional.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train and test predictor widths differ.")

    medians = np.nanmedian(x_train, axis=0)
    if np.any(~np.isfinite(medians)):
        bad = np.flatnonzero(~np.isfinite(medians)).tolist()
        raise ValueError(f"All-missing/nonfinite training columns: {bad}")
    train_missing = int(np.isnan(x_train).sum())
    test_missing = int(np.isnan(x_test).sum())
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_test = np.where(np.isnan(x_test), medians, x_test)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Predictors contain nonfinite values after imputation.")

    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0, ddof=0)
    zero_scale = scales <= np.finfo(np.float64).eps
    scales[zero_scale] = 1.0
    return (
        (x_train - means) / scales,
        (x_test - means) / scales,
        {
            "train_imputed_cells": train_missing,
            "test_imputed_cells": test_missing,
            "zero_scale_columns": int(zero_scale.sum()),
        },
    )


def ridge_path_predict(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[np.ndarray, Mapping[str, int]]:
    """Predict a multi-target Ridge path with one eigensolve per training fold.

    For squared-error Ridge, columns of ``y_train`` are separable. Solving them in
    one matrix operation is mathematically identical to fitting each target in a
    separate Ridge model with the same alpha.
    """

    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0 or np.any(alphas <= 0):
        raise ValueError("Ridge alphas must be a nonempty positive vector.")
    x_train_z, x_test_z, transform_summary = _training_fold_transform(x_train, x_test)
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_train.shape[0] != x_train_z.shape[0]:
        raise ValueError("Training predictor and target row counts differ.")
    if not np.isfinite(y_train).all():
        raise ValueError("Targets must be complete and finite.")

    y_mean = y_train.mean(axis=0)
    y_centered = y_train - y_mean
    gram = x_train_z.T @ x_train_z
    cross = x_train_z.T @ y_centered
    # SciPy 1.10's dsyevd workspace query fails for a 1 x 1 matrix.  The
    # eigensystem is analytic in that scalar-score case; larger systems retain
    # the faster divide-and-conquer driver used by the 2,048-D analysis.
    if gram.shape == (1, 1):
        eigenvalues = np.asarray([gram[0, 0]], dtype=np.float64)
        eigenvectors = np.ones((1, 1), dtype=np.float64)
    else:
        eigenvalues, eigenvectors = eigh(
            gram, overwrite_a=True, check_finite=False, driver="evd"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ cross

    predictions = np.empty(
        (alphas.size, x_test_z.shape[0], y_train.shape[1]), dtype=np.float64
    )
    for alpha_index, alpha in enumerate(alphas):
        coefficients = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
        predictions[alpha_index] = x_test_z @ coefficients + y_mean
    return predictions, transform_summary


@dataclass
class NestedRidgeResult:
    oof_predictions: Dict[str, np.ndarray]
    fold_assignment: np.ndarray
    tuning_rows: List[dict]
    transform_rows: List[dict]


def nested_ridge_probe(
    inputs: Mapping[str, np.ndarray],
    targets: np.ndarray,
    target_names: Sequence[str],
    alphas: Sequence[float],
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> NestedRidgeResult:
    """Generate leakage-safe outer-fold predictions for all probe inputs/targets."""

    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != len(target_names):
        raise ValueError("Target matrix does not match target names.")
    if not np.isfinite(targets).all():
        raise ValueError("The common-cohort target matrix must be complete.")
    n = targets.shape[0]
    input_arrays = {name: np.asarray(value) for name, value in inputs.items()}
    if any(value.ndim != 2 or value.shape[0] != n for value in input_arrays.values()):
        raise ValueError("Every probe input must be a two-dimensional n-row matrix.")

    alphas = np.asarray(alphas, dtype=np.float64)
    outer_splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer_splitter.split(np.arange(n)))
    fold_assignment = np.full(n, -1, dtype=np.int16)
    oof = {
        name: np.full(targets.shape, np.nan, dtype=np.float64)
        for name in input_arrays
    }
    tuning_rows: List[dict] = []
    transform_rows: List[dict] = []

    for fold_index, (outer_train, outer_test) in enumerate(outer_splits):
        fold_assignment[outer_test] = fold_index
        inner_seed = stable_seed(seed, f"ridge-inner-{fold_index}")
        inner_splitter = KFold(
            n_splits=inner_folds, shuffle=True, random_state=inner_seed
        )
        inner_splits = list(inner_splitter.split(outer_train))

        for input_name, x in input_arrays.items():
            sum_squared_error = np.zeros(
                (alphas.size, targets.shape[1]), dtype=np.float64
            )
            validation_count = 0
            for inner_fold, (inner_train_rel, inner_valid_rel) in enumerate(inner_splits):
                inner_train = outer_train[inner_train_rel]
                inner_valid = outer_train[inner_valid_rel]
                path_predictions, transform_summary = ridge_path_predict(
                    x[inner_train],
                    x[inner_valid],
                    targets[inner_train],
                    alphas,
                )
                residuals = path_predictions - targets[inner_valid][None, :, :]
                sum_squared_error += np.square(residuals).sum(axis=1)
                validation_count += inner_valid.size
                transform_rows.append(
                    {
                        "stage": "inner",
                        "input_model": input_name,
                        "outer_fold": fold_index,
                        "inner_fold": inner_fold,
                        "train_n": int(inner_train.size),
                        "validation_n": int(inner_valid.size),
                        **transform_summary,
                    }
                )

            inner_mse = sum_squared_error / validation_count
            chosen_indices = np.argmin(inner_mse, axis=0)
            final_path, transform_summary = ridge_path_predict(
                x[outer_train], x[outer_test], targets[outer_train], alphas
            )
            transform_rows.append(
                {
                    "stage": "outer_refit",
                    "input_model": input_name,
                    "outer_fold": fold_index,
                    "inner_fold": None,
                    "train_n": int(outer_train.size),
                    "validation_n": int(outer_test.size),
                    **transform_summary,
                }
            )
            for target_index, target_name in enumerate(target_names):
                alpha_index = int(chosen_indices[target_index])
                oof[input_name][outer_test, target_index] = final_path[
                    alpha_index, :, target_index
                ]
                tuning_rows.append(
                    {
                        "input_model": input_name,
                        "target": target_name,
                        "outer_fold": fold_index,
                        "selected_alpha": float(alphas[alpha_index]),
                        "inner_cv_mse": float(inner_mse[alpha_index, target_index]),
                        "at_grid_minimum": bool(alpha_index == 0),
                        "at_grid_maximum": bool(alpha_index == alphas.size - 1),
                    }
                )

    if np.any(fold_assignment < 0) or any(not np.isfinite(v).all() for v in oof.values()):
        raise RuntimeError("Outer-fold prediction coverage is incomplete.")
    return NestedRidgeResult(oof, fold_assignment, tuning_rows, transform_rows)


def conditional_permutation_metrics(
    target: np.ndarray,
    predictions: np.ndarray,
    replicates: int,
    seed: int,
    batch_size: int = 250,
) -> Mapping[str, np.ndarray]:
    """Conditional permutation P values for fixed cross-fitted predictions."""

    target = np.asarray(target, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if predictions.shape[0] != target.size:
        raise ValueError("Prediction and target row counts differ.")

    target_rank = rankdata(target).astype(np.float64)
    prediction_ranks = np.column_stack(
        [rankdata(predictions[:, j]) for j in range(predictions.shape[1])]
    ).astype(np.float64)
    target_rank -= target_rank.mean()
    prediction_ranks -= prediction_ranks.mean(axis=0)
    rank_denominator = np.sqrt(
        np.square(target_rank).sum() * np.square(prediction_ranks).sum(axis=0)
    )
    observed_rho = (target_rank @ prediction_ranks) / rank_denominator

    target_sst = np.square(target - target.mean()).sum()
    target_square_sum = np.square(target).sum()
    prediction_square_sum = np.square(predictions).sum(axis=0)
    observed_sse = np.square(target[:, None] - predictions).sum(axis=0)
    observed_r2 = 1.0 - observed_sse / target_sst

    rho_extreme = np.zeros(predictions.shape[1], dtype=np.int64)
    r2_extreme = np.zeros(predictions.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < replicates:
        current = min(batch_size, replicates - completed)
        permutations = np.empty((current, target.size), dtype=np.int32)
        for row in range(current):
            permutations[row] = rng.permutation(target.size)
        permuted_rank = target_rank[permutations]
        permuted_rho = (permuted_rank @ prediction_ranks) / rank_denominator
        rho_extreme += (np.abs(permuted_rho) >= np.abs(observed_rho)).sum(axis=0)

        permuted_target = target[permutations]
        cross_products = permuted_target @ predictions
        permuted_sse = (
            target_square_sum + prediction_square_sum[None, :] - 2.0 * cross_products
        )
        permuted_r2 = 1.0 - permuted_sse / target_sst
        r2_extreme += (permuted_r2 >= observed_r2).sum(axis=0)
        completed += current

    return {
        "observed_spearman_rho": observed_rho,
        "observed_r2": observed_r2,
        "permutation_p_spearman": (rho_extreme + 1.0) / (replicates + 1.0),
        "permutation_p_r2": (r2_extreme + 1.0) / (replicates + 1.0),
    }


def paired_probe_bootstrap(
    target: np.ndarray,
    predictions: np.ndarray,
    replicates: int,
    seed: int,
) -> Mapping[str, np.ndarray]:
    """Exact paired participant bootstrap for probe rho and R-squared."""

    target = np.asarray(target, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if predictions.shape[0] != target.size:
        raise ValueError("Prediction and target row counts differ.")
    rho = np.empty((replicates, predictions.shape[1]), dtype=np.float64)
    r2 = np.empty_like(rho)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sample = rng.integers(0, target.size, size=target.size, dtype=np.int32)
        y = target[sample]
        pred = predictions[sample]
        for model_index in range(predictions.shape[1]):
            rho[replicate, model_index] = spearmanr(
                y, pred[:, model_index]
            ).statistic
            r2[replicate, model_index] = r2_score(y, pred[:, model_index])
    return {"spearman_rho": rho, "r2": r2}


def percentile_interval(values: np.ndarray, confidence_percent: float) -> np.ndarray:
    alpha = (100.0 - confidence_percent) / 200.0
    return np.quantile(values, [alpha, 1.0 - alpha], axis=0)


def two_sided_bootstrap_p(values: np.ndarray, null_value: float = 0.0) -> np.ndarray:
    """Two-sided sign/bootstrap-tail P value with a +1 finite-sample correction."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    lower = ((values <= null_value).sum(axis=0) + 1.0) / (values.shape[0] + 1.0)
    upper = ((values >= null_value).sum(axis=0) + 1.0) / (values.shape[0] + 1.0)
    return np.minimum(1.0, 2.0 * np.minimum(lower, upper))


@dataclass
class NestedLogisticResult:
    oof_probabilities: Dict[str, np.ndarray]
    fold_assignment: np.ndarray
    tuning_rows: List[dict]


def nested_logistic_models(
    inputs: Mapping[str, np.ndarray],
    outcome: np.ndarray,
    c_values: Sequence[float],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    n_jobs: int = 4,
) -> NestedLogisticResult:
    """Fit identically split, leakage-safe nested-CV L2 logistic models."""

    outcome = np.asarray(outcome, dtype=np.int64)
    if set(np.unique(outcome)) != {0, 1}:
        raise ValueError("Outcome must contain both binary classes 0 and 1.")
    n = outcome.size
    input_arrays = {name: np.asarray(value, dtype=np.float64) for name, value in inputs.items()}
    if any(value.ndim != 2 or value.shape[0] != n for value in input_arrays.values()):
        raise ValueError("Every logistic input must be a two-dimensional n-row matrix.")

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(np.zeros(n), outcome))
    fold_assignment = np.full(n, -1, dtype=np.int16)
    oof = {name: np.full(n, np.nan, dtype=np.float64) for name in input_arrays}
    tuning_rows: List[dict] = []

    for fold_index, (outer_train, outer_test) in enumerate(outer_splits):
        fold_assignment[outer_test] = fold_index
        inner_seed = stable_seed(seed, f"logistic-inner-{fold_index}")
        inner = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=inner_seed
        )
        inner_splits = list(
            inner.split(np.zeros(outer_train.size), outcome[outer_train])
        )
        for model_name, x in input_arrays.items():
            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l2",
                            solver="liblinear",
                            max_iter=5000,
                            random_state=stable_seed(seed, f"{model_name}-{fold_index}"),
                        ),
                    ),
                ]
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid={"model__C": list(c_values)},
                scoring="roc_auc",
                cv=inner_splits,
                refit=True,
                n_jobs=n_jobs,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(x[outer_train], outcome[outer_train])
            oof[model_name][outer_test] = search.predict_proba(x[outer_test])[:, 1]
            tuning_rows.append(
                {
                    "model": model_name,
                    "outer_fold": fold_index,
                    "train_n": int(outer_train.size),
                    "test_n": int(outer_test.size),
                    "train_events": int(outcome[outer_train].sum()),
                    "test_events": int(outcome[outer_test].sum()),
                    "selected_c": float(search.best_params_["model__C"]),
                    "inner_cv_auc": float(search.best_score_),
                }
            )

    if np.any(fold_assignment < 0) or any(not np.isfinite(v).all() for v in oof.values()):
        raise RuntimeError("Outer-fold logistic prediction coverage is incomplete.")
    return NestedLogisticResult(oof, fold_assignment, tuning_rows)


def classification_metrics(outcome: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    outcome = np.asarray(outcome, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return np.asarray(
        [
            roc_auc_score(outcome, probabilities),
            brier_score_loss(outcome, probabilities),
            log_loss(outcome, clipped, labels=[0, 1]),
        ],
        dtype=np.float64,
    )


def paired_classification_bootstrap(
    outcome: np.ndarray,
    probabilities: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Return B x model x metric paired participant-bootstrap estimates."""

    outcome = np.asarray(outcome, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    if probabilities.shape[0] != outcome.size:
        raise ValueError("Probability and outcome row counts differ.")
    result = np.empty((replicates, probabilities.shape[1], 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sample = rng.integers(0, outcome.size, size=outcome.size, dtype=np.int32)
        y = outcome[sample]
        if np.unique(y).size != 2:
            raise RuntimeError("A bootstrap sample unexpectedly contains one class.")
        for model_index in range(probabilities.shape[1]):
            result[replicate, model_index] = classification_metrics(
                y, probabilities[sample, model_index]
            )
    return result
