import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis_core import (
    benjamini_hochberg,
    conditional_permutation_metrics,
    nested_ridge_probe,
    ridge_path_predict,
)


def test_ridge_path_matches_sklearn_pipeline():
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(80, 12))
    x_test = rng.normal(size=(17, 12))
    x_train[2, 3] = np.nan
    x_test[4, 3] = np.nan
    y_train = rng.normal(size=(80, 3))
    alphas = [0.1, 10.0, 1000.0]

    observed, summary = ridge_path_predict(x_train, x_test, y_train, alphas)
    assert summary["train_imputed_cells"] == 1
    assert summary["test_imputed_cells"] == 1
    for alpha_index, alpha in enumerate(alphas):
        expected = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")),
            ]
        ).fit(x_train, y_train).predict(x_test)
        np.testing.assert_allclose(observed[alpha_index], expected, rtol=1e-8, atol=1e-9)


def test_scalar_ridge_path_matches_sklearn():
    rng = np.random.default_rng(12)
    x_train = rng.normal(size=(60, 1))
    x_test = rng.normal(size=(13, 1))
    y_train = rng.normal(size=(60, 2))
    observed, _ = ridge_path_predict(x_train, x_test, y_train, [0.1, 100.0])
    for alpha_index, alpha in enumerate([0.1, 100.0]):
        expected = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")),
            ]
        ).fit(x_train, y_train).predict(x_test)
        np.testing.assert_allclose(observed[alpha_index], expected, rtol=1e-10, atol=1e-11)


def test_bh_known_values_and_nan_preservation():
    observed = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(observed[:3], [0.03, 0.04, 0.04])
    assert np.isnan(observed[3])


def test_nested_ridge_covers_every_row_once_and_all_inputs():
    rng = np.random.default_rng(23)
    n = 75
    x = rng.normal(size=(n, 8))
    y = np.column_stack([x[:, 0] + rng.normal(scale=0.1, size=n), rng.normal(size=n)])
    result = nested_ridge_probe(
        inputs={"a": x, "b": x[:, :2]},
        targets=y,
        target_names=["t0", "t1"],
        alphas=[0.1, 1.0, 10.0],
        outer_folds=5,
        inner_folds=5,
        seed=19,
    )
    assert sorted(np.unique(result.fold_assignment).tolist()) == [0, 1, 2, 3, 4]
    assert np.bincount(result.fold_assignment).sum() == n
    assert set(result.oof_predictions) == {"a", "b"}
    assert all(value.shape == y.shape for value in result.oof_predictions.values())
    assert all(np.isfinite(value).all() for value in result.oof_predictions.values())
    assert len(result.tuning_rows) == 5 * 2 * 2


def test_conditional_permutation_observed_metrics_are_consistent():
    rng = np.random.default_rng(41)
    y = rng.normal(size=90)
    predictions = np.column_stack([y + rng.normal(scale=0.2, size=90), rng.normal(size=90)])
    result = conditional_permutation_metrics(y, predictions, replicates=99, seed=7)
    assert result["observed_spearman_rho"].shape == (2,)
    assert result["observed_r2"].shape == (2,)
    assert np.all((result["permutation_p_spearman"] > 0) & (result["permutation_p_spearman"] <= 1))
    assert result["permutation_p_spearman"][0] <= result["permutation_p_spearman"][1]
