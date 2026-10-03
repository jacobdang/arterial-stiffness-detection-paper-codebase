#!/usr/bin/env python3
"""Run frozen-feature probing and incremental analyses.

Participant-level artifacts are written only below ``private_outputs``. Aggregate
statistics, the protocol summary, and checksums are written below ``results``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from analysis_core import (
    benjamini_hochberg,
    classification_metrics,
    conditional_permutation_metrics,
    nested_logistic_models,
    nested_ridge_probe,
    paired_classification_bootstrap,
    paired_probe_bootstrap,
    percentile_interval,
    stable_seed,
    two_sided_bootstrap_p,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
CONFIG_PATH = EXPERIMENT_DIR / "config.example.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        compression = "gzip" if path.name.endswith(".gz") else None
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def resolve_and_verify_inputs(config: Mapping) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for name, record in config["inputs"].items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {name}: {path}")
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {name}: expected {record['sha256']}, got {observed}"
            )
        resolved[name] = path
    return resolved


@dataclass
class AnalysisData:
    patient_ids: np.ndarray
    vessel_targets: np.ndarray
    target_names: List[str]
    latent_features: np.ndarray
    image_score: np.ndarray
    probe_clinical: np.ndarray
    incremental_clinical: np.ndarray
    incremental_clinical_names: List[str]
    outcome: np.ndarray
    summary: dict


def _assert_unique_ids(frame: pd.DataFrame, label: str) -> None:
    if frame["patient_id"].isna().any():
        raise RuntimeError(f"{label} contains missing patient IDs.")
    duplicated = int(frame["patient_id"].duplicated().sum())
    if duplicated:
        raise RuntimeError(f"{label} contains {duplicated} duplicate patient IDs.")


def load_analysis_data(config: Mapping, paths: Mapping[str, Path]) -> AnalysisData:
    # This exact checksum-bound study pickle is explicitly trusted in config.
    pickle_record = config["inputs"]["reference_ensemble_pickle"]
    if pickle_record.get("trusted_reference_pickle") is not True:
        raise RuntimeError("Refusing to load a reference pickle that is not explicitly trusted.")
    with paths["reference_ensemble_pickle"].open("rb") as handle:
        reference = pickle.load(handle)
    required_pickle_keys = {"config", "gt_set", "pred_set", "id_set", "feat_set"}
    if set(reference) != required_pickle_keys:
        raise RuntimeError(f"Unexpected reference pickle keys: {sorted(reference)}")

    all_ids = np.asarray(reference["id_set"], dtype=np.int64)
    outcome_all = np.asarray(reference["gt_set"], dtype=np.int64)
    score_all = np.asarray(reference["pred_set"], dtype=np.float64)
    latent_all = np.asarray(reference["feat_set"], dtype=np.float64)
    if all_ids.shape != (7331,) or np.unique(all_ids).size != all_ids.size:
        raise RuntimeError("Reference participant IDs fail the expected 7,331-row uniqueness gate.")
    if outcome_all.shape != all_ids.shape or score_all.shape != all_ids.shape:
        raise RuntimeError("Reference outcomes/scores do not align with IDs.")
    if latent_all.shape != (all_ids.size, 2048):
        raise RuntimeError(f"Unexpected latent shape: {latent_all.shape}")
    if not np.isfinite(score_all).all() or not np.isfinite(latent_all).all():
        raise RuntimeError("Reference scores/features contain nonfinite values.")
    if set(np.unique(outcome_all)) != {0, 1}:
        raise RuntimeError("Reference outcome is not binary 0/1.")

    clinical = pd.read_csv(paths["figure5_clinical_table"])
    paper = pd.read_csv(paths["paper_values_crosscheck_table"])
    vessel = pd.read_csv(paths["vessel_participant_roi_table"])
    for frame in (clinical, paper, vessel):
        frame["patient_id"] = pd.to_numeric(frame["patient_id"], errors="raise").astype(
            np.int64
        )
    _assert_unique_ids(clinical, "Figure 5 clinical table")
    _assert_unique_ids(paper, "paper-value cross-check table")
    if vessel.duplicated(["patient_id", "roi"]).any():
        raise RuntimeError("Vessel table contains duplicate participant-ROI rows.")

    expected_rois = list(config["rois"])
    expected_features = list(config["vessel_features"])
    unexpected_rois = sorted(set(vessel["roi"].dropna()) - set(expected_rois))
    if unexpected_rois:
        raise RuntimeError(f"Unexpected vessel ROIs: {unexpected_rois}")
    missing_columns = set(expected_features) - set(vessel.columns)
    if missing_columns:
        raise RuntimeError(f"Missing vessel columns: {sorted(missing_columns)}")

    target_names = [
        f"{roi}__{feature}" for roi in expected_rois for feature in expected_features
    ]
    target_series = {}
    for roi in expected_rois:
        roi_frame = vessel.loc[vessel["roi"] == roi].set_index("patient_id")
        for feature in expected_features:
            target_series[f"{roi}__{feature}"] = pd.to_numeric(
                roi_frame[feature], errors="coerce"
            )
    target_wide = pd.DataFrame(target_series)
    target_wide = target_wide.loc[:, target_names]
    complete_target_wide = target_wide.dropna(axis=0, how="any")

    all_position = pd.Series(np.arange(all_ids.size), index=all_ids)
    cohort_mask = np.isin(all_ids, complete_target_wide.index.to_numpy(dtype=np.int64))
    patient_ids = all_ids[cohort_mask]
    if patient_ids.size != 5187:
        raise RuntimeError(
            f"Common 27-target cohort changed: expected 5,187, observed {patient_ids.size}."
        )
    positions = all_position.loc[patient_ids].to_numpy(dtype=np.int64)
    vessel_targets = complete_target_wide.loc[patient_ids, target_names].to_numpy(
        dtype=np.float64
    )
    latent_features = latent_all[positions]
    image_score = score_all[positions]
    outcome = outcome_all[positions]

    clinical_indexed = clinical.set_index("patient_id")
    paper_indexed = paper.set_index("patient_id")
    if not set(patient_ids).issubset(clinical_indexed.index):
        raise RuntimeError("Clinical table is missing common-cohort participants.")
    if not set(patient_ids).issubset(paper_indexed.index):
        raise RuntimeError("Paper-value table is missing common-cohort participants.")
    clinical_cohort = clinical_indexed.loc[patient_ids]
    paper_cohort = paper_indexed.loc[patient_ids]

    probe_names = list(config["probe_clinical_features"])
    incremental_names = list(config["incremental_clinical_features"])
    missing_clinical_columns = (
        set(probe_names) | set(incremental_names) | {"as_event"}
    ) - set(clinical_cohort.columns)
    if missing_clinical_columns:
        raise RuntimeError(f"Missing clinical columns: {sorted(missing_clinical_columns)}")
    probe_clinical = clinical_cohort[probe_names].to_numpy(dtype=np.float64)
    incremental_clinical = clinical_cohort[incremental_names].to_numpy(dtype=np.float64)

    # Four independent row-level bindings must agree before any model is fit.
    outcome_mismatch_figure5 = int(
        np.not_equal(outcome, clinical_cohort["as_event"].to_numpy(dtype=np.int64)).sum()
    )
    outcome_mismatch_paper = int(
        np.not_equal(
            outcome,
            paper_cohort["pwv_risk_class_ge_1400"].to_numpy(dtype=np.int64),
        ).sum()
    )
    score_crosscheck_max_abs = float(
        np.max(
            np.abs(
                image_score
                - paper_cohort["image_only_probability"].to_numpy(dtype=np.float64)
            )
        )
    )
    vessel_score_by_id = vessel.groupby("patient_id", sort=False)[
        "paper_image_only_score"
    ].first()
    vessel_score_crosscheck_max_abs = float(
        np.max(np.abs(image_score - vessel_score_by_id.loc[patient_ids].to_numpy()))
    )
    age_crosscheck_max_abs = float(
        np.nanmax(
            np.abs(
                clinical_cohort["age_years"].to_numpy(dtype=float)
                - paper_cohort["age_years"].to_numpy(dtype=float)
            )
        )
    )
    hba1c_crosscheck_max_abs = float(
        np.nanmax(
            np.abs(
                clinical_cohort["hba1c"].to_numpy(dtype=float)
                - paper_cohort["hba1c_percent_original"].to_numpy(dtype=float)
            )
        )
    )
    female_crosscheck_mismatch = int(
        np.not_equal(
            clinical_cohort["female"].to_numpy(dtype=np.int64),
            (paper_cohort["sex_code"].to_numpy(dtype=np.int64) == 2).astype(np.int64),
        ).sum()
    )
    if outcome_mismatch_figure5 or outcome_mismatch_paper:
        raise RuntimeError("Outcome cross-check failed.")
    if score_crosscheck_max_abs > 1e-12 or vessel_score_crosscheck_max_abs > 1e-6:
        raise RuntimeError("Image-score cross-check failed.")
    if age_crosscheck_max_abs > 1e-12 or hba1c_crosscheck_max_abs > 1e-12:
        raise RuntimeError("Age/HbA1c cross-check failed.")
    if female_crosscheck_mismatch:
        raise RuntimeError("Sex/female cross-check failed.")

    per_target_available = {
        target: int(target_wide[target].notna().sum()) for target in target_names
    }
    summary = {
        "reference_all_participants": int(all_ids.size),
        "reference_events": int(outcome_all.sum()),
        "reference_nonevents": int((1 - outcome_all).sum()),
        "latent_shape_all": list(latent_all.shape),
        "latent_dtype_in_pickle": str(np.asarray(reference["feat_set"]).dtype),
        "common_complete_27_target_participants": int(patient_ids.size),
        "common_events": int(outcome.sum()),
        "common_nonevents": int((1 - outcome).sum()),
        "vessel_participants_with_any_target": int(target_wide.notna().any(axis=1).sum()),
        "target_count": len(target_names),
        "per_target_available_n": per_target_available,
        "clinical_missing_cells_common_cohort": {
            name: int(clinical_cohort[name].isna().sum())
            for name in incremental_names
        },
        "outcome_mismatch_pickle_vs_figure5": outcome_mismatch_figure5,
        "outcome_mismatch_pickle_vs_paper_table": outcome_mismatch_paper,
        "score_max_abs_pickle_vs_paper_table": score_crosscheck_max_abs,
        "score_max_abs_pickle_vs_vessel_table": vessel_score_crosscheck_max_abs,
        "age_max_abs_figure5_vs_paper_table": age_crosscheck_max_abs,
        "hba1c_max_abs_figure5_vs_paper_table": hba1c_crosscheck_max_abs,
        "female_mismatch_figure5_vs_paper_table": female_crosscheck_mismatch,
        "image_score_range": [float(image_score.min()), float(image_score.max())],
        "target_names": target_names,
        "probe_clinical_features": probe_names,
        "incremental_clinical_features": incremental_names,
        "primary_cohort_rule": "complete observed values for all 27 vessel targets",
        "participant_id_order": "study ensemble pickle order",
    }
    return AnalysisData(
        patient_ids=patient_ids,
        vessel_targets=vessel_targets,
        target_names=target_names,
        latent_features=latent_features,
        image_score=image_score,
        probe_clinical=probe_clinical,
        incremental_clinical=incremental_clinical,
        incremental_clinical_names=incremental_names,
        outcome=outcome,
        summary=summary,
    )


def add_bh_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for metric in ("spearman", "r2"):
        p_column = f"permutation_p_{metric}"
        q_column = f"bh_q_27_{metric}"
        frame[q_column] = np.nan
        for model in frame["input_model"].unique():
            mask = frame["input_model"] == model
            frame.loc[mask, q_column] = benjamini_hochberg(
                frame.loc[mask, p_column].to_numpy()
            )
        frame[f"bh_q_global_81_{metric}"] = benjamini_hochberg(
            frame[p_column].to_numpy()
        )
    return frame


def run_probe(
    config: Mapping, data: AnalysisData, result_dir: Path, private_dir: Path
) -> Mapping[str, object]:
    input_order = ["scalar_score", "latent_2048", "clinical_age_sex_hba1c"]
    probe_inputs = {
        "scalar_score": data.image_score[:, None],
        "latent_2048": data.latent_features,
        "clinical_age_sex_hba1c": data.probe_clinical,
    }
    print("[probe] fitting nested 5x5 Ridge models", flush=True)
    fitted = nested_ridge_probe(
        inputs=probe_inputs,
        targets=data.vessel_targets,
        target_names=data.target_names,
        alphas=config["ridge_alphas"],
        outer_folds=int(config["outer_folds"]),
        inner_folds=int(config["inner_folds"]),
        seed=int(config["random_seed"]),
    )

    private_frame = pd.DataFrame(
        {
            "patient_id": data.patient_ids,
            "outer_fold": fitted.fold_assignment,
        }
    )
    for target_index, target in enumerate(data.target_names):
        private_frame[f"observed__{target}"] = data.vessel_targets[:, target_index]
        for model in input_order:
            private_frame[f"oof__{model}__{target}"] = fitted.oof_predictions[model][
                :, target_index
            ]
    atomic_csv(private_frame, private_dir / "probe_oof_predictions.csv.gz")
    atomic_csv(pd.DataFrame(fitted.tuning_rows), result_dir / "probe_tuning.csv")
    atomic_csv(
        pd.DataFrame(fitted.transform_rows), result_dir / "probe_transform_summary.csv"
    )

    rows: List[dict] = []
    delta_rows: List[dict] = []
    ci_level = float(config["bootstrap_ci_percent"])
    bootstrap_replicates = int(config["bootstrap_replicates"])
    permutation_replicates = int(config["permutation_replicates"])
    latent_index = input_order.index("latent_2048")
    comparisons = [
        ("latent_minus_scalar", latent_index, input_order.index("scalar_score")),
        (
            "latent_minus_clinical",
            latent_index,
            input_order.index("clinical_age_sex_hba1c"),
        ),
    ]

    for target_index, target in enumerate(data.target_names):
        print(
            f"[probe] inference {target_index + 1:02d}/{len(data.target_names)} {target}",
            flush=True,
        )
        y = data.vessel_targets[:, target_index]
        predictions = np.column_stack(
            [fitted.oof_predictions[name][:, target_index] for name in input_order]
        )
        permutation = conditional_permutation_metrics(
            target=y,
            predictions=predictions,
            replicates=permutation_replicates,
            seed=stable_seed(int(config["random_seed"]), f"perm-{target}"),
        )
        bootstrap = paired_probe_bootstrap(
            target=y,
            predictions=predictions,
            replicates=bootstrap_replicates,
            seed=stable_seed(int(config["random_seed"]), f"bootstrap-{target}"),
        )
        rho_ci = percentile_interval(bootstrap["spearman_rho"], ci_level)
        r2_ci = percentile_interval(bootstrap["r2"], ci_level)
        for model_index, model in enumerate(input_order):
            rho = float(spearmanr(y, predictions[:, model_index]).statistic)
            r2 = float(r2_score(y, predictions[:, model_index]))
            if not np.isclose(
                rho, permutation["observed_spearman_rho"][model_index], atol=1e-12
            ):
                raise RuntimeError("Spearman implementation cross-check failed.")
            if not np.isclose(r2, permutation["observed_r2"][model_index], atol=1e-12):
                raise RuntimeError("R-squared implementation cross-check failed.")
            rows.append(
                {
                    "target": target,
                    "roi": target.split("__", 1)[0],
                    "vessel_feature": target.split("__", 1)[1],
                    "input_model": model,
                    "n": int(y.size),
                    "spearman_rho": rho,
                    "spearman_ci_low": float(rho_ci[0, model_index]),
                    "spearman_ci_high": float(rho_ci[1, model_index]),
                    "cv_r2": r2,
                    "r2_ci_low": float(r2_ci[0, model_index]),
                    "r2_ci_high": float(r2_ci[1, model_index]),
                    "permutation_p_spearman": float(
                        permutation["permutation_p_spearman"][model_index]
                    ),
                    "permutation_p_r2": float(
                        permutation["permutation_p_r2"][model_index]
                    ),
                    "permutation_type": "conditional_fixed_cross_fitted_predictions",
                    "bootstrap_replicates": bootstrap_replicates,
                    "permutation_replicates": permutation_replicates,
                }
            )

        for contrast, positive_index, reference_index in comparisons:
            for metric_name, bootstrap_values in bootstrap.items():
                observed_values = (
                    permutation["observed_spearman_rho"]
                    if metric_name == "spearman_rho"
                    else permutation["observed_r2"]
                )
                delta_bootstrap = (
                    bootstrap_values[:, positive_index]
                    - bootstrap_values[:, reference_index]
                )
                delta_ci = percentile_interval(delta_bootstrap, ci_level)
                delta_rows.append(
                    {
                        "target": target,
                        "roi": target.split("__", 1)[0],
                        "vessel_feature": target.split("__", 1)[1],
                        "contrast": contrast,
                        "metric": metric_name,
                        "n": int(y.size),
                        "delta": float(
                            observed_values[positive_index]
                            - observed_values[reference_index]
                        ),
                        "ci_low": float(delta_ci[0]),
                        "ci_high": float(delta_ci[1]),
                        "bootstrap_p_two_sided": float(
                            two_sided_bootstrap_p(delta_bootstrap)[0]
                        ),
                        "bootstrap_replicates": bootstrap_replicates,
                    }
                )

    metrics = add_bh_columns(pd.DataFrame(rows))
    deltas = pd.DataFrame(delta_rows)
    deltas["bh_q_27"] = np.nan
    for (contrast, metric), indices in deltas.groupby(["contrast", "metric"]).groups.items():
        deltas.loc[indices, "bh_q_27"] = benjamini_hochberg(
            deltas.loc[indices, "bootstrap_p_two_sided"].to_numpy()
        )
    atomic_csv(metrics, result_dir / "probe_metrics.csv")
    atomic_csv(deltas, result_dir / "probe_paired_deltas.csv")

    tuning = pd.DataFrame(fitted.tuning_rows)
    boundary = (
        tuning.groupby("input_model")[["at_grid_minimum", "at_grid_maximum"]]
        .sum()
        .astype(int)
        .to_dict(orient="index")
    )
    summary = {
        "n": int(data.patient_ids.size),
        "targets": len(data.target_names),
        "input_models": input_order,
        "outer_folds": int(config["outer_folds"]),
        "inner_folds": int(config["inner_folds"]),
        "ridge_alpha_grid": list(config["ridge_alphas"]),
        "alpha_boundary_selection_counts_across_135_outer_target_fits": boundary,
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
        "permutation_type": "conditional labels versus fixed cross-fitted predictions; no pipeline refit",
        "bh_families": "27 per input and metric; global 81 sensitivity also reported",
    }
    atomic_json(result_dir / "probe_run_summary.json", summary)
    return summary


def run_incremental(
    config: Mapping, data: AnalysisData, result_dir: Path, private_dir: Path
) -> Mapping[str, object]:
    model_order = ["M0_clinical", "M1_clinical_vessels", "M2_clinical_image", "M3_all"]
    inputs = {
        "M0_clinical": data.incremental_clinical,
        "M1_clinical_vessels": np.column_stack(
            [data.incremental_clinical, data.vessel_targets]
        ),
        "M2_clinical_image": np.column_stack(
            [data.incremental_clinical, data.image_score]
        ),
        "M3_all": np.column_stack(
            [data.incremental_clinical, data.vessel_targets, data.image_score]
        ),
    }
    print("[incremental] fitting four nested 5x5 logistic models", flush=True)
    fitted = nested_logistic_models(
        inputs=inputs,
        outcome=data.outcome,
        c_values=config["logistic_c_values"],
        outer_folds=int(config["outer_folds"]),
        inner_folds=int(config["inner_folds"]),
        seed=stable_seed(int(config["random_seed"]), "incremental-outer"),
        n_jobs=min(4, os.cpu_count() or 1),
    )
    probabilities = np.column_stack(
        [fitted.oof_probabilities[name] for name in model_order]
    )
    private = pd.DataFrame(
        {
            "patient_id": data.patient_ids,
            "outer_fold": fitted.fold_assignment,
            "pwv_risk_class_ge_1400": data.outcome,
            **{
                f"oof_probability__{name}": fitted.oof_probabilities[name]
                for name in model_order
            },
        }
    )
    atomic_csv(private, private_dir / "incremental_oof_predictions.csv.gz")
    atomic_csv(
        pd.DataFrame(fitted.tuning_rows), result_dir / "incremental_tuning.csv"
    )

    print("[incremental] paired participant bootstrap", flush=True)
    bootstrap = paired_classification_bootstrap(
        outcome=data.outcome,
        probabilities=probabilities,
        replicates=int(config["bootstrap_replicates"]),
        seed=stable_seed(int(config["random_seed"]), "incremental-bootstrap"),
    )
    metric_names = ["auroc", "brier_score", "log_loss"]
    observed = np.vstack(
        [classification_metrics(data.outcome, probabilities[:, j]) for j in range(4)]
    )
    ci = percentile_interval(bootstrap, float(config["bootstrap_ci_percent"]))
    metric_rows: List[dict] = []
    for model_index, model in enumerate(model_order):
        for metric_index, metric in enumerate(metric_names):
            metric_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n": int(data.outcome.size),
                    "events": int(data.outcome.sum()),
                    "nonevents": int((1 - data.outcome).sum()),
                    "estimate": float(observed[model_index, metric_index]),
                    "ci_low": float(ci[0, model_index, metric_index]),
                    "ci_high": float(ci[1, model_index, metric_index]),
                    "bootstrap_replicates": int(config["bootstrap_replicates"]),
                }
            )

    contrasts = [
        ("M3_minus_M1", 3, 1),
        ("M3_minus_M2", 3, 2),
        ("M1_minus_M0", 1, 0),
        ("M2_minus_M0", 2, 0),
    ]
    contrast_rows: List[dict] = []
    for contrast, positive, reference in contrasts:
        delta_bootstrap = bootstrap[:, positive, :] - bootstrap[:, reference, :]
        delta_ci = percentile_interval(
            delta_bootstrap, float(config["bootstrap_ci_percent"])
        )
        delta_p = two_sided_bootstrap_p(delta_bootstrap)
        for metric_index, metric in enumerate(metric_names):
            contrast_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "n": int(data.outcome.size),
                    "delta_model_minus_reference": float(
                        observed[positive, metric_index]
                        - observed[reference, metric_index]
                    ),
                    "ci_low": float(delta_ci[0, metric_index]),
                    "ci_high": float(delta_ci[1, metric_index]),
                    "bootstrap_p_two_sided": float(delta_p[metric_index]),
                    "direction_note": (
                        "positive favors first model"
                        if metric == "auroc"
                        else "negative favors first model"
                    ),
                    "bootstrap_replicates": int(config["bootstrap_replicates"]),
                }
            )
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame["bh_q_4_contrasts"] = np.nan
    for metric, indices in contrast_frame.groupby("metric").groups.items():
        contrast_frame.loc[indices, "bh_q_4_contrasts"] = benjamini_hochberg(
            contrast_frame.loc[indices, "bootstrap_p_two_sided"].to_numpy()
        )
    atomic_csv(pd.DataFrame(metric_rows), result_dir / "incremental_metrics.csv")
    atomic_csv(contrast_frame, result_dir / "incremental_paired_deltas.csv")

    summary = {
        "n": int(data.outcome.size),
        "events": int(data.outcome.sum()),
        "nonevents": int((1 - data.outcome).sum()),
        "models": {
            "M0_clinical": data.incremental_clinical_names,
            "M1_clinical_vessels": "M0 + all 27 vessel features",
            "M2_clinical_image": "M0 + study top-five ensemble image probability",
            "M3_all": "M0 + all 27 vessel features + study image probability",
        },
        "outer_folds": int(config["outer_folds"]),
        "inner_folds": int(config["inner_folds"]),
        "logistic_c_grid": list(config["logistic_c_values"]),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "interpretation": "exploratory nested cross-validation within the original internal-test cohort",
    }
    atomic_json(result_dir / "incremental_run_summary.json", summary)
    return summary


def build_output_manifest(
    result_dir: Path, private_dir: Path, paths: Mapping[str, Path]
) -> dict:
    records = []
    roots = [EXPERIMENT_DIR, result_dir, private_dir]
    skip_names = {"output_manifest.json", "COMPLETE.json"}
    for directory in roots:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name in skip_names or "__pycache__" in path.parts:
                continue
            records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "classification": (
                        "private" if "private_inputs" in path.parts or "private_outputs" in path.parts else "aggregate_or_code"
                    ),
                }
            )
    input_records = [
        {
            "name": name,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    ]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": result_dir.name,
        "input_records": input_records,
        "output_and_code_records": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["preflight", "probe", "incremental", "all"], default="all"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    analysis_id = config["analysis_id"]
    result_dir = ROOT / "results" / analysis_id
    private_dir = ROOT / "private_outputs" / analysis_id
    result_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    completion_path = result_dir / "COMPLETE.json"
    if completion_path.exists() and args.mode != "preflight":
        raise RuntimeError(
            f"Completed result exists at {completion_path}; use a new analysis ID rather than overwrite it."
        )

    print("[preflight] verifying all frozen input hashes", flush=True)
    paths = resolve_and_verify_inputs(config)
    data = load_analysis_data(config, paths)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    input_summary = {
        "analysis_id": analysis_id,
        "protocol_version": config["protocol_version"],
        "all_input_hashes_verified": True,
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "data_summary": data.summary,
        "environment": environment,
        "feature_provenance": {
            "artifact": "study top_checkpoint_ensemble_test_result.pickle feat_set",
            "representation": "participant-level 2048-D arithmetic mean across five validation-selected checkpoint feature sets",
            "checkpoint_epochs_from_execution_log": [19, 23, 14, 21, 16],
            "not_single_checkpoint": True,
            "not_last_loop_residue": True,
        },
    }
    atomic_json(result_dir / "input_and_linkage_summary.json", input_summary)
    print(
        f"[preflight] PASS: n={data.patient_ids.size}, events={data.outcome.sum()}, targets={len(data.target_names)}",
        flush=True,
    )
    if args.mode == "preflight":
        return 0

    run_summaries = {}
    if args.mode in ("probe", "all"):
        run_summaries["probe"] = run_probe(config, data, result_dir, private_dir)
    if args.mode in ("incremental", "all"):
        run_summaries["incremental"] = run_incremental(
            config, data, result_dir, private_dir
        )

    manifest = build_output_manifest(result_dir, private_dir, paths)
    atomic_json(result_dir / "output_manifest.json", manifest)
    completion = {
        "analysis_id": analysis_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "status": "COMPLETE",
        "run_summaries": run_summaries,
        "output_manifest_sha256": sha256_file(result_dir / "output_manifest.json"),
        "privacy_gate": "participant-level inputs and OOF outputs remain private",
    }
    atomic_json(completion_path, completion)
    print(f"[complete] {completion_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
