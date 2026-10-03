#!/usr/bin/env python3
"""Calculate global SHAP importance and participant distributions across 20 fusion models.

Inputs are authorized training-set tables, predictions, and fitted models.
Global importance is the mean absolute log-odds SHAP value, averaged across
imputations. Error bars show between-imputation standard deviations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn


FEATURE_COLUMNS = [f"param{i}" for i in range(1, 13)] + ["img_score"]
FEATURE_NAMES = [
    "Sex",
    "Age",
    "Diastolic BP",
    "Systolic BP",
    "Heart rate",
    "BMI",
    "Diabetes duration",
    "Hypertension",
    "Hyperlipidemia",
    "Cardiovascular disease",
    "Smoking status",
    "Drinking status",
    "Image-predicted AS score",
]
MODEL_BASENAME = (
    "feat_['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', "
    "'11', '12']_trained_all_submodel.pickle"
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_set_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(v) for v in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_image_scores(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    missing = {"id_set", "pred_set"} - set(obj)
    if missing:
        raise KeyError(f"Image prediction object is missing keys: {sorted(missing)}")
    ids = np.asarray(obj["id_set"])
    scores = np.asarray(obj["pred_set"], dtype=float)
    if ids.ndim != 1 or scores.ndim != 1 or len(ids) != len(scores):
        raise ValueError("id_set and pred_set must be equal-length one-dimensional arrays")
    frame = pd.DataFrame({"patient_id": ids, "img_score": scores})
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        raise ValueError("Image prediction patient IDs must be nonmissing and unique")
    if not np.isfinite(scores).all():
        raise ValueError("Image predictions contain non-finite values")
    return frame


def paths_for_imputation(repo_root: Path, mi_index: int) -> Tuple[Path, Path]:
    table = repo_root / "AUTHORIZED_DATA/shap/tables" / f"imputation_{mi_index}.csv"
    model = repo_root / "AUTHORIZED_DATA/shap/models" / f"imputation_{mi_index}.pickle"
    return table, model


def load_imputation(
    repo_root: Path, mi_index: int, image_scores: pd.DataFrame
) -> Tuple[pd.DataFrame, Any, Path, Path]:
    table_path, model_path = paths_for_imputation(repo_root, mi_index)
    if not table_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Missing MI {mi_index} input: table={table_path.is_file()}, "
            f"model={model_path.is_file()}"
        )
    frame = pd.read_csv(table_path)
    required = {"patient_id", *FEATURE_COLUMNS[:-1]}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"MI {mi_index} table is missing columns: {sorted(missing)}")
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        raise ValueError(f"MI {mi_index} patient IDs must be nonmissing and unique")
    merged = frame.loc[:, ["patient_id", *FEATURE_COLUMNS[:-1]]].merge(
        image_scores, on="patient_id", how="inner", validate="one_to_one"
    )
    if len(merged) != len(frame) or len(merged) != len(image_scores):
        raise ValueError(
            f"MI {mi_index} ID mismatch: table={len(frame)}, image={len(image_scores)}, "
            f"intersection={len(merged)}"
        )
    features = merged.loc[:, FEATURE_COLUMNS].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"MI {mi_index} feature matrix contains non-finite values")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "coef_") or np.asarray(model.coef_).shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(f"MI {mi_index} model is not the expected 13-feature linear model")
    return features, model, table_path, model_path


def save_bar_plot(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("pooled_mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.barh(
        ordered["feature_name"],
        ordered["pooled_mean_abs_shap"],
        xerr=ordered["between_imputation_sd"],
        color="#3A77B8",
        alpha=0.9,
        ecolor="#333333",
        capsize=2,
    )
    ax.set_xlabel("Mean |SHAP value| (log-odds), pooled over 20 imputations")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing AUTHORIZED_DATA/shap/ inputs",
    )
    parser.add_argument(
        "--image-predictions",
        type=Path,
        required=True,
        help="Authorized train prediction pickle containing unique id_set/pred_set arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: OUTPUTS/shap_20mi)",
    )
    parser.add_argument("--mi-count", type=int, default=20)
    parser.add_argument("--beeswarm", action="store_true", help="Also render participant SHAP distributions (Figure 2B) within the authorized environment")
    parser.add_argument("--beeswarm-imputation", type=int, default=9, help="Representative imputation for Figure 2B (1–20)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    image_path = args.image_predictions.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "OUTPUTS/shap_20mi"
    )
    if args.mi_count != 20:
        raise ValueError("The manuscript analysis requires exactly 20 imputations")
    if not 1 <= args.beeswarm_imputation <= args.mi_count:
        raise ValueError("Select a beeswarm imputation between 1 and 20")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_scores = load_image_scores(image_path)
    canonical_id_hash = id_set_sha256(image_scores["patient_id"])
    beeswarm_values = None
    beeswarm_features = None
    per_imputation: List[Dict[str, Any]] = []
    input_manifest: List[Dict[str, Any]] = [
        {
            "role": "train_image_predictions",
            "path": str(image_path),
            "sha256": sha256(image_path),
            "bytes": image_path.stat().st_size,
        }
    ]

    for mi_index in range(1, args.mi_count + 1):
        features, model, table_path, model_path = load_imputation(
            repo_root, mi_index, image_scores
        )
        if id_set_sha256(pd.read_csv(table_path, usecols=["patient_id"])["patient_id"]) != canonical_id_hash:
            raise ValueError(f"MI {mi_index} patient-ID set differs from image predictions")

        # SHAP 0.44.1 uses a deterministic 100-row Independent masker here.
        explainer = shap.LinearExplainer(model, features)
        shap_values = np.asarray(explainer.shap_values(features), dtype=float)
        if shap_values.shape != features.shape:
            raise ValueError(
                f"MI {mi_index} SHAP shape {shap_values.shape} != feature shape {features.shape}"
            )
        if args.beeswarm and mi_index == args.beeswarm_imputation:
            table_ids = pd.read_csv(table_path, usecols=["patient_id"])["patient_id"]
            positions = pd.Series(np.arange(len(table_ids)), index=table_ids).loc[image_scores["patient_id"]].to_numpy()
            beeswarm_values = shap_values[positions]
            beeswarm_features = features.to_numpy()[positions]
        importance = np.mean(np.abs(shap_values), axis=0)
        for feature_column, feature_name, value in zip(
            FEATURE_COLUMNS, FEATURE_NAMES, importance
        ):
            per_imputation.append(
                {
                    "imputation": mi_index,
                    "feature_column": feature_column,
                    "feature_name": feature_name,
                    "mean_abs_shap": float(value),
                }
            )
        input_manifest.extend(
            [
                {
                    "role": f"mi_{mi_index}_feature_table",
                    "path": str(table_path),
                    "sha256": sha256(table_path),
                    "bytes": table_path.stat().st_size,
                },
                {
                    "role": f"mi_{mi_index}_linear_model",
                    "path": str(model_path),
                    "sha256": sha256(model_path),
                    "bytes": model_path.stat().st_size,
                    "class": f"{type(model).__module__}.{type(model).__name__}",
                },
            ]
        )

    per_mi = pd.DataFrame(per_imputation)
    wide = per_mi.pivot(
        index=["feature_column", "feature_name"],
        columns="imputation",
        values="mean_abs_shap",
    )
    summary = wide.agg(["mean", "std", "min", "max"], axis=1).reset_index()
    summary.columns = [
        "feature_column",
        "feature_name",
        "pooled_mean_abs_shap",
        "between_imputation_sd",
        "min_mean_abs_shap",
        "max_mean_abs_shap",
    ]
    summary["rank_20mi"] = summary["pooled_mean_abs_shap"].rank(
        method="min", ascending=False
    ).astype(int)
    summary = summary.sort_values("rank_20mi").reset_index(drop=True)

    per_mi_path = output_dir / "shap_mean_abs_per_imputation.csv"
    summary_path = output_dir / "shap_feature_importance_20mi.csv"
    figure_path = output_dir / "shap_pooled_bar_20mi.png"
    manifest_path = output_dir / "input_manifest.json"
    run_info_path = output_dir / "shap_20mi_summary.json"
    per_mi.to_csv(per_mi_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_bar_plot(summary, figure_path)
    if args.beeswarm:
        np.random.seed(42)
        shap.summary_plot(beeswarm_values, beeswarm_features, feature_names=FEATURE_NAMES, show=False, max_display=len(FEATURE_NAMES))
        plt.xlabel("SHAP value (log-odds), representative imputation")
        plt.savefig(output_dir / "shap_representative_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
    manifest_path.write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    run_info = {
        "analysis": "fusion-model global SHAP importance pooled across 20 MICE datasets",
        "n_imputations": args.mi_count,
        "beeswarm_imputation": args.beeswarm_imputation if args.beeswarm else None,
        "n_participants_per_imputation": len(image_scores),
        "n_features": len(FEATURE_COLUMNS),
        "patient_id_set_sha256": canonical_id_hash,
        "shap_definition": (
            "mean absolute log-odds SHAP value from shap.LinearExplainer; "
            "default deterministic 100-row Independent masker per imputation"
        ),
        "pooling_definition": (
            "arithmetic mean of global mean absolute SHAP values over 20 imputations; "
            "error bars are between-imputation SD and are not confidence intervals"
        ),
        "top_features_20mi": summary.loc[
            :, ["rank_20mi", "feature_name", "pooled_mean_abs_shap"]
        ].head(5).to_dict(orient="records"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {},
    }
    for path in [per_mi_path, summary_path, figure_path, manifest_path]:
        run_info["outputs"][path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    run_info_path.write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
