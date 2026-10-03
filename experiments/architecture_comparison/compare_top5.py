#!/usr/bin/env python3
"""Validate and compare a DeiT top-five ensemble with study TinyNet-C.

The input pickle files are trusted local experiment artifacts with the reference
``id_set``, ``gt_set`` and ``pred_set`` schema.  All comparisons are paired at
participant level after exact ID alignment. Bootstrap confidence intervals use
the 2.5th and 97.5th percentiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import sklearn
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


SCHEMA_VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="New DeiT top_checkpoint_ensemble_test_result.pickle",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Study TinyNet-C top-five test prediction pickle",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", default="DeiT-Tiny")
    parser.add_argument("--reference-name", default="TinyNet-C")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--require-candidate-run-manifest",
        action="store_true",
        help="Require and validate run_manifest.json beside the candidate",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return sha256_bytes(encoded)


def array_sha256(value: np.ndarray, dtype: Any) -> str:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return sha256_bytes(little_endian.tobytes(order="C"))


def canonical_id(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        raise ValueError("Missing participant ID")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite participant ID")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("Participant IDs must be scalar and hashable") from error
    return value


def id_sequence_sha256(ids: Sequence[Any]) -> str:
    # Type tags prevent a string ID such as "1" from hashing like integer 1.
    serializable = [
        {"type": type(item).__name__, "value": str(item)}
        for item in ids
    ]
    return canonical_json_sha256(serializable)


def binary_labels(value: Any, path: Path) -> np.ndarray:
    raw = np.asarray(value).reshape(-1)
    if raw.size == 0:
        raise ValueError("Empty label vector in {}".format(path))
    try:
        labels = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Non-numeric labels in {}".format(path)) from error
    if not np.isfinite(labels).all():
        raise ValueError("Non-finite labels in {}".format(path))
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Labels are not exactly binary (0/1) in {}".format(path))
    labels = labels.astype(np.int8)
    if np.unique(labels).size != 2:
        raise ValueError("Both outcome classes are required in {}".format(path))
    return labels


def load_prediction(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Pickle is intentionally limited to trusted local experiment artifacts.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"id_set", "gt_set", "pred_set"}
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError("Unexpected prediction schema in {}".format(path))

    raw_ids = np.asarray(payload["id_set"], dtype=object).reshape(-1)
    ids = [canonical_id(item) for item in raw_ids]
    if len(ids) != len(set(ids)):
        duplicate_count = len(ids) - len(set(ids))
        raise ValueError(
            "{} duplicate participant-ID rows in {}".format(duplicate_count, path)
        )
    labels = binary_labels(payload["gt_set"], path)
    try:
        scores = np.asarray(payload["pred_set"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("Non-numeric prediction scores in {}".format(path)) from error
    if not (len(ids) == len(labels) == len(scores)):
        raise ValueError("ID/label/score length mismatch in {}".format(path))
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores in {}".format(path))
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("Scores are not probabilities in [0, 1] in {}".format(path))

    config = payload.get("config", {})
    return {
        "path": path,
        "ids": ids,
        "labels": labels,
        "scores": scores,
        "config": config,
        "file_sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
    }


def align_candidate(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    reference_ids = list(reference["ids"])
    candidate_ids = list(candidate["ids"])
    reference_set = set(reference_ids)
    candidate_set = set(candidate_ids)
    if reference_set != candidate_set:
        missing = reference_set - candidate_set
        unexpected = candidate_set - reference_set
        raise ValueError(
            "Participant-ID sets differ: {} missing and {} unexpected in candidate".format(
                len(missing), len(unexpected)
            )
        )

    candidate_index = {participant_id: index for index, participant_id in enumerate(candidate_ids)}
    order = np.asarray([candidate_index[item] for item in reference_ids], dtype=np.int64)
    aligned_labels = np.asarray(candidate["labels"], dtype=np.int8)[order]
    reference_labels = np.asarray(reference["labels"], dtype=np.int8)
    mismatch_count = int(np.count_nonzero(aligned_labels != reference_labels))
    if mismatch_count:
        raise ValueError(
            "Outcome labels differ for {} participants after ID alignment".format(mismatch_count)
        )
    aligned_scores = np.asarray(candidate["scores"], dtype=np.float64)[order]
    validation = {
        "reference_n": len(reference_ids),
        "candidate_n": len(candidate_ids),
        "reference_unique_ids": len(reference_set),
        "candidate_unique_ids": len(candidate_set),
        "id_sets_exactly_equal": True,
        "input_order_identical": bool(reference_ids == candidate_ids),
        "candidate_rows_moved_during_alignment": int(np.count_nonzero(order != np.arange(len(order)))),
        "labels_exactly_equal_after_id_alignment": True,
        "label_mismatch_count_after_id_alignment": 0,
        "reference_id_order_sha256": id_sequence_sha256(reference_ids),
        "candidate_input_id_order_sha256": id_sequence_sha256(candidate_ids),
        "candidate_aligned_id_order_sha256": id_sequence_sha256(reference_ids),
        "aligned_label_vector_sha256": array_sha256(reference_labels, np.int8),
        "reference_score_vector_sha256": array_sha256(reference["scores"], np.float64),
        "candidate_aligned_score_vector_sha256": array_sha256(aligned_scores, np.float64),
    }
    return reference_labels, aligned_scores, validation


def percentile_interval(values: np.ndarray, lower: float, upper: float) -> List[float]:
    return [
        float(np.percentile(values, lower)),
        float(np.percentile(values, upper)),
    ]


def paired_bootstrap(
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    n_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    if n_resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    rng = np.random.RandomState(seed)
    candidate_auc: List[float] = []
    reference_auc: List[float] = []
    rejected = 0
    for _ in range(n_resamples):
        index = rng.randint(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if np.unique(sampled_labels).size < 2:
            rejected += 1
            continue
        candidate_auc.append(roc_auc_score(sampled_labels, candidate_scores[index]))
        reference_auc.append(roc_auc_score(sampled_labels, reference_scores[index]))
    candidate_values = np.asarray(candidate_auc, dtype=np.float64)
    reference_values = np.asarray(reference_auc, dtype=np.float64)
    if candidate_values.size == 0:
        raise RuntimeError("No valid bootstrap samples")
    delta_values = candidate_values - reference_values
    return {
        "candidate_auc_values": candidate_values,
        "reference_auc_values": reference_values,
        "delta_auc_values": delta_values,
        "requested": int(n_resamples),
        "accepted": int(candidate_values.size),
        "single_class_rejected": int(rejected),
        "seed": int(seed),
    }


def midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and ordered[right] == ordered[left]:
            right += 1
        ranks[left:right] = 0.5 * (left + right - 1) + 1.0
        left = right
    result = np.empty(len(values), dtype=np.float64)
    result[order] = ranks
    return result


def paired_delong(
    labels: np.ndarray, reference_scores: np.ndarray, candidate_scores: np.ndarray
) -> Dict[str, Any]:
    """Paired DeLong test; reported estimand is candidate minus reference."""

    order = np.argsort(-labels)
    predictions = np.vstack((reference_scores, candidate_scores))[:, order]
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    positive_scores = predictions[:, :positives]
    negative_scores = predictions[:, positives:]
    tx = np.empty((2, positives), dtype=np.float64)
    ty = np.empty((2, negatives), dtype=np.float64)
    tz = np.empty_like(predictions, dtype=np.float64)
    for row in range(2):
        tx[row] = midrank(positive_scores[row])
        ty[row] = midrank(negative_scores[row])
        tz[row] = midrank(predictions[row])
    aucs = (tz[:, :positives].sum(axis=1) - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    v01 = (tz[:, :positives] - tx) / negatives
    v10 = 1.0 - (tz[:, positives:] - ty) / positives
    covariance = np.cov(v01) / positives + np.cov(v10) / negatives
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if not np.isfinite(variance) or variance < -1e-15:
        raise ValueError("Invalid paired DeLong contrast variance: {}".format(variance))
    variance = max(variance, 0.0)
    standard_error = float(np.sqrt(variance))
    delta = float(aucs[1] - aucs[0])
    if standard_error == 0.0:
        z_value = 0.0 if delta == 0.0 else math.copysign(math.inf, delta)
        p_value = 1.0 if delta == 0.0 else 0.0
    else:
        z_value = delta / standard_error
        p_value = float(2.0 * norm.sf(abs(z_value)))
    critical = float(norm.ppf(0.975))
    return {
        "auc_reference": float(aucs[0]),
        "auc_candidate": float(aucs[1]),
        "delta_auc_candidate_minus_reference": delta,
        "variance": variance,
        "standard_error": standard_error,
        "z_candidate_minus_reference": float(z_value),
        "two_sided_p_value": p_value,
        "analytic_normal_95_ci": [
            float(delta - critical * standard_error),
            float(delta + critical * standard_error),
        ],
        "critical_value": critical,
    }


def file_fact(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def adjacent_provenance(prediction_path: Path) -> Dict[str, Any]:
    run_dir = prediction_path.resolve().parent
    relative_names = (
        "prelaunch_manifest.json",
        "run_manifest.json",
        "top5_evaluation_manifest.json",
        "train_main.log",
        "test_main.log",
        ".hydra/config.yaml",
    )
    return {
        name: file_fact(run_dir / name)
        for name in relative_names
        if (run_dir / name).is_file()
    }


def validate_candidate_manifest(
    candidate: Mapping[str, Any], candidate_auc: float, required: bool
) -> Dict[str, Any]:
    path = Path(candidate["path"]).parent / "run_manifest.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError("Required candidate run manifest is missing: {}".format(path))
        return {"present": False, "required": required, "validation": "not_performed"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if manifest.get("completed") is not True:
        failures.append("completed is not true")
    if manifest.get("model") != "deit_tiny":
        failures.append("model is not deit_tiny")
    if manifest.get("timm_version") != "0.9.2":
        failures.append("timm_version is not 0.9.2")
    top_five = manifest.get("top_five")
    if not isinstance(top_five, list) or len(top_five) != 5:
        failures.append("top_five does not contain exactly five entries")
    else:
        listed_hashes = []
        for item in top_five:
            checkpoint_path = Path(item.get("path", ""))
            listed_hash = item.get("sha256")
            if not checkpoint_path.is_file():
                failures.append("selected checkpoint is missing: {}".format(checkpoint_path))
                continue
            actual_hash = sha256_file(checkpoint_path)
            if actual_hash != listed_hash:
                failures.append("checkpoint hash mismatch: {}".format(checkpoint_path))
            listed_hashes.append(listed_hash)
        if len(listed_hashes) != len(set(listed_hashes)):
            failures.append("top_five contains duplicate checkpoint hashes")
    manifest_auc = manifest.get("ensemble_auc", {}).get("test")
    if manifest_auc is None or not np.isclose(
        float(manifest_auc), candidate_auc, rtol=0.0, atol=1e-12
    ):
        failures.append("manifest test AUROC does not match prediction file")
    prelaunch_path = path.parent / "prelaunch_manifest.json"
    if prelaunch_path.is_file():
        prelaunch = json.loads(prelaunch_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(prelaunch.get("config")) != canonical_json_sha256(
            candidate.get("config")
        ):
            failures.append("prediction config differs from prelaunch config")
        if prelaunch.get("timm_version") != "0.9.2":
            failures.append("prelaunch timm_version is not 0.9.2")
    if failures:
        raise ValueError("Candidate run-manifest validation failed: " + "; ".join(failures))
    return {
        "present": True,
        "required": required,
        "validation": "passed",
        "file": file_fact(path),
        "model": manifest["model"],
        "precision": manifest.get("precision"),
        "seed": manifest.get("seed"),
        "epochs": manifest.get("epochs"),
        "timm_version": manifest["timm_version"],
        "torch_version": manifest.get("torch_version"),
        "top_five_checkpoint_hashes_verified": True,
        "manifest_test_auc_matches": True,
    }


def model_report(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    bootstrap_values: np.ndarray,
) -> Dict[str, Any]:
    return {
        "name": name,
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "nonevents": int(len(labels) - labels.sum()),
        "auc": float(roc_auc_score(labels, scores)),
        "percentile_95_ci": percentile_interval(bootstrap_values, 2.5, 97.5),
    }


def write_csv(report: Mapping[str, Any], path: Path) -> None:
    candidate = report["results"]["candidate"]
    reference = report["results"]["reference"]
    comparison = report["results"]["paired_comparison"]
    bootstrap = comparison["participant_bootstrap"]
    delong = comparison["paired_delong"]
    provenance = report["provenance"]
    row = {
        "reference": reference["name"],
        "candidate": candidate["name"],
        "n": reference["n"],
        "events": reference["events"],
        "reference_auc": reference["auc"],
        "reference_auc_ci_lower": reference["percentile_95_ci"][0],
        "reference_auc_ci_upper": reference["percentile_95_ci"][1],
        "candidate_auc": candidate["auc"],
        "candidate_auc_ci_lower": candidate["percentile_95_ci"][0],
        "candidate_auc_ci_upper": candidate["percentile_95_ci"][1],
        "delta_estimand": "candidate_minus_reference",
        "delta_auc": comparison["delta_auc_candidate_minus_reference"],
        "paired_bootstrap_delta_ci_lower": bootstrap["percentile_95_ci"][0],
        "paired_bootstrap_delta_ci_upper": bootstrap["percentile_95_ci"][1],
        "paired_bootstrap_requested": bootstrap["requested"],
        "paired_bootstrap_accepted": bootstrap["accepted"],
        "paired_bootstrap_seed": bootstrap["seed"],
        "paired_delong_delta_ci_lower": delong["analytic_normal_95_ci"][0],
        "paired_delong_delta_ci_upper": delong["analytic_normal_95_ci"][1],
        "paired_delong_standard_error": delong["standard_error"],
        "paired_delong_z": delong["z_candidate_minus_reference"],
        "paired_delong_p_two_sided": delong["two_sided_p_value"],
        "reference_prediction_sha256": provenance["reference_prediction"]["sha256"],
        "candidate_prediction_sha256": provenance["candidate_prediction"]["sha256"],
        "analysis_script_sha256": provenance["analysis_script"]["sha256"],
        "id_sets_exactly_equal": report["validation"]["id_sets_exactly_equal"],
        "labels_exactly_equal_after_id_alignment": report["validation"][
            "labels_exactly_equal_after_id_alignment"
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    candidate = load_prediction(args.candidate)
    reference = load_prediction(args.reference)
    labels, candidate_scores, validation = align_candidate(reference, candidate)
    reference_scores = np.asarray(reference["scores"], dtype=np.float64)

    boot = paired_bootstrap(
        labels,
        candidate_scores,
        reference_scores,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    candidate_report = model_report(
        args.candidate_name, labels, candidate_scores, boot["candidate_auc_values"]
    )
    reference_report = model_report(
        args.reference_name, labels, reference_scores, boot["reference_auc_values"]
    )
    point_delta = candidate_report["auc"] - reference_report["auc"]
    delong = paired_delong(labels, reference_scores, candidate_scores)
    if not np.isclose(
        point_delta,
        delong["delta_auc_candidate_minus_reference"],
        rtol=0.0,
        atol=1e-14,
    ):
        raise RuntimeError("DeLong and sklearn point-estimate directions differ")
    manifest_validation = validate_candidate_manifest(
        candidate, candidate_report["auc"], args.require_candidate_run_manifest
    )

    script_path = Path(__file__).resolve()
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "paired study-protocol DeiT-Tiny versus TinyNet-C top-five ensembles",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "candidate AUROC minus reference AUROC on the identical internal-test participants",
        "validation": validation,
        "methods": {
            "sampling_unit": "participant",
            "bootstrap": {
                "resampling": "paired nonparametric bootstrap with replacement",
                "requested": boot["requested"],
                "accepted": boot["accepted"],
                "single_class_rejected": boot["single_class_rejected"],
                "seed": boot["seed"],
                "confidence_interval": "2.5th and 97.5th percentiles",
                "coverage": 0.95,
                "numpy_percentile_method": "linear interpolation (default)",
            },
            "hypothesis_test": "two-sided paired DeLong test",
            "delong_confidence_interval": "analytic normal 95% interval for candidate minus reference",
        },
        "results": {
            "reference": reference_report,
            "candidate": candidate_report,
            "paired_comparison": {
                "delta_auc_candidate_minus_reference": float(point_delta),
                "participant_bootstrap": {
                    "percentile_95_ci": percentile_interval(
                        boot["delta_auc_values"], 2.5, 97.5
                    ),
                    "requested": boot["requested"],
                    "accepted": boot["accepted"],
                    "single_class_rejected": boot["single_class_rejected"],
                    "seed": boot["seed"],
                },
                "paired_delong": delong,
            },
        },
        "candidate_run_manifest_validation": manifest_validation,
        "provenance": {
            "candidate_prediction": file_fact(Path(candidate["path"])),
            "reference_prediction": file_fact(Path(reference["path"])),
            "candidate_config_canonical_sha256": canonical_json_sha256(candidate["config"]),
            "reference_config_canonical_sha256": canonical_json_sha256(reference["config"]),
            "candidate_adjacent_artifacts": adjacent_provenance(Path(candidate["path"])),
            "reference_adjacent_artifacts": adjacent_provenance(Path(reference["path"])),
            "analysis_script": file_fact(script_path),
            "command": [str(script_path)] + sys.argv[1:],
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deit_vs_study_tinynet_comparison.json"
    csv_path = output_dir / "deit_vs_study_tinynet_comparison.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(report, csv_path)
    checksums = {
        json_path.name: sha256_file(json_path),
        csv_path.name: sha256_file(csv_path),
    }
    checksum_path = output_dir / "comparison_output_sha256.json"
    checksum_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_auc": candidate_report["auc"],
                "candidate_auc_95_ci": candidate_report["percentile_95_ci"],
                "reference_auc": reference_report["auc"],
                "reference_auc_95_ci": reference_report["percentile_95_ci"],
                "delta_auc_candidate_minus_reference": point_delta,
                "paired_bootstrap_delta_auc_95_ci": report["results"][
                    "paired_comparison"
                ]["participant_bootstrap"]["percentile_95_ci"],
                "paired_delong_p_value": delong["two_sided_p_value"],
                "outputs": [str(json_path), str(csv_path), str(checksum_path)],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
