"""Check the published aggregate screening-cost and net-benefit calculations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _close(observed: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=atol):
        raise ValueError(f"{label}: {observed!r} != {expected!r}")


def validate_economic_aggregate(path: Path) -> Dict[str, Any]:
    """Verify the public per-screened-participant cost-scenario arithmetic."""

    report = _load_json(path)
    n = int(report["n"])
    events = int(report["events"])
    nonevents = int(report["nonevents"])
    pwv_cost = float(report["pwv_test_cost_cny"])
    if n <= 0 or events <= 0 or nonevents <= 0 or events + nonevents != n:
        raise ValueError("Invalid aggregate cohort counts")
    if pwv_cost <= 0:
        raise ValueError("PWV test cost must be positive")

    points = report.get("operating_points")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("Expected four frozen operating points")
    previous_target = -math.inf
    verified = []
    for index, point in enumerate(points, start=1):
        tp, tn, fp, fn = (int(point[name]) for name in ("tp", "tn", "fp", "fn"))
        if min(tp, tn, fp, fn) < 0:
            raise ValueError(f"Operating point {index} contains a negative count")
        if tp + fn != events or tn + fp != nonevents or tp + tn + fp + fn != n:
            raise ValueError(f"Operating point {index} confusion counts do not close")

        target = float(point["target_sensitivity"])
        threshold = float(point["threshold"])
        if point.get("threshold_rounded_for_public_release") is not True:
            raise ValueError(f"Operating point {index} threshold lacks public-rounding label")
        if not math.isclose(threshold, round(threshold, 3), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Operating point {index} exposes an unrounded empirical cutoff")
        achieved = tp / events
        referral = (tp + fp) / n
        specificity = tn / nonevents
        precision = tp / (tp + fp)
        if target <= previous_target:
            raise ValueError("Target sensitivities are not strictly increasing")
        previous_target = target
        _close(point["achieved_sensitivity"], achieved, f"point {index} sensitivity")
        _close(point["referral_fraction"], referral, f"point {index} referral fraction")
        _close(point["specificity"], specificity, f"point {index} specificity")
        _close(point["precision"], precision, f"point {index} precision")

        cost_zero = referral * pwv_cost
        cost_forty = 40.0 + cost_zero
        break_even = pwv_cost - cost_zero
        _close(
            point["cost_cny_no_attendance_adjustment_fundus_0"],
            cost_zero,
            f"point {index} zero-fundus cost",
        )
        _close(
            point["cost_cny_no_attendance_adjustment_fundus_40"],
            cost_forty,
            f"point {index} 40-CNY-fundus cost",
        )
        _close(
            point["break_even_fundus_cost_cny"],
            break_even,
            f"point {index} break-even cost",
        )
        verified.append(
            {
                "target_sensitivity": target,
                "achieved_sensitivity": achieved,
                "referral_fraction": referral,
                "two_stage_cost_cny_zero_incremental_fundus": cost_zero,
                "break_even_incremental_fundus_cost_cny": break_even,
            }
        )

    return {
        "status": "PASS",
        "scope": "Screening costs calculated from aggregate confusion counts",
        "n": n,
        "events": events,
        "pwv_test_cost_cny": pwv_cost,
        "operating_points": verified,
    }


def validate_clinical_utility_aggregate(path: Path) -> Dict[str, Any]:
    """Verify the public standard net-benefit calculation at threshold 0.20."""

    report = _load_json(path)
    point = report["standard_net_benefit_at_0_20"]
    n = int(point["n"])
    events = int(point["events"])
    tp = int(point["tp"])
    fp = int(point["fp"])
    threshold = float(point["threshold"])
    if not (0.0 < threshold < 1.0):
        raise ValueError("Decision threshold must lie strictly between zero and one")
    if not (0 <= tp <= events <= n) or not (0 <= fp <= n - events):
        raise ValueError("Invalid aggregate decision counts")
    odds = threshold / (1.0 - threshold)
    model_nb = tp / n - fp / n * odds
    all_nb = events / n - (n - events) / n * odds
    _close(point["fusion_model"], model_nb, "fusion net benefit")
    _close(point["treat_all"], all_nb, "treat-all net benefit")
    _close(point["treat_none"], 0.0, "treat-none net benefit")
    _close(
        point["fusion_minus_treat_all"],
        model_nb - all_nb,
        "fusion minus treat-all net benefit",
    )
    return {
        "status": "PASS",
        "scope": "Net benefit calculated at a decision threshold of 0.20",
        "threshold": threshold,
        "n": n,
        "events": events,
        "fusion_net_benefit": model_nb,
        "treat_all_net_benefit": all_nb,
        "treat_none_net_benefit": 0.0,
    }


def validate_public_aggregates(repo_root: Path) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    return {
        "status": "PASS",
        "economic": validate_economic_aggregate(
            root / "results/clinical_utility/screening_costs.json"
        ),
        "clinical_utility": validate_clinical_utility_aggregate(
            root / "results/clinical_utility/net_benefit.json"
        ),
    }
