import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pwv_repro.public_aggregate_validation import validate_public_aggregates


def test_public_aggregate_arithmetic_closes_exactly():
    result = validate_public_aggregates(ROOT)
    assert result["status"] == "PASS"
    economic = result["economic"]
    assert economic["n"] == 7331
    assert economic["events"] == 5158
    assert len(economic["operating_points"]) == 4
    assert round(economic["operating_points"][0]["two_stage_cost_cny_zero_incremental_fundus"], 2) == 83.60
    utility = result["clinical_utility"]
    assert utility["threshold"] == 0.20
    assert utility["fusion_net_benefit"] > utility["treat_all_net_benefit"]


def test_public_aggregate_cli_runs_without_authorized_rows():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_public_aggregates.py"),
            "--repo-root",
            str(ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["economic"]["pwv_test_cost_cny"] == 120.0
