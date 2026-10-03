from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_cli_help_renders_without_argparse_format_errors():
    completed = subprocess.run(
        [sys.executable, "-m", "pwv_repro.public_cli", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verify-aggregates" in completed.stdout
    assert "predict-images" in completed.stdout


def test_public_cli_verifies_aggregates_without_authorized_rows():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pwv_repro.public_cli",
            "--repo-root",
            str(ROOT),
            "verify-aggregates",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["economic"]["n"] == 7331


def test_predict_images_cli_forwards_user_supplied_bilateral_paths(monkeypatch, tmp_path, capsys):
    from pwv_repro import public_cli

    left = tmp_path / "authorized_left.png"
    right = tmp_path / "authorized_right.png"
    left.write_bytes(b"left fixture placeholder")
    right.write_bytes(b"right fixture placeholder")
    observed = {}

    def fake_predict(left_image, right_image, checkpoint, device):
        observed.update(
            left=left_image,
            right=right_image,
            checkpoint=checkpoint,
            device=device,
        )
        return {"status": "PASS", "input_contract": "fixture"}

    monkeypatch.setattr(public_cli, "predict_preprocessed_pair", fake_predict)
    assert public_cli.main(
        [
            "--repo-root",
            str(ROOT),
            "predict-images",
            "--left",
            str(left),
            "--right",
            str(right),
            "--device",
            "cpu",
        ]
    ) == 0
    assert observed == {
        "left": left,
        "right": right,
        "checkpoint": ROOT
        / "models/public_release/paper_main_tinynet_c/"
        "tinynet_c_epoch0019_network_state_dict.safetensors",
        "device": "cpu",
    }
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
