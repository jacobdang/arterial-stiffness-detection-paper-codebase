from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def test_json_fusion_models_are_exact_non_executable_representations():
    path = ROOT / "models/public_release/fusion_lr_20mi/fusion_lr_20mi_coefficients.json"
    if not path.is_file():
        pytest.skip("released model file is not available")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["models"]) == 20
    assert payload["verification"]["authorized_reference_max_abs_error"] <= 1e-12
    assert payload["input_contract"]["values_are_not_raw_clinical_measurements"] is True
    manual_by_fixture = [[] for _ in payload["synthetic_engineering_fixtures"]]
    for record in payload["models"]:
        coefficient = np.asarray(
            [item["coefficient"] for item in record["coefficients"]], dtype=np.float64
        )
        for fixture_index, fixture in enumerate(payload["synthetic_engineering_fixtures"]):
            array = np.asarray(fixture, dtype=np.float64)
            observed = _sigmoid(float(record["intercept"]) + float(coefficient @ array))
            expected = float(record["synthetic_fixture_positive_probabilities"][fixture_index])
            assert abs(observed - expected) <= 1e-12
            manual_by_fixture[fixture_index].append(observed)
    assert all(np.isfinite(np.mean(values)) for values in manual_by_fixture)


@pytest.mark.parametrize("epoch", [14, 19])
def test_released_safetensors_are_strictly_loadable(epoch):
    safetensors = pytest.importorskip("safetensors.torch")
    torch = pytest.importorskip("torch")
    safe = ROOT / (
        "models/public_release/paper_main_tinynet_c/"
        f"tinynet_c_epoch{epoch:04d}_network_state_dict.safetensors"
    )
    if not safe.is_file():
        pytest.skip(f"epoch-{epoch} safetensors export has not been generated")
    loaded = safetensors.load_file(safe, device="cpu")
    assert len(loaded) == 348
    assert all(torch.isfinite(tensor).all() for tensor in loaded.values())
    from pwv_repro.public_models import build_tinynet_c_classifier

    model = build_tinynet_c_classifier(pretrained=False)
    incompatible = model.load_state_dict(loaded, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
