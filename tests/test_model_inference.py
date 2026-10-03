from pathlib import Path

import numpy as np
import pytest


def test_release_checkpoint_runs_id_free_cpu_demo(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    image_module = pytest.importorskip("PIL.Image")
    from pwv_repro.model_inference import predict_preprocessed_pair

    root = Path(__file__).resolve().parents[1]
    checkpoint = (
        root
        / "models/public_release/paper_main_tinynet_c/tinynet_c_epoch0019_network_state_dict.safetensors"
    )
    if not checkpoint.exists():
        pytest.skip("Release checkpoint is distributed separately from the lightweight source bundle")
    gradient = np.linspace(0, 255, 384, dtype=np.uint8)
    pixels = np.repeat(gradient[None, :, None], 384, axis=0)
    pixels = np.repeat(pixels, 3, axis=2)
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    image_module.fromarray(pixels).save(left)
    image_module.fromarray(np.flip(pixels, axis=1)).save(right)

    result = predict_preprocessed_pair(left, right, checkpoint, device="cpu")
    assert 0.0 <= result["arterial_stiffness_score"] <= 1.0
    assert "individual TinyNet-C checkpoint" in result["checkpoint_role"]
    assert result["input_contract"]["shape"] == [1, 2, 3, 384, 384]
    assert result["checkpoint"] == checkpoint.name
    assert result["checkpoint_sha256"] == (
        "5f0aa9851341b5a1c39168feeba6a7c8a1f79149e0acf555a785afcb4b1fbe22"
    )
    assert str(checkpoint.parent) not in str(result)
