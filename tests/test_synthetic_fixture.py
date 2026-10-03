import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_fixture_is_explicitly_nonclinical_and_hash_bound():
    manifest_path = ROOT / "examples/synthetic_fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert manifest["nonclinical"] is True
    assert manifest["contains_patient_data"] is False
    assert manifest["derived_from_real_retinal_image"] is False
    assert manifest["performance_validation"] is False
    generator = ROOT / "examples" / manifest["generator"]
    assert sha256(generator) == manifest["generator_sha256"]
    for label in ("left", "right"):
        record = manifest["images"][label]
        path = ROOT / "examples" / record["path"]
        assert sha256(path) == record["sha256"]
        with Image.open(path) as image:
            assert image.size == (384, 384)
            assert image.mode == "RGB"


def test_epoch19_cpu_smoke_record_is_public_safe_and_single_checkpoint_only():
    result = json.loads(
        (ROOT / "examples/epoch19_cpu_smoke_result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "PASS"
    assert result["device"] == "cpu"
    assert result["strict_checkpoint_load"] is True
    assert result["contains_patient_data"] is False
    assert result["fixture_manifest_sha256"] == sha256(
        ROOT / "examples/synthetic_fixture_manifest.json"
    )
    assert 0.0 <= result["synthetic_score"] <= 1.0
    assert "individual TinyNet-C checkpoint" in result["checkpoint_role"]
    serialized = json.dumps(result)
    for marker in tuple("/" + name for name in ("ssd", "mnt", "home")):
        assert marker not in serialized
