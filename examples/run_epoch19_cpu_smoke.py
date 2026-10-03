"""Run the public single-checkpoint model on the synthetic bilateral fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pwv_repro.model_inference import predict_preprocessed_pair


EXPECTED_CHECKPOINT_SHA256 = (
    "5f0aa9851341b5a1c39168feeba6a7c8a1f79149e0acf555a785afcb4b1fbe22"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON. If omitted, print only and do not modify the repository.",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    example_dir = Path(__file__).resolve().parent
    fixture_manifest_path = example_dir / "synthetic_fixture_manifest.json"
    fixture = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    if not (
        fixture.get("synthetic") is True
        and fixture.get("nonclinical") is True
        and fixture.get("contains_patient_data") is False
        and fixture.get("derived_from_real_retinal_image") is False
    ):
        raise RuntimeError("Synthetic fixture privacy/role contract failed")
    for label in ("left", "right"):
        image = example_dir / fixture["images"][label]["path"]
        if sha256(image) != fixture["images"][label]["sha256"]:
            raise RuntimeError(f"Synthetic {label} fixture hash mismatch")

    checkpoint = (
        repo
        / "models/public_release/paper_main_tinynet_c/"
        "tinynet_c_epoch0019_network_state_dict.safetensors"
    )
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Released epoch-19 checkpoint hash mismatch")
    inference = predict_preprocessed_pair(
        example_dir / fixture["images"]["left"]["path"],
        example_dir / fixture["images"]["right"]["path"],
        checkpoint,
        device="cpu",
    )
    score = float(inference["arterial_stiffness_score"])
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("Synthetic smoke score is outside [0,1]")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "device": "cpu",
        "fixture_manifest": fixture_manifest_path.name,
        "fixture_manifest_sha256": sha256(fixture_manifest_path),
        "checkpoint": "models/public_release/paper_main_tinynet_c/tinynet_c_epoch0019_network_state_dict.safetensors",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "strict_checkpoint_load": True,
        "synthetic_score": score,
        "score_role": "engineering smoke output only; not clinical or performance evidence",
        "checkpoint_role": (
            "individual TinyNet-C checkpoint"
        ),
        "contains_patient_data": False,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
