"""Generate a deterministic, nonclinical bilateral RGB engineering fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SIZE = 384
SEEDS = {"left": 2026082601, "right": 2026082602}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_eye(seed: int, mirror: bool) -> Image.Image:
    """Draw an intentionally schematic circular field and branching curves."""

    rng = random.Random(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cx = cy = (SIZE - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = radius <= 178
    radial = np.clip(1.0 - radius / 178.0, 0.0, 1.0)
    array = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    array[..., 0] = np.where(mask, 78 + 118 * radial, 0).astype(np.uint8)
    array[..., 1] = np.where(mask, 20 + 57 * radial, 0).astype(np.uint8)
    array[..., 2] = np.where(mask, 17 + 32 * radial, 0).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGB")

    disc_x = 272 if not mirror else 112
    disc_y = 190
    draw.ellipse(
        (disc_x - 22, disc_y - 29, disc_x + 22, disc_y + 29),
        fill=(224, 161, 83),
        outline=(244, 193, 112),
        width=3,
    )

    for branch in range(18):
        angle = (branch / 18.0) * 2.0 * np.pi + rng.uniform(-0.10, 0.10)
        length = rng.randint(86, 154)
        points = [(disc_x, disc_y)]
        for step in range(1, 8):
            fraction = step / 7.0
            bend = rng.uniform(-9.0, 9.0) * fraction
            px = disc_x + np.cos(angle) * length * fraction + np.sin(angle) * bend
            py = disc_y + np.sin(angle) * length * fraction - np.cos(angle) * bend
            points.append((int(round(px)), int(round(py))))
        width = 3 if branch % 3 else 4
        color = (78, 13, 18) if branch % 2 else (112, 25, 27)
        draw.line(points, fill=color, width=width, joint="curve")

    # A slight deterministic blur suppresses pixel-level drawing aliasing. This
    # remains a schematic engineering pattern, not a simulated clinical image.
    image = image.filter(ImageFilter.GaussianBlur(radius=0.55))
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mirror else image


def main() -> int:
    output_dir = Path(__file__).resolve().parent
    records = {}
    for label, seed in SEEDS.items():
        path = output_dir / f"synthetic_{label}_384.png"
        image = synthetic_eye(seed, mirror=(label == "right"))
        image.save(path, format="PNG", compress_level=9, optimize=False)
        records[label] = {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "pixels": [SIZE, SIZE],
            "mode": "RGB",
            "seed": seed,
        }
    manifest = {
        "schema_version": 1,
        "fixture_role": "deterministic engineering smoke test only",
        "synthetic": True,
        "nonclinical": True,
        "contains_patient_data": False,
        "derived_from_real_retinal_image": False,
        "performance_validation": False,
        "generator": Path(__file__).name,
        "generator_sha256": sha256(Path(__file__)),
        "images": records,
    }
    manifest_path = output_dir / "synthetic_fixture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
