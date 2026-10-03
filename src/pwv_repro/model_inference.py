"""Single-checkpoint inference for preprocessed bilateral fundus images."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

from .public_models import build_tinynet_c_classifier


DEFAULT_RELEASE_CHECKPOINT = (
    Path("models")
    / "public_release"
    / "paper_main_tinynet_c"
    / "tinynet_c_epoch0019_network_state_dict.safetensors"
)


def _load_rgb_tensor(path: Path, image_size: int = 384):
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageOps
    except ImportError as error:  # pragma: no cover - depends on optional DL environment
        raise RuntimeError(
            "Image inference requires the inference optional dependencies and Pillow."
        ) from error

    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError("Image not found: {}".format(image_path))
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        # Model inputs are fundus-field cropped, padded to a
        # square, and resized to 384 px.  ImageOps.pad is a safe convenience for
        # a preprocessed image of another size; it is not a substitute for the
        # study raw-image field-of-view extraction.
        if image.size != (image_size, image_size):
            image = ImageOps.pad(
                image,
                (image_size, image_size),
                method=Image.Resampling.BICUBIC,
                color=(0, 0, 0),
                centering=(0.5, 0.5),
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def predict_preprocessed_pair(
    left_image: Path,
    right_image: Path,
    checkpoint: Path,
    device: str = "auto",
) -> Dict[str, object]:
    """Return a single-checkpoint sigmoid score for a left/right image pair."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional DL environment
        raise RuntimeError(
            "Image inference requires torch, timm, Pillow, and safetensors."
        ) from error

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    if device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = device

    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The .safetensors format requires the inference optional dependencies."
            ) from error
        saved = load_safetensors(str(checkpoint_path), device="cpu")
        serialization = "safetensors"
    else:
        raise RuntimeError(
            "The public inference interface accepts safetensors only. "
            "Convert trusted weights to safetensors before using this interface."
        )
    model = build_tinynet_c_classifier(pretrained=False)
    incompatible = model.load_state_dict(saved, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load unexpectedly returned incompatible keys")
    model.eval().to(selected_device)

    left = _load_rgb_tensor(left_image)
    right = _load_rgb_tensor(right_image)
    batch = torch.stack((left, right), dim=0).unsqueeze(0).to(selected_device)
    with torch.inference_mode():
        logit, _ = model(batch)
        score = torch.sigmoid(logit).reshape(-1)[0].item()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "arterial_stiffness_score": float(score),
        "output_semantics": "single-checkpoint sigmoid score",
        # Never serialize the resolved local path: authorized-site paths can
        # contain institutional or user-specific information.
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_role": "individual TinyNet-C checkpoint",
        "serialization": serialization,
        "architecture": "GenericBiStreamImgOnly / TinyNet-C / mean eye aggregation / gated 2048-d head",
        "input_contract": {
            "eye_order": ["left", "right"],
            "color": "RGB",
            "shape": [1, 2, 3, 384, 384],
            "normalization": "(pixel/255 - 0.5) / 0.5, yielding [-1, 1]",
            "required_preprocessing": "fundus field cropped, black-padded square, resized to 384x384",
            "missing_eye_rule": (
                "if only one eye is available, duplicate that same preprocessed image "
                "into both eye inputs"
            ),
        },
        "device": str(selected_device),
    }
