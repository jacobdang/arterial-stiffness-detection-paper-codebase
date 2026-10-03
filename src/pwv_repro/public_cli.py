"""Public, data-free command line for aggregate verification and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_inference import DEFAULT_RELEASE_CHECKPOINT, predict_preprocessed_pair
from .public_aggregate_validation import validate_public_aggregates


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="public repository root (default: current directory)",
    )
    subcommands = command.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "verify-aggregates",
        help="verify ID-free economics and net-benefit arithmetic at threshold 0.20",
    )
    inference = subcommands.add_parser(
        "predict-images",
        help="run single-checkpoint bilateral image inference",
    )
    inference.add_argument("--left", type=Path, required=True)
    inference.add_argument("--right", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, default=DEFAULT_RELEASE_CHECKPOINT)
    inference.add_argument("--device", default="cpu")
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    if args.command == "verify-aggregates":
        result = validate_public_aggregates(root)
    elif args.command == "predict-images":
        checkpoint = args.checkpoint
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        result = predict_preprocessed_pair(
            args.left,
            args.right,
            checkpoint,
            device=args.device,
        )
    else:  # pragma: no cover - argparse enforces the finite command set
        raise RuntimeError(f"Unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
