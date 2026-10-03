#!/usr/bin/env python3
"""Recover/evaluate the validation-ranked top-five ensemble from a run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_CODE = REPO_ROOT / "main_cls_code_dl"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the runner first: it prepends and verifies the isolated timm 0.9.2
# package before reference network modules have an opportunity to import timm.
from run_top5 import BilateralDeiTTiny, sha256_file  # noqa: E402

sys.path.insert(0, str(IMAGE_CODE))
from procedure import default_test_loop  # noqa: E402
from test_main import get_dataloader_helper, get_top_checkpoint_list  # noqa: E402
from utils import get_logger, logging_info  # noqa: E402
from network.generic_bi_stream_img_only import GenericBiStreamImgOnly  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("deit_tiny", "tinynet_c"), required=True)
    return parser.parse_args()


def read_config(log_path: Path):
    first_line = log_path.read_text().splitlines()[0]
    marker = "]: "
    if marker not in first_line:
        raise ValueError(f"Cannot parse configuration from {log_path}")
    return ast.literal_eval(first_line.split(marker, 1)[1])


def checkpoint_auc(path: Path) -> float:
    return float(path.stem.rsplit("_auc_", 1)[1])


def save_payload(path, config, labels, probabilities, patient_ids, features):
    payload = {
        "config": config,
        "gt_set": labels,
        "pred_set": probabilities,
        "id_set": patient_ids,
        "feat_set": features,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = read_config(run_dir / "train_main.log")
    logger = get_logger(f"top5_eval_{args.model}", run_dir / "top5_evaluation.log")

    if args.model == "deit_tiny":
        network = BilateralDeiTTiny(config["model"]["approximate_head_dim"])
    else:
        network = GenericBiStreamImgOnly("tinynet_c", config["model"]["approximate_head_dim"])
    network = network.cuda()
    model = torch.nn.DataParallel(network)

    checkpoints = [
        Path(path) for path in get_top_checkpoint_list(str(run_dir), no_checkpoints=5)
    ]
    if len(checkpoints) != 5:
        raise RuntimeError(f"Expected five checkpoints, found {len(checkpoints)}")
    logging_info(logger, "top_ranked_checkpoint_list:" + str([str(p) for p in checkpoints]))

    loaders = {
        split: get_dataloader_helper(config, split, logger)
        for split in ("train", "val", "test")
    }
    predictions = {split: [] for split in loaders}
    features = {split: [] for split in loaders}
    references = {}
    per_checkpoint = []

    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        network.load_state_dict(checkpoint["network"], strict=True)
        record = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "validation_auc_saved": float(checkpoint["config"].get("validation_auc", checkpoint_auc(checkpoint_path))) if isinstance(checkpoint.get("config"), dict) else checkpoint_auc(checkpoint_path),
        }
        for split, loader in loaders.items():
            labels, scores, patient_ids, _, auc, feats = default_test_loop(
                loader, model, "img_only", ema=None
            )
            if split not in references:
                references[split] = (labels.copy(), patient_ids.copy())
            elif not np.array_equal(references[split][0], labels) or not np.array_equal(references[split][1], patient_ids):
                raise RuntimeError(f"{split} labels/IDs changed across checkpoints")
            predictions[split].append(scores)
            features[split].append(feats)
            record[f"{split}_auc"] = float(auc)
        per_checkpoint.append(record)

    ensemble_auc = {}
    output_hashes = {}
    for split in loaders:
        labels, patient_ids = references[split]
        mean_probability = np.mean(np.asarray(predictions[split]), axis=0)
        mean_feature = np.mean(np.asarray(features[split]), axis=0)
        ensemble_auc[split] = float(roc_auc_score(labels, mean_probability))
        output_path = run_dir / f"top_checkpoint_ensemble_{split}_result.pickle"
        save_payload(output_path, config, labels, mean_probability, patient_ids, mean_feature)
        output_hashes[split] = sha256_file(output_path)

    manifest = {
        "completed": True,
        "model": args.model,
        "selection": "five highest validation-AUROC epoch checkpoints",
        "aggregation": "arithmetic mean of five fixed checkpoint probabilities",
        "top_five": per_checkpoint,
        "ensemble_auc": ensemble_auc,
        "output_sha256": output_hashes,
    }
    (run_dir / "top5_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
