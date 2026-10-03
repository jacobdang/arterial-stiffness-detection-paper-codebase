#!/usr/bin/env python3
"""Run a DeiT/TinyNet model through the reference one-seed top-five protocol.

The reference data pipeline and its training/evaluation helpers are imported
directly.  Only the minimal DeiT backbone adapter and an FP32 form of the old
training loop are local to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import CyclicLR


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_TIMM = Path(__file__).resolve().parent / "vendor" / "timm_0_9_2"
if not VENDORED_TIMM.is_dir():
    raise RuntimeError(f"Frozen timm 0.9.2 directory is missing: {VENDORED_TIMM}")
sys.path.insert(0, str(VENDORED_TIMM))
import timm  # noqa: E402

if timm.__version__ != "0.9.2":
    raise RuntimeError(f"Study protocol requires timm 0.9.2, found {timm.__version__}")

IMAGE_CODE = REPO_ROOT / "main_cls_code_dl"
sys.path.insert(0, str(IMAGE_CODE))

from dataloader import get_data_loader  # noqa: E402
from network.generic_bi_stream_img_only import GenericBiStreamImgOnly  # noqa: E402
from procedure import (  # noqa: E402
    default_test_loop,
    default_train_loop,
    get_loss_and_optimizer,
    save_checkpoint,
)
from test_main import get_dataloader_helper, get_top_checkpoint_list  # noqa: E402
from utils import get_logger, logging_info  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    """Hash tensor names/content without depending on checkpoint serialization."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_module_finite(module: nn.Module, context: str) -> None:
    for name, value in module.named_parameters():
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"non-finite parameter after {context}: {name}")


class BilateralDeiTTiny(nn.Module):
    """DeiT-Tiny with a 2,048-dimensional adapter and bilateral attention head."""

    def __init__(self, head_dim: int = 2048):
        super().__init__()
        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=True,
            num_classes=0,
            img_size=384,
        )
        out_channels = int(self.backbone.num_features)
        self.project = nn.Sequential(
            nn.Linear(out_channels, head_dim),
            nn.BatchNorm1d(head_dim),
            nn.SiLU(inplace=True),
        )
        squeezed = max(1, int(0.5 * head_dim))
        self.combined_meta_attention = nn.Sequential(
            nn.Linear(head_dim, squeezed),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(squeezed),
            nn.Linear(squeezed, head_dim),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(head_dim, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        eye_features = []
        for eye_index in range(inputs.size(1)):
            vector = self.backbone(inputs[:, eye_index, ...])
            projected = self.project(vector)
            eye_features.append(projected)
        combined = torch.stack(eye_features).mean(dim=0)
        attention = self.combined_meta_attention(combined)
        final_feature = combined * attention
        return self.fc(final_feature), final_feature


def fp32_train_loop(epoch, config, loader, model, criterion, optimizer, scheduler, max_batches=None):
    """Literal reference loop with autocast/scaler operations disabled."""

    from procedure import eval_main_tensor

    model.train()
    losses, accuracies, aucs = [], [], []
    for batch_index, (data_input, ground_truth, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad()
        ground_truth = ground_truth.unsqueeze(1).float().cuda(non_blocking=True)
        image = data_input.cuda(non_blocking=True)
        logits, _ = model(image)
        probability = torch.sigmoid(logits)
        loss = criterion(logits, ground_truth)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.module.parameters(), config["gradient_clip"])
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
            raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
        optimizer.step()
        scheduler.step()
        accuracy, auc = eval_main_tensor(probability, ground_truth)
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float(accuracy.detach().cpu()))
        aucs.append(float(auc.detach().cpu()))
    return float(np.mean(losses)), float(np.mean(accuracies)), float(np.mean(aucs))


def amp_smoke_train_loop(epoch, config, loader, model, criterion, optimizer, scheduler, scaler, max_batches):
    """Bounded copy of the reference AMP loop, used only for preflight."""

    from procedure import eval_main_tensor

    model.train()
    losses, accuracies, aucs = [], [], []
    for batch_index, (data_input, ground_truth, _) in enumerate(loader):
        if batch_index >= max_batches:
            break
        optimizer.zero_grad()
        ground_truth = ground_truth.unsqueeze(1).float().cuda(non_blocking=True)
        image = data_input.cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            logits, _ = model(image)
            probability = torch.sigmoid(logits)
            loss = criterion(logits, ground_truth)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite AMP smoke loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), config["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        accuracy, auc = eval_main_tensor(probability, ground_truth)
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float(accuracy.detach().cpu()))
        aucs.append(float(auc.detach().cpu()))
    return float(np.mean(losses)), float(np.mean(accuracies)), float(np.mean(aucs))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("deit_tiny", "tinynet_c"), default="deit_tiny")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Directory containing the already cropped/padded 384x384 study images",
    )
    parser.add_argument(
        "--split-dir", type=Path, required=True,
        help="Directory containing the three authorized participant split CSV files",
    )
    parser.add_argument(
        "--occlusion-root", type=Path,
        help="Optional precomputed occlusion-image tree (unused for the Q2 run)",
    )
    parser.add_argument("--precision", choices=("amp", "fp32"), default="amp")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--skip-test-each-epoch", action="store_true")
    parser.add_argument("--skip-final-ensemble", action="store_true")
    return parser.parse_args()


def build_config(args):
    split_dir = args.split_dir.expanduser().resolve()
    return {
        "dataset": {
            "name": "random_split_img_only_datset",
            "occlusion_seg_path": (
                str(args.occlusion_root.expanduser().resolve())
                if args.occlusion_root is not None else None
            ),
            "root_path": str(args.data_root.expanduser().resolve()),
            "train_csv_name": str(split_dir / "train_table_orig_random_img_only_patient_level.csv"),
            "val_csv_name": str(split_dir / "val_table_orig_random_img_only_patient_level.csv"),
            "test_csv_name": str(split_dir / "test_table_orig_random_img_only_patient_level.csv"),
            "feature_list": None,
            "img_size": 384,
            "th": 1400,
        },
        "model": {
            "network_arch": "GenericBiStreamImgOnly",
            "backbone_name": "deit_tiny_patch16_224" if args.model == "deit_tiny" else "tinynet_c",
            "approximate_head_dim": 2048,
            "wd": 0.001,
        },
        "lr_scheduler": {
            "name": "CyclicLR",
            "base_lr": 0.0002,
            "max_lr": 0.002,
            "step_size_up": 1500,
            "gamma": 0.99995,
        },
        "desc": "study_top5_protocol_20260903",
        "seed": 0,
        "train_ratio": None,
        "weighted_class_sampling_type": 1,
        "aug_level": 2,
        "batch_size": 64,
        "num_workers": int(args.num_workers),
        "max_epochs": int(args.epochs),
        "occlusion_seg_index": None,
        "loss_type": "bce_logits_loss",
        "optimizer_type": "adamw",
        "lr": 0.001,
        "ema_decay": None,
        "gradient_clip": 5,
        "pretrained_checkpoint_path": "",
        "test_checkpoint_path": "",
        "resume_training": False,
        "epoch_saving_interval": 1,
        "precision": args.precision,
        "model_key": args.model,
    }


def assert_protocol(config, args):
    expected = {
        "seed": 0,
        "weighted_class_sampling_type": 1,
        "aug_level": 2,
        "batch_size": 64,
        "max_epochs": 45,
        "loss_type": "bce_logits_loss",
        "optimizer_type": "adamw",
        "lr": 0.001,
        "ema_decay": None,
        "gradient_clip": 5,
        "epoch_saving_interval": 1,
        "num_workers": 16,
    }
    if args.max_train_batches is None:
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Study production runs require exactly one visible GPU; "
                f"found {torch.cuda.device_count()}"
            )
        mismatches = {key: (config[key], value) for key, value in expected.items() if config[key] != value}
        if mismatches:
            raise RuntimeError(f"Production protocol mismatch: {mismatches}")
    expected_hashes = {
        "train": "df381b8cd8f0d8b9170cae2ee17bbdcc3677bcdea47e7f8961de254c94bb0abd",
        "val": "5b4c216e0e0501077aa6b93637289cf1b25bbc06125d2639cc499a62a8600ca0",
        "test": "c3f54c8b2a89e5e2d453dd50da9e5bdc5445feb623aad895671cfe4aa7cc260f",
    }
    actual = {}
    for split in expected_hashes:
        path = Path(config["dataset"][f"{split}_csv_name"])
        actual[split] = sha256_file(path)
    if actual != expected_hashes:
        raise RuntimeError(f"Split hash mismatch: {actual}")
    return actual


def save_result(config, output_dir, name, result):
    ground_truth, predictions, patient_ids, _, auc, features = result
    payload = {
        "config": config,
        "gt_set": ground_truth,
        "pred_set": predictions,
        "id_set": patient_ids,
        "feat_set": features,
    }
    with (output_dir / name).open("wb") as handle:
        pickle.dump(payload, handle)
    return float(auc)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoint").mkdir()
    config = build_config(args)
    split_hashes = assert_protocol(config, args)

    torch.cuda.empty_cache()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    logger = get_logger(f"study_top5_{args.model}_{args.precision}", output_dir / "train_main.log")
    logging_info(logger, str(config))
    logging_info(logger, f"visible CUDA devices: {torch.cuda.device_count()}")
    logging_info(logger, f"environment: torch={torch.__version__}, timm={timm.__version__}, python={platform.python_version()}")

    if args.model == "deit_tiny":
        network = BilateralDeiTTiny(config["model"]["approximate_head_dim"])
    else:
        network = GenericBiStreamImgOnly("tinynet_c", config["model"]["approximate_head_dim"])
    source_hashes_at_start = {
        "runner": sha256_file(Path(__file__)),
        "dataloader.py": sha256_file(IMAGE_CODE / "dataloader.py"),
        "procedure.py": sha256_file(IMAGE_CODE / "procedure.py"),
        "test_main.py": sha256_file(IMAGE_CODE / "test_main.py"),
        "train_main.py": sha256_file(IMAGE_CODE / "train_main.py"),
        "vendored_timm_metadata": sha256_file(
            VENDORED_TIMM / "timm-0.9.2.dist-info" / "METADATA"
        ),
    }
    prelaunch_manifest = {
        "status": "prelaunch_pass",
        "model": args.model,
        "precision": args.precision,
        "study_precision_match": args.precision == "amp",
        "timm_version": timm.__version__,
        "timm_module": str(Path(timm.__file__).resolve()),
        "torch_version": torch.__version__,
        "visible_cuda_devices": torch.cuda.device_count(),
        "split_sha256": split_hashes,
        "source_sha256": source_hashes_at_start,
        "initial_model_state_sha256": state_dict_sha256(network),
        "config": config,
    }
    (output_dir / "prelaunch_manifest.json").write_text(
        json.dumps(prelaunch_manifest, indent=2, sort_keys=True) + "\n"
    )
    logging_info(logger, str(network))
    network = network.cuda()
    model = torch.nn.DataParallel(network)
    train_loader, val_loader, test_loader = get_data_loader(config, logger)
    criterion, loss_need_sigmoid, optimizer = get_loss_and_optimizer(config, network)
    if loss_need_sigmoid:
        raise RuntimeError("Study BCE-with-logits configuration changed unexpectedly")
    scheduler = CyclicLR(
        optimizer,
        base_lr=config["lr_scheduler"]["base_lr"],
        max_lr=config["lr_scheduler"]["max_lr"],
        step_size_up=config["lr_scheduler"]["step_size_up"],
        mode="exp_range",
        gamma=config["lr_scheduler"]["gamma"],
        cycle_momentum=False,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "amp")

    old_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        _, _, _, val_acc, val_auc, _ = default_test_loop(val_loader, model, "img_only", ema=None)
        _, _, _, test_acc, test_auc, _ = default_test_loop(test_loader, model, "img_only", ema=None)
        logging_info(logger, f"epoch -1, learning rate: {optimizer.param_groups[0]['lr']:.6f}, ave val batch acc: {val_acc:.6f}, ave val batch auc: {val_auc:.6f}, avg test batch acc: {test_acc:.6f}, avg test batch auc: {test_auc:.6f}")

        best_score = None
        history = []
        start = time.monotonic()
        for epoch in range(config["max_epochs"]):
            if args.precision == "amp" and args.max_train_batches is not None:
                train_metrics = amp_smoke_train_loop(
                    epoch, config, train_loader, model, criterion, optimizer, scheduler,
                    scaler, args.max_train_batches,
                )
            elif args.precision == "amp":
                train_metrics = default_train_loop(
                    scaler, epoch, config, train_loader, model, "img_only", None,
                    criterion, False, optimizer, scheduler,
                )
            else:
                train_metrics = fp32_train_loop(
                    epoch, config, train_loader, model, criterion, optimizer, scheduler,
                    args.max_train_batches,
                )
            assert_module_finite(network, f"training epoch {epoch}")
            _, _, _, val_acc, val_auc, _ = default_test_loop(val_loader, model, "img_only", ema=None)
            if args.skip_test_each_epoch:
                test_acc = test_auc = float("nan")
            else:
                _, _, _, test_acc, test_auc, _ = default_test_loop(test_loader, model, "img_only", ema=None)
            if not np.isfinite(val_auc):
                raise FloatingPointError(f"non-finite validation AUROC at epoch {epoch}")
            train_loss, train_acc, train_auc = train_metrics
            logging_info(logger, f"epoch {epoch:04d}, learning rate: {optimizer.param_groups[0]['lr']:.6f}, avg train loss: {train_loss:.6f}")
            logging_info(logger, f"avg train acc: {train_acc:.6f}, avg val acc: {val_acc:.6f}, avg test acc: {test_acc:.6f}")
            logging_info(logger, f"avg train auc: {train_auc:.6f}, avg val auc: {val_auc:.6f}, avg test auc: {test_auc:.6f}")
            torch.cuda.synchronize()
            best_score = save_checkpoint(config, logger, epoch, network, optimizer, scheduler, scaler, None, best_score, val_auc)
            history.append({"epoch": epoch, "val_auc": float(val_auc), "test_auc_monitor": float(test_auc), "elapsed_seconds": time.monotonic() - start})
            (output_dir / "epoch_metrics.json").write_text(json.dumps(history, indent=2) + "\n")

        if args.max_train_batches is not None:
            smoke_manifest = {
                "completed_smoke": True,
                "model": args.model,
                "precision": args.precision,
                "epochs": config["max_epochs"],
                "max_train_batches": args.max_train_batches,
                "visible_cuda_devices": torch.cuda.device_count(),
                "last_validation_auc": float(history[-1]["val_auc"]),
                "split_sha256": split_hashes,
                "timm_version": timm.__version__,
                "torch_version": torch.__version__,
                "source_sha256": source_hashes_at_start,
            }
            (output_dir / "smoke_manifest.json").write_text(json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n")
            print(json.dumps(smoke_manifest, indent=2, sort_keys=True), flush=True)
            return

        if args.skip_final_ensemble:
            training_manifest = {
                "completed_training": True,
                "final_ensemble_deferred": True,
                "model": args.model,
                "precision": args.precision,
                "seed": 0,
                "epochs": config["max_epochs"],
                "split_sha256": split_hashes,
                "timm_version": timm.__version__,
                "visible_cuda_devices": torch.cuda.device_count(),
                "elapsed_seconds": time.monotonic() - start,
            }
            (output_dir / "training_manifest.json").write_text(
                json.dumps(training_manifest, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(training_manifest, indent=2, sort_keys=True), flush=True)
            return

        def validation_auc_from_name(path):
            return float(path.stem.rsplit("_auc_", 1)[1])
        top_five = [Path(path) for path in get_top_checkpoint_list(str(output_dir), no_checkpoints=5)]
        if len(top_five) != 5:
            raise RuntimeError(f"Expected five ranked checkpoints, found {len(top_five)}")
        logging_info(logger, "top_ranked_checkpoint_list:" + str([str(path) for path in top_five]))

        # Match reference test_main.py: deterministic test transforms and sequential
        # ordering are used for all three stored prediction files, including the
        # training split.  Do not reuse the augmented weighted training loader.
        split_loaders = {
            split: get_dataloader_helper(config, split, logger)
            for split in ("train", "val", "test")
        }
        all_results = {split: [] for split in split_loaders}
        reference = {}
        for checkpoint_path in top_five:
            checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
            network.load_state_dict(checkpoint["network"], strict=True)
            for split, loader in split_loaders.items():
                result = default_test_loop(loader, model, "img_only", ema=None)
                ground_truth, predictions, patient_ids, _, _, features = result
                if split not in reference:
                    reference[split] = (ground_truth.copy(), patient_ids.copy())
                else:
                    if not np.array_equal(reference[split][0], ground_truth) or not np.array_equal(reference[split][1], patient_ids):
                        raise RuntimeError(f"{split} order changed during checkpoint ensemble")
                all_results[split].append((predictions, features))

        ensemble_auc = {}
        for split in split_loaders:
            predictions = np.mean(np.asarray([item[0] for item in all_results[split]]), axis=0)
            features = np.mean(np.asarray([item[1] for item in all_results[split]]), axis=0)
            ground_truth, patient_ids = reference[split]
            auc = roc_auc_score(ground_truth, predictions)
            result = (ground_truth, predictions, patient_ids, None, auc, features)
            ensemble_auc[split] = save_result(config, output_dir, f"top_checkpoint_ensemble_{split}_result.pickle", result)

        manifest = {
            "completed": True,
            "model": args.model,
            "precision": args.precision,
            "seed": 0,
            "epochs": config["max_epochs"],
            "top_five": [{"path": str(path), "validation_auc_rounded": validation_auc_from_name(path), "sha256": sha256_file(path)} for path in top_five],
            "ensemble_auc": ensemble_auc,
            "split_sha256": split_hashes,
            "timm_version": timm.__version__,
            "torch_version": torch.__version__,
            "source_sha256_at_start": source_hashes_at_start,
            "reference_code_sha256": {
                name: sha256_file(IMAGE_CODE / name)
                for name in ("dataloader.py", "procedure.py", "test_main.py", "train_main.py")
            },
            # The runner hash was captured before changing into output_dir.
            # Re-reading a relative __file__ after chdir would resolve beneath
            # output_dir and fail after all prediction work had completed.
            "runner_sha256": source_hashes_at_start["runner"],
            "elapsed_seconds": time.monotonic() - start,
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
