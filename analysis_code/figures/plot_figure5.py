#!/usr/bin/env python3
"""Render the study Figure 5 table without refitting any model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


SCORES = [
    ("retain_vessel", "Vessel-only"),
    ("retain_artery", "Artery-only"),
    ("retain_vein", "Vein-only"),
    ("retain_disc", "Optic disc-only"),
    ("retain_macular_1dd", "Macula-only (1 disc diameter)"),
    ("retain_macular_2dd", "Macula-only (2 disc diameters)"),
    ("retain_all_with_macular_2dd", "Retained structures (incl. 2-DD macula)"),
    ("remove_vessel_inpaint_biharmonic", "Vessels removed"),
    ("remove_artery_inpaint_biharmonic", "Arteries removed"),
    ("remove_vein_inpaint_biharmonic", "Veins removed"),
    ("remove_disc_inpaint_biharmonic", "Optic disc removed"),
    ("remove_macular_1dd_inpaint_biharmonic", "1-DD macula removed"),
    ("remove_macular_2dd_inpaint_biharmonic", "2-DD macula removed"),
    (
        "remove_all_with_macular_2dd_inpaint_biharmonic",
        "Vessels, optic disc, and 2-DD macula removed (retinal background)",
    ),
    ("full_image", "Full image"),
]
MODELS = ["model0", "model1", "model2"]
MODEL_SHORT = {"model0": "Model 0", "model1": "Model 1", "model2": "Model 2"}
MODEL_DEFINITIONS = {
    "model0": "score only",
    "model1": "score + age + sex",
    "model2": (
        "score + age + sex + diabetes duration + UACR + "
        "smoking + drinking + BMI + systolic BP + HbA1c + coronary heart disease"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    expected = [(score, model) for score, _ in SCORES for model in MODELS]
    observed = [(row["score"], row["model"]) for row in rows]
    if observed != expected:
        raise RuntimeError("Unexpected score/model row order")
    if len(rows) != 45:
        raise RuntimeError("Expected 45 Figure 5 rows")
    for row in rows:
        if row["n"] != "7331" or row["events"] != "5158":
            raise RuntimeError("Unexpected cohort or event count")
        estimate = float(row["odds_ratio"])
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        if not (np.isfinite([estimate, lower, upper]).all() and 0 < lower <= estimate <= upper):
            raise RuntimeError("Invalid odds-ratio confidence interval")
        if not (0 <= float(row["p_value"]) <= float(row["q_value"]) <= 1):
            raise RuntimeError("Invalid P or adjusted P value")
    return rows, fieldnames


def write_source_table(rows, original_fields, destination: Path) -> None:
    labels = dict(SCORES)
    added = [
        "display_order",
        "score_label",
        "model_label",
        "model_definition",
        "effect_scale",
        "primary_missing_data_method",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=added + original_fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "display_order": index,
                    "score_label": labels[row["score"]],
                    "model_label": MODEL_SHORT[row["model"]],
                    "model_definition": MODEL_DEFINITIONS[row["model"]],
                    "effect_scale": "odds ratio per 1-SD higher score",
                    "primary_missing_data_method": "existing completed clinical values followed by mean substitution",
                    **row,
                }
            )


def format_or(row: dict) -> str:
    return f"{float(row['odds_ratio']):.2f} ({float(row['ci_lower']):.2f}–{float(row['ci_upper']):.2f})"


def format_q(value: str) -> str:
    number = float(value)
    return f"{number:.2e}" if number < 0.001 else f"{number:.3f}"


def render(rows, output_dir: Path) -> list[Path]:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "svg.hashsalt": "retinal-structure-associations",
        }
    )
    row_map = {(row["score"], row["model"]): row for row in rows}
    labels = [textwrap.fill(label, width=38) for _, label in SCORES]
    y_base = np.arange(len(SCORES) - 1, -1, -1, dtype=float)
    colors = {"model0": "#0072B2", "model1": "#E69F00", "model2": "#009E73"}
    markers = {"model0": "o", "model1": "s", "model2": "D"}
    offsets = {"model0": 0.22, "model1": 0.0, "model2": -0.22}

    fig, (forest, table) = plt.subplots(
        1,
        2,
        figsize=(15.2, 10.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.08, 1.32], "wspace": 0.03},
    )
    for index, y in enumerate(y_base):
        if index % 2 == 0:
            forest.axhspan(y - 0.48, y + 0.48, color="#F3F4F6", zorder=0)
            table.axhspan(y - 0.48, y + 0.48, color="#F3F4F6", zorder=0)

    for model in MODELS:
        estimates, lows, highs = [], [], []
        for score, _ in SCORES:
            row = row_map[(score, model)]
            estimates.append(float(row["odds_ratio"]))
            lows.append(float(row["ci_lower"]))
            highs.append(float(row["ci_upper"]))
        estimates = np.asarray(estimates)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        forest.errorbar(
            estimates,
            y_base + offsets[model],
            xerr=np.vstack((estimates - lows, highs - estimates)),
            fmt=markers[model],
            markersize=4.6,
            markeredgewidth=0.6,
            capsize=2.2,
            elinewidth=1.15,
            color=colors[model],
            label=MODEL_SHORT[model],
            zorder=3,
        )

    forest.axvline(1.0, color="#4B5563", linewidth=1.0, linestyle="--", zorder=1)
    forest.set_xscale("log")
    forest.set_xlim(0.92, 3.35)
    ticks = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    forest.set_xticks(ticks)
    forest.set_xticklabels(["1.0", "1.25", "1.5", "2.0", "2.5", "3.0"])
    forest.set_yticks(y_base)
    forest.set_yticklabels(labels, fontsize=8.4)
    forest.set_ylim(-0.75, len(SCORES) - 0.25)
    forest.set_xlabel("Odds ratio (95% CI), per 1-SD higher score")
    forest.grid(axis="x", color="#D1D5DB", linewidth=0.55, alpha=0.85)
    forest.tick_params(axis="y", length=0, pad=6)
    forest.spines["top"].set_visible(False)
    forest.spines["right"].set_visible(False)

    table.set_xlim(0, 1)
    table.set_ylim(forest.get_ylim())
    table.set_xticks([])
    table.tick_params(axis="y", left=False, labelleft=False)
    for spine in table.spines.values():
        spine.set_visible(False)
    transform = blended_transform_factory(table.transAxes, table.transData)
    columns = {"model0": 0.02, "model1": 0.31, "model2": 0.60}
    for score_index, (score, _) in enumerate(SCORES):
        y = y_base[score_index]
        for model, x in columns.items():
            table.text(
                x,
                y,
                format_or(row_map[(score, model)]),
                transform=transform,
                ha="left",
                va="center",
                fontsize=7.6,
                color="#111827",
            )
        table.text(
            0.91,
            y,
            format_q(row_map[(score, "model2")]["q_value"]),
            transform=transform,
            ha="left",
            va="center",
            fontsize=7.4,
            color="#111827",
        )
    header_y = len(SCORES) - 0.16
    for model, x in columns.items():
        table.text(
            x,
            header_y,
            f"{MODEL_SHORT[model]} OR (95% CI)",
            transform=transform,
            ha="left",
            va="bottom",
            fontsize=8.1,
            fontweight="bold",
        )
    table.text(
        0.91,
        header_y,
        "Model 2 q",
        transform=transform,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="bold",
    )

    handles, legend_labels = forest.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.923),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.suptitle(
        "Association of retinal structure scores with arterial stiffness",
        x=0.50,
        y=0.983,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.950,
        "Odds ratios per 1-SD higher score; N = 7,331 (5,158 events)",
        ha="center",
        va="center",
        fontsize=10,
    )
    footnote = (
        "Model 0: score only.  Model 1: score, age, and sex.  Model 2: Model 1 plus diabetes duration, "
        "UACR, smoking, drinking, BMI, systolic blood pressure, HbA1c, and coronary heart disease.\n"
        "Each score was standardized once in the full test cohort and fitted in a separate logistic model. "
        "Continuous covariates used existing completed values and mean substitution.\n"
        "q values are Benjamini–Hochberg adjusted across the 15 score tests within each model. "
        "Outcome: baPWV ≥1,400 cm/s. DD, disc diameter; CI, confidence interval."
    )
    fig.text(0.035, 0.025, footnote, ha="left", va="bottom", fontsize=7.7, linespacing=1.35)
    fig.subplots_adjust(left=0.255, right=0.985, top=0.89, bottom=0.125)

    destinations = [
        output_dir / "figure5_forest.svg",
        output_dir / "figure5_forest.pdf",
        output_dir / "figure5_forest.png",
    ]
    fig.savefig(destinations[0], metadata={"Date": None})
    fig.savefig(destinations[1], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destinations[2], dpi=300, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return destinations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Publication Figure 5 output directory must be empty: {output_dir}")

    rows, fields = load_frozen_rows(input_path)
    source_path = output_dir / "figure5_forest_source_data.csv"
    write_source_table(rows, fields, source_path)
    figures = render(rows, output_dir)
    script_path = Path(__file__).resolve()
    manifest = {
        "protocol": "retinal_structure_associations",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_model_refit": True,
        "input": {"basename": input_path.name, "sha256": sha256_file(input_path)},
        "script": {"basename": script_path.name, "sha256": sha256_file(script_path)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        "row_count": len(rows),
        "order": [score for score, _ in SCORES],
        "source_data_preserves_all_input_fields_as_strings": True,
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [source_path, *figures]
        },
    }
    manifest_path = output_dir / "figure5_forest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
