"""Study figure renderers using aggregate tables and authorized ROC coordinates."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


ROUNDED_BOX_PAD = 0.012

FEATURE_NAMES = [
    "Sex",
    "Age",
    "Diastolic BP",
    "Systolic BP",
    "Heart rate",
    "BMI",
    "Diabetes duration",
    "Hypertension",
    "Hyperlipidemia",
    "Cardiovascular disease",
    "Smoking status",
    "Drinking status",
    "Image-predicted AS score",
]

FACTOR_NAMES = FEATURE_NAMES[:12]

def save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> List[Path]:
    outputs: List[Path] = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = dpi
            kwargs["metadata"] = {"Software": "pwv-repro deterministic figure builder"}
        else:
            kwargs["metadata"] = {
                "Creator": "pwv-repro deterministic figure builder",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    png = stem.with_suffix(".png")
    tiff = stem.with_suffix(".tiff")
    with Image.open(png) as image:
        rgb = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            rgb.paste(image, mask=image.getchannel("A"))
        else:
            rgb.paste(image.convert("RGB"))
        rgb.save(tiff, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    outputs.append(tiff)
    return outputs

def plot_figure1(metrics: pd.DataFrame, coordinates: pd.DataFrame, output: Path) -> List[Path]:
    cohorts = [
        ("development", "Development (training + validation)"),
        ("internal_test", "Internal test"),
        ("external_validation", "External validation"),
    ]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 13))
    line_styles = ["-", "--", "-.", ":"] * 4
    models = ["Image model", *FACTOR_NAMES]
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 16.2))
    for panel, (ax, (cohort, title)) in enumerate(zip(axes, cohorts)):
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        cohort_metrics = metrics.loc[metrics["cohort"] == cohort].set_index("model")
        for index, model in enumerate(models):
            curve = coordinates.loc[
                (coordinates["cohort"] == cohort) & (coordinates["model"] == model)
            ]
            row = cohort_metrics.loc[model]
            label = (
                f"{model}: {row['auc']:.3f} "
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            )
            ax.plot(
                curve["false_positive_rate"],
                curve["true_positive_rate"],
                color=colors[index],
                linewidth=2.6 if model == "Image model" else 1.45,
                linestyle="-" if model == "Image model" else line_styles[index],
                label=label,
            )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
        ax.set_title(f"{chr(65 + panel)}  {title}", loc="left", fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8.2,
            title="AUROC [95% participant-bootstrap CI]",
            title_fontsize=8.5,
        )
    fig.subplots_adjust(left=0.08, right=0.57, top=0.98, bottom=0.05, hspace=0.38)
    return save_figure(fig, output)

def _figure4_build_canvas(
    metrics: Mapping[str, Mapping[str, Any]],
) -> Tuple[plt.Figure, Sequence[plt.Axes], Mapping[str, List[Any]], pd.DataFrame]:
    """Build Figure 4 with a fixed numeric column and protected centre gap."""
    retain = [
        ("retain_artery", "Artery only"),
        ("retain_vein", "Vein only"),
        ("retain_macular_2dd", "Macula, 2-DD"),
        ("retain_optic_disc", "Optic disc"),
        ("retain_macular_1dd", "Macula, 1-DD"),
        ("retain_vessel", "Artery + vein"),
        (
            "retain_all_with_macular_2dd",
            "Vessels + optic disc + 2-DD\nmacula retained",
        ),
    ]
    remove = [
        ("remove_artery_inpaint_biharmonic", "Artery removed"),
        ("remove_vein_inpaint_biharmonic", "Vein removed"),
        ("remove_macular_2dd_inpaint_biharmonic", "Macula 2-DD removed"),
        ("remove_optic_disc_inpaint_biharmonic", "Optic disc removed"),
        ("remove_macular_1dd_inpaint_biharmonic", "Macula 1-DD removed"),
        ("remove_vessel_inpaint_biharmonic", "Artery + vein removed"),
        (
            "remove_all_with_macular_2dd_inpaint_biharmonic",
            "Vessels + optic disc +\n2-DD macula removed\n(retinal background\nretained)",
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for panel, members in [("retained", retain), ("removed", remove)]:
        for order, (key, label) in enumerate(members, start=1):
            value = metrics[key]
            rows.append(
                {
                    "panel": panel,
                    "display_order": order,
                    "analysis": key,
                    "label": label,
                    "n": value["n"],
                    "events": value["positive"],
                    "auc": value["auc"],
                    "ci_lower": value["auc_ci_lower"],
                    "ci_upper": value["auc_ci_upper"],
                    "bootstrap_resamples": value["bootstrap_resamples"],
                    "bootstrap_seed": value["bootstrap_seed"],
                }
            )
    source = pd.DataFrame(rows)
    full_auc = float(metrics["full_image"]["auc"])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.3), sharex=True)
    annotations: Dict[str, List[Any]] = {"retained": [], "removed": []}
    numeric_separator_x = 0.801
    numeric_column_x = 0.806
    for ax, panel, title, color in zip(
        axes,
        ["retained", "removed"],
        ["Retained-structure datasets", "Structure-removal datasets"],
        ["#86A9D3", "#E58B3A"],
    ):
        selected = source.loc[source["panel"] == panel].sort_values("display_order", ascending=False)
        y = np.arange(len(selected))
        x = selected["auc"].to_numpy()
        errors = np.vstack([x - selected["ci_lower"], selected["ci_upper"] - x])
        ax.errorbar(
            x,
            y,
            xerr=errors,
            fmt="o",
            color=color,
            ecolor="#333333",
            capsize=3,
            zorder=3,
        )
        # Place the reference label outside the points and confidence intervals.
        ax.axvline(
            full_auc,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        ax.set_yticks(y, selected["label"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Participant-level AUROC\n[95% bootstrap CI]")
        ax.set_xlim(0.67, 0.875)
        ax.grid(axis="x", alpha=0.2)
        ax.axvline(
            numeric_separator_x,
            color="#D5D9DE",
            linewidth=0.8,
            zorder=0,
        )
        for x_value, y_value, lower, upper in zip(
            x, y, selected["ci_lower"], selected["ci_upper"]
        ):
            annotation = ax.text(
                numeric_column_x,
                y_value,
                f"{x_value:.3f} [{lower:.3f}, {upper:.3f}]",
                ha="left",
                va="center",
                fontsize=7.5,
                zorder=5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.94,
                    "pad": 0.6,
                },
            )
            annotations[panel].append(annotation)
    fig.suptitle(
        f"Dashed vertical line denotes full-image AUROC = {full_auc:.3f}",
        y=0.975,
        fontsize=10,
    )
    # A deliberately wide inter-panel gutter gives the right panel's anatomy
    # labels their own region. Numeric values occupy a fixed column within each
    # forest axis, rather than following CI endpoints into the centre gutter.
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.86, wspace=1.40)
    return fig, axes, annotations, source

def _figure4_centre_clearance_points(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    annotations: Mapping[str, Sequence[Any]],
) -> float:
    """Return horizontal clearance between left values and right anatomy labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_value_right = max(
        artist.get_window_extent(renderer=renderer).x1
        for artist in annotations["retained"]
    )
    right_label_left = min(
        label.get_window_extent(renderer=renderer).x0
        for label in axes[1].get_yticklabels()
        if label.get_visible() and label.get_text()
    )
    pixels = right_label_left - left_value_right
    return float(pixels * 72.0 / fig.dpi)

def _figure4_enforce_layout(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    annotations: Mapping[str, Sequence[Any]],
) -> Mapping[str, float]:
    """Fail closed on centre collisions at native and 7-inch layouts."""
    original_width, original_height = fig.get_size_inches()
    widths = (float(original_width), 7.0)
    clearances: Dict[str, float] = {}
    try:
        for width in widths:
            fig.set_size_inches(
                width,
                float(original_height) * width / float(original_width),
                forward=True,
            )
            label = "native" if width == float(original_width) else "submission_7inch"
            clearance = _figure4_centre_clearance_points(fig, axes, annotations)
            clearances[label] = clearance
            if clearance < 6.0:
                raise RuntimeError(
                    "Figure 4 centre-gap clearance gate failed for "
                    f"{label}: {clearance:.3f} points < 6.000 points"
                )
    finally:
        fig.set_size_inches(original_width, original_height, forward=True)
        fig.canvas.draw()
    return clearances

def plot_figure4(metrics: Mapping[str, Mapping[str, Any]], output: Path) -> Tuple[List[Path], pd.DataFrame]:
    fig, axes, annotations, source = _figure4_build_canvas(metrics)
    _figure4_enforce_layout(fig, axes, annotations)
    return save_figure(fig, output), source

def rounded_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#F6F8FA",
    edgecolor: str = "#4C566A",
    fontsize: float = 9,
    linewidth: float = 1.3,
) -> None:
    x, y = xy
    if (
        x < ROUNDED_BOX_PAD
        or y < ROUNDED_BOX_PAD
        or x + width > 1 - ROUNDED_BOX_PAD
        or y + height > 1 - ROUNDED_BOX_PAD
    ):
        raise ValueError(
            "Rounded box plus padding would cross the normalized axes boundary: "
            f"xy={xy}, width={width}, height={height}, pad={ROUNDED_BOX_PAD}"
        )
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={ROUNDED_BOX_PAD},rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)

def arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = "#4C566A") -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color)
    )

def plot_figure6(output: Path) -> List[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 13.8))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("Stage 1. Cohorts and participant-level split", loc="left", fontweight="bold")
    rounded_box(
        ax,
        (0.02, 0.57),
        0.22,
        0.25,
        "Internal MMC cohort\n155 centers\n36,412 participants\n64,440 images",
        facecolor="#FFF2E8",
    )
    rounded_box(ax, (0.39, 0.64), 0.22, 0.18, "Training + validation\n29,081 participants\n51,497 images", facecolor="#FFF2E8")
    rounded_box(ax, (0.39, 0.34), 0.22, 0.18, "Internal test\n7,331 participants\n12,943 images", facecolor="#FFF2E8")
    rounded_box(ax, (0.75, 0.48), 0.22, 0.25, "External validation\n2 independent hospitals\n3,899 participants\n6,217 unique images", facecolor="#E8F4FF")
    arrow(ax, (0.24, 0.69), (0.39, 0.73))
    arrow(ax, (0.24, 0.66), (0.39, 0.43))
    ax.text(0.315, 0.80, "participant-level split", ha="center", fontsize=8)

    ax = axes[1]
    ax.set_title("Stage 2. Prediction pipeline", loc="left", fontweight="bold")
    rounded_box(ax, (0.02, 0.60), 0.13, 0.20, "Left fundus\n(if available)", facecolor="#E8F4FF")
    rounded_box(ax, (0.02, 0.25), 0.13, 0.20, "Right fundus\n(if available)", facecolor="#E8F4FF")
    rounded_box(ax, (0.20, 0.43), 0.16, 0.24, "Shared-weight\nTinyNet-C encoder\n(per-eye features)", facecolor="#EFE8FF")
    rounded_box(ax, (0.42, 0.43), 0.13, 0.24, "Arithmetic mean\nof available\neye features", facecolor="#EFE8FF")
    rounded_box(ax, (0.61, 0.43), 0.14, 0.24, "Lightweight\nchannel-attention\ngate + classifier", facecolor="#EFE8FF")
    rounded_box(ax, (0.81, 0.43), 0.17, 0.24, "Image-only\nAS probability", facecolor="#E7F7ED")
    for start, end in [((0.15, 0.70), (0.20, 0.57)), ((0.15, 0.35), (0.20, 0.51)), ((0.36, 0.55), (0.42, 0.55)), ((0.55, 0.55), (0.61, 0.55)), ((0.75, 0.55), (0.81, 0.55))]:
        arrow(ax, start, end)
    rounded_box(ax, (0.20, 0.04), 0.25, 0.17, "12 clinical variables\nin each of 20 MI datasets", facecolor="#FFF4D6")
    rounded_box(ax, (0.57, 0.04), 0.22, 0.17, "20 logistic fusion fits\n(image score + clinical inputs)", facecolor="#FFF4D6")
    rounded_box(ax, (0.84, 0.04), 0.14, 0.17, "Mean fusion\nprobability", facecolor="#E7F7ED")
    arrow(ax, (0.45, 0.125), (0.57, 0.125))
    arrow(ax, (0.895, 0.43), (0.72, 0.21))
    arrow(ax, (0.79, 0.125), (0.84, 0.125))

    ax = axes[2]
    ax.set_title("Stage 3. Evaluation and exploratory analyses", loc="left", fontweight="bold")
    rounded_box(ax, (0.02, 0.54), 0.19, 0.26, "Primary evaluation\nparticipant AUROC\nAP, Brier, calibration\npaired comparisons", facecolor="#E8F4FF")
    rounded_box(ax, (0.27, 0.54), 0.19, 0.26, "Robustness\nage/sex/HbA1c\ncentre and device\nexternal validation", facecolor="#E8F4FF")
    rounded_box(ax, (0.52, 0.54), 0.20, 0.26, "Retinal perturbations\nretention/removal\nretrained classifiers\nstructure-specific inputs", facecolor="#F4ECFF")
    rounded_box(ax, (0.78, 0.54), 0.20, 0.26, "Interpretability\nSHAP/attribution\nmanual vessel measures\nadjusted score–AS models", facecolor="#F4ECFF")
    rounded_box(ax, (0.27, 0.10), 0.46, 0.22, "Screening target: arterial stiffness defined as\nbrachial–ankle PWV ≥1,400 cm/s\nModel output is a screening probability, not a PWV measurement", facecolor="#FFF4D6", fontsize=10)
    fig.tight_layout(h_pad=1.2)
    return save_figure(fig, output)

def plot_supplement_s1(output: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.24, 0.95, "Internal MMC cohort (155 centers)", ha="center", fontweight="bold", fontsize=13)
    ax.text(0.77, 0.95, "External-validation cohort (2 hospitals)", ha="center", fontweight="bold", fontsize=13)
    rounded_box(ax, (0.02, 0.67), 0.23, 0.17, "Enrolled\n40,665 participants; 71,970 qualified images", facecolor="#FFF2E8")
    rounded_box(ax, (0.02, 0.37), 0.23, 0.14, "Excluded: 4,253 participants\nmissing PWV", facecolor="#FDECEC", edgecolor="#D9534F")
    rounded_box(ax, (0.31, 0.67), 0.25, 0.17, "Analysis cohort\n36,412 participants; 64,440 images", facecolor="#E8F4FF")
    # The two lower boxes are sibling children of the analysis cohort. There is
    # deliberately no development-to-test edge and no line crosses either box.
    rounded_box(ax, (0.26, 0.14), 0.18, 0.16, "Development\n(training + validation)\n29,081 participants; 51,497 images", facecolor="#F6F8FA", fontsize=8.4)
    rounded_box(ax, (0.46, 0.14), 0.18, 0.16, "Internal test\n7,331 participants; 12,943 images", facecolor="#F6F8FA")
    arrow(ax, (0.25, 0.755), (0.31, 0.755))
    arrow(ax, (0.135, 0.67), (0.135, 0.51))
    arrow(ax, (0.415, 0.67), (0.35, 0.30))
    arrow(ax, (0.455, 0.67), (0.55, 0.30))
    ax.text(0.45, 0.50, "participant-level split", ha="center", fontsize=8)
    rounded_box(ax, (0.69, 0.67), 0.28, 0.17, "Image-quality eligible\n4,221 participants; 6,753 images", facecolor="#FFF4D6")
    rounded_box(ax, (0.69, 0.37), 0.28, 0.14, "Filtering step: exclude 322 participants\nwithout PWV-linked eligible images (536 images)", facecolor="#FDECEC", edgecolor="#D9534F", fontsize=8.4)
    rounded_box(ax, (0.69, 0.08), 0.28, 0.17, "Final external validation\n3,899 participants; 6,217 unique images", facecolor="#E8F4FF")
    arrow(ax, (0.83, 0.67), (0.83, 0.51))
    arrow(ax, (0.83, 0.37), (0.83, 0.25))
    fig.tight_layout()
    return save_figure(fig, output)

def plot_supplement_s4(output: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(14.0, 5.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Image and clinical-fusion architecture", loc="left", fontweight="bold", fontsize=14)
    rounded_box(ax, (0.02, 0.58), 0.10, 0.20, "Left eye\n(optional)", facecolor="#E8F4FF")
    rounded_box(ax, (0.02, 0.22), 0.10, 0.20, "Right eye\n(optional)", facecolor="#E8F4FF")
    rounded_box(ax, (0.16, 0.39), 0.13, 0.26, "Shared TinyNet-C\nencoder\n(per-eye 2048-d)", facecolor="#EFE8FF")
    rounded_box(ax, (0.34, 0.39), 0.12, 0.26, "Arithmetic mean\nof available\neye features", facecolor="#EFE8FF")
    rounded_box(ax, (0.51, 0.39), 0.13, 0.26, "Channel-attention\ngate + linear\nimage classifier", facecolor="#EFE8FF")
    rounded_box(ax, (0.69, 0.39), 0.10, 0.26, "Image-only\nAS score", facecolor="#E7F7ED")
    rounded_box(ax, (0.69, 0.05), 0.10, 0.20, "12 clinical\ninputs", facecolor="#FFF4D6")
    rounded_box(ax, (0.84, 0.24), 0.14, 0.34, "20 MI-specific\nlogistic fusion models\nthen arithmetic mean\nof probabilities", facecolor="#FFF4D6")
    for start, end in [((0.11, 0.68), (0.16, 0.55)), ((0.11, 0.32), (0.16, 0.49)), ((0.29, 0.52), (0.34, 0.52)), ((0.46, 0.52), (0.51, 0.52)), ((0.64, 0.52), (0.69, 0.52)), ((0.79, 0.52), (0.84, 0.47)), ((0.79, 0.15), (0.84, 0.33))]:
        arrow(ax, start, end)
    ax.text(0.91, 0.12, "Final AS screening probability", ha="center", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output)

def plot_supplement_s7(economic: Mapping[str, Any], output: Path) -> Tuple[List[Path], pd.DataFrame]:
    source = pd.DataFrame(economic["operating_points"])
    source = source.loc[
        :,
        [
            "target_sensitivity",
            "achieved_sensitivity",
            "threshold",
            "threshold_rounded_for_public_release",
            "referral_fraction",
            "specificity",
            "tp",
            "fp",
            "tn",
            "fn",
            "cost_cny_no_attendance_adjustment_fundus_0",
            "break_even_fundus_cost_cny",
        ],
    ].copy()
    source["universal_pwv_cost_cny"] = float(economic["pwv_test_cost_cny"])
    source["savings_vs_universal_cny"] = (
        source["universal_pwv_cost_cny"]
        - source["cost_cny_no_attendance_adjustment_fundus_0"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = source["target_sensitivity"]
    axes[0].plot(x, source["cost_cny_no_attendance_adjustment_fundus_0"], "o-", color="#4C86C6", label="Two-stage, zero incremental fundus cost")
    axes[0].axhline(float(economic["pwv_test_cost_cny"]), color="#D9534F", linestyle="--", label="Universal PWV testing")
    axes[0].set(xlabel="Target sensitivity", ylabel="Cost per screened participant (CNY)", ylim=(75, 126))
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(alpha=0.2)
    axes[0].set_title("A  Direct screening cost", loc="left", fontweight="bold")
    for x_value, y_value in zip(x, source["cost_cny_no_attendance_adjustment_fundus_0"]):
        axes[0].text(x_value, y_value + 1.1, f"{y_value:.2f}", ha="center", fontsize=8)

    axes[1].bar(x.astype(str), source["savings_vs_universal_cny"], color="#70AD75")
    axes[1].set(xlabel="Target sensitivity", ylabel="Savings / break-even fundus cost (CNY)")
    axes[1].set_title("B  Savings relative to universal PWV", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    for index, value in enumerate(source["savings_vs_universal_cny"]):
        axes[1].text(index, value + 0.8, f"{value:.2f}", ha="center", fontsize=8)
    fig.suptitle("Direct screening cost per participant", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, output), source

def plot_supplement_s8(output: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Proposed two-stage screening pathway", loc="left", fontweight="bold", fontsize=14)
    rounded_box(ax, (0.02, 0.40), 0.16, 0.25, "Existing retinal image\nwhere routinely acquired", facecolor="#E8F4FF")
    rounded_box(ax, (0.23, 0.40), 0.16, 0.25, "Locally validated and\ncalibrated fusion model", facecolor="#EFE8FF")
    rounded_box(ax, (0.44, 0.40), 0.14, 0.25, "Locally selected\ndecision threshold", facecolor="#FFF4D6")
    rounded_box(ax, (0.64, 0.63), 0.16, 0.21, "Screen positive\nrefer for PWV", facecolor="#FDECEC", edgecolor="#D9534F")
    rounded_box(ax, (0.64, 0.20), 0.16, 0.21, "Screen negative\nroutine clinical follow-up", facecolor="#E7F7ED")
    rounded_box(ax, (0.85, 0.40), 0.13, 0.25, "Prospective monitoring\nof safety, workflow,\nand calibration drift", facecolor="#F6F8FA")
    arrow(ax, (0.18, 0.525), (0.23, 0.525))
    arrow(ax, (0.39, 0.525), (0.44, 0.525))
    arrow(ax, (0.58, 0.56), (0.64, 0.71))
    arrow(ax, (0.58, 0.49), (0.64, 0.30))
    arrow(ax, (0.80, 0.72), (0.85, 0.57))
    arrow(ax, (0.80, 0.30), (0.85, 0.48))
    ax.text(
        0.50,
        0.08,
        "Referral thresholds and follow-up can be adapted to local clinical practice.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    return save_figure(fig, output)
