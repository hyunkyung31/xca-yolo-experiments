#!/usr/bin/env python3
"""Replot saved aggregate CSVs; never train a model or read medical images.

Run from any directory:
    python scripts/plot_experiment_results.py
    python scripts/plot_experiment_results.py --output-dir /tmp/xca-plots

Only the three generated PNG names and visualization_manifest.json are written.
The five originally extracted figures and every input CSV remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path

# Keep the headless plotting cache outside the checkout and avoid a read-only
# user config directory. Respect a caller's existing Matplotlib configuration.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xca-yolo-matplotlib"))
if "XDG_CACHE_HOME" not in os.environ:
    font_cache = Path(tempfile.gettempdir()) / "xca-yolo-font-cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(font_cache)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


COLORS = {"Original": "#526780", "CLAHE": "#AD815C", "CLAHE+Gamma": "#4F8473"}
ORDER = tuple(COLORS)
QUALITY_CSV = "03_quality_metrics__cell_007_output_03.csv"
YOLO_CSV = "yolo_best_checkpoint_validation.csv"
HISTORY_FILES = (
    ("Original", 30, "05B_cell_008__frcnn_resnet50_fpn__original_history.csv"),
    ("CLAHE+Gamma", 30, "05B_cell_008__frcnn_resnet50_fpn__clahe_gamma_history.csv"),
    ("CLAHE+Gamma", 100, "05B_cell_007__single_resnet50_fpn__clahe_gamma_history.csv"),
)
BBOX_FIGURE = "results/figures/01_dataset__cell_023_output_01.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path.name}")
    return rows


def number(row: dict[str, str], key: str, ratio: bool = False) -> float:
    value = float(row[key])
    if not math.isfinite(value) or value < 0 or (ratio and value > 1):
        raise ValueError(f"Invalid {key}: {row[key]}")
    return value


def unique_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    selected = [row for row in rows if row[key] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one row for {key}={value}; found {len(selected)}")
    return selected[0]


def quality_data(tables: Path) -> list[dict]:
    rows = read_csv(tables / QUALITY_CSV)
    keys = ("original", "01_clahe", "04_clahe_gamma")
    result = []
    for label, key in zip(ORDER, keys):
        row = unique_row(rows, "pipeline", key)
        result.append({"label": label, "pipeline": key,
                       "roi_contrast": number(row, "roi_contrast"),
                       "roi_edge_density": number(row, "roi_edge_density", ratio=True)})
    return result


def yolo_data(tables: Path) -> list[dict]:
    rows = read_csv(tables / YOLO_CSV)
    keys = ("yolo11n_original", "yolo11n_clahe", "yolo11n_clahe_gamma")
    result = []
    for label, key in zip(ORDER, keys):
        row = unique_row(rows, "experiment", key)
        # The other 50-epoch rows are different runs, even on the same pipeline.
        if (row["epochs_recorded"] != "100"
                or row["source_notebook"] != "04_train_yolo(cursor).ipynb.ipynb"
                or row["selection"] != "final best.pt validation (rounded saved log)"):
            raise ValueError(f"Unexpected validation source for {key}")
        result.append({"label": label, "experiment": key,
                       "epochs": int(row["epochs_recorded"]),
                       "images": int(row["images"]), "instances": int(row["instances"]),
                       "mAP50": number(row, "mAP50", ratio=True),
                       "mAP50-95": number(row, "mAP50-95", ratio=True)})
    if len({(r["images"], r["instances"]) for r in result}) != 1:
        raise ValueError("Validation population differs between runs")
    return result


def history_data(tables: Path) -> list[dict]:
    summaries = read_csv(tables / "frcnn_history_summary.csv")
    result = []
    for label, expected_epochs, filename in HISTORY_FILES:
        rows = read_csv(tables / filename)
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != list(range(1, expected_epochs + 1)):
            raise ValueError(f"Missing, duplicate or out-of-order epochs: {filename}")
        scores = [number(row, "AP50", ratio=True) for row in rows]
        populations = {int(row["n_gt"]) for row in rows}
        if len(populations) != 1:
            raise ValueError(f"Validation population changes: {filename}")
        # max() chooses the earliest occurrence of a tied maximum, as the
        # existing frcnn_history_summary.csv does. Check against that source.
        best_index = max(range(len(scores)), key=scores.__getitem__)
        summary = unique_row(summaries, "history_file", "results/tables/" + filename)
        actual = (len(rows), epochs[best_index], scores[best_index], scores[-1])
        recorded = (int(summary["epochs_recorded"]), int(summary["best_epoch"]),
                    number(summary, "best_AP50", ratio=True),
                    number(summary, "last_AP50", ratio=True))
        if actual != recorded:
            raise ValueError(f"History and saved summary disagree: {filename}")
        result.append({"label": label, "epochs_recorded": len(rows),
                       "source": "results/tables/" + filename,
                       "epochs": epochs, "AP50": scores,
                       "best_epoch": epochs[best_index], "best_AP50": scores[best_index],
                       "last_AP50": scores[-1], "instances": populations.pop()})
    return result


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.edgecolor": "#CBD2D9", "axes.labelcolor": "#293747",
        "text.color": "#293747", "xtick.color": "#465466", "ytick.color": "#465466",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#E5E9ED", "grid.linewidth": 0.7,
    })


def clean_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.tick_params(length=0, pad=8)


def bars(ax: plt.Axes, labels: list[str], values: list[float], fmt: str) -> None:
    rects = ax.bar(labels, values, color=[COLORS[label] for label in labels], width=0.55)
    ax.bar_label(rects, labels=[format(value, fmt) for value in values], padding=6, fontsize=11)
    ax.set_ylim(0, max(values) * 1.27)
    clean_axes(ax)


def quality_figure(data: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    labels = [row["label"] for row in data]
    bars(axes[0], labels, [row["roi_contrast"] for row in data], ".2f")
    axes[0].set_title("ROI contrast", pad=16)
    axes[0].set_ylabel("Intensity standard deviation (gray levels)")
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    bars(axes[1], labels, [row["roi_edge_density"] * 100 for row in data], ".2f")
    axes[1].set_title("ROI edge density", pad=16)
    axes[1].set_ylabel("Canny edge pixels / ROI pixels (%)")
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("Preprocessing: lesion ROI quality", fontsize=17, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.895, "Training-split means · image quality metrics, not detection AP",
             ha="center", fontsize=10, color="#586675")
    fig.text(0.5, 0.10, "Contrast: 8-bit intensity std · edges: Canny thresholds 50/150 · higher can include noise",
             ha="center", fontsize=9, color="#586675")
    fig.text(0.5, 0.045, "Source: results/tables/" + QUALITY_CSV + " · saved outputs replotted",
             ha="center", fontsize=8.5, color="#687786")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.77, wspace=0.35)
    return fig


def yolo_figure(data: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    labels = [row["label"] for row in data]
    for ax, metric in zip(axes, ("mAP50", "mAP50-95")):
        bars(ax, labels, [row[metric] for row in data], ".3f")
        ax.set_title(metric, pad=16)
        ax.set_ylabel("Average precision (ratio)")
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("YOLO11n: final best-checkpoint validation", fontsize=17, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.895,
             f"{data[0]['epochs']} epochs each · validation: {data[0]['images']} images / {data[0]['instances']} instances",
             ha="center", fontsize=10, color="#586675")
    fig.text(0.5, 0.10, "Zero-based axes · different panel scales · rounded final validation logs, not best-epoch CSV metrics",
             ha="center", fontsize=9, color="#586675")
    fig.text(0.5, 0.045, "Source: results/tables/" + YOLO_CSV + " · saved outputs replotted",
             ha="center", fontsize=8.5, color="#687786")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.77, wspace=0.3)
    return fig


def history_figure(data: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.8))
    upper = math.ceil(max(row["best_AP50"] for row in data) * 1.15 / 0.05) * 0.05
    for ax, length in zip(axes, (30, 100)):
        for row in (row for row in data if row["epochs_recorded"] == length):
            color = COLORS[row["label"]]
            ax.plot(row["epochs"], row["AP50"], color=color, lw=2, label=row["label"])
            ax.scatter(row["best_epoch"], row["best_AP50"], marker="D", s=45,
                       color=color, edgecolor="white", linewidth=0.8, zorder=4)
            ax.scatter(row["epochs"][-1], row["last_AP50"], marker="o", s=45,
                       color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_title(f"{length} epochs · " + ("input comparison" if length == 30 else "separate single run"), pad=16)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("VOC 11-point AP50 (ratio)")
        ax.set_ylim(0, upper)
        ax.set_xlim(0, length + 1)
        ax.xaxis.set_major_locator(MultipleLocator(5 if length == 30 else 20))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        clean_axes(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.suptitle("Faster R-CNN: AP50 across epochs", fontsize=17, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.915, "ResNet50-FPN · validation IoU 0.50 · score threshold 0.05 · no smoothing",
             ha="center", fontsize=10, color="#586675")
    table_ax = fig.add_axes((0.08, 0.15, 0.86, 0.18))
    table_ax.axis("off")
    cells = [[f"{row['label']} / {row['epochs_recorded']} epochs", f"{row['best_AP50']:.6f}",
              str(row["best_epoch"]), f"{row['last_AP50']:.6f}"] for row in data]
    table = table_ax.table(cellText=cells, colLabels=["Run", "Best AP50", "Best epoch", "Last AP50"],
                           cellLoc="center", colLoc="center", bbox=(0, 0, 1, 1),
                           colWidths=[0.45, 0.20, 0.15, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E9ED")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#F1F4F6")
            cell.set_text_props(weight="semibold")
        elif col_idx == 0:
            cell.set_text_props(color=COLORS[data[row_idx - 1]["label"]])
    fig.text(0.5, 0.095, "Diamond: earliest maximum · circle: last epoch · 30/100-epoch runs are separate experiments",
             ha="center", fontsize=9, color="#586675")
    fig.text(0.5, 0.045, "Source: results/tables/05B_cell_007 / 008 history CSVs · VOC AP50 is not Ultralytics mAP50",
             ha="center", fontsize=8.5, color="#687786")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.43, top=0.82, wspace=0.3)
    return fig


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(repo: Path, output: Path, dpi: int = 240) -> dict:
    if dpi < 100:
        raise ValueError("Use dpi >= 100 for readable figures")
    tables = repo / "results/tables"
    configure_style()
    # Load and validate all evidence before writing any figures.
    quality, yolo, histories = quality_data(tables), yolo_data(tables), history_data(tables)
    specs = [
        ("preprocessing_roi_quality.png", quality_figure, quality, [QUALITY_CSV],
         {"roi_contrast": "8-bit intensity standard deviation, gray levels",
          "roi_edge_density": "ratio; displayed as percentage (x100)"}),
        ("yolo11n_final_validation.png", yolo_figure, yolo, [YOLO_CSV],
         {"mAP50": "Ultralytics AP ratio at IoU 0.50",
          "mAP50-95": "Ultralytics AP ratio averaged over IoU 0.50:0.95"}),
        ("frcnn_ap50_history.png", history_figure, histories,
         [row[2] for row in HISTORY_FILES] + ["frcnn_history_summary.csv"],
         {"AP50": "VOC 11-point AP ratio, IoU 0.50, score >= 0.05"}),
    ]
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"purpose": "Revisualization of saved historical aggregate results; no model execution",
                "colors": COLORS, "dpi": dpi,
                "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
                "reused_bbox": {"path": BBOX_FIGURE, "sha256": file_hash(repo / BBOX_FIGURE),
                                "source": "01_dataset.ipynb cell 23 saved histogram",
                                "raw_bbox_data_available": False, "regenerated": False},
                "figures": []}
    for filename, builder, data, inputs, units in specs:
        fig = builder(data)
        path = output / filename
        try:
            fig.savefig(path, dpi=dpi, metadata={"Software": "scripts/plot_experiment_results.py"})
        finally:
            plt.close(fig)
        manifest["figures"].append({"filename": filename, "sha256": file_hash(path),
            "inputs": [{"path": "results/tables/" + name, "sha256": file_hash(tables / name)} for name in inputs],
            "units": units, "plotted_data": data})
    (output / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Default: <repo-root>/results/figures")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = generate(root, args.output_dir or root / "results/figures", args.dpi)
    for figure in manifest["figures"]:
        print("Generated:", figure["filename"])
    print("Reused original bbox histogram; wrote visualization_manifest.json")


if __name__ == "__main__":
    main()
