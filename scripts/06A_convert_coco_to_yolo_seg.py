#!/usr/bin/env python3
"""
COCO Instance Segmentation (polygon) -> YOLO-seg label (.txt) converter

ARCADE stenosis:
  - Keep only category_id == 26 (stenosis)
  - Remap to YOLO class_id == 0
  - Output format per instance polygon:
      0 x1/w y1/h x2/w y2/h ... xn/w yn/h

Why a separate yolo_seg tree?
  Detect labels are "cls cx cy w h" (5 floats).
  Seg labels are "cls x1 y1 x2 y2 ..." (variable length).
  Mixing them in the same labels/ folder will silently break training.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGET_COCO_ID = 26
YOLO_CLASS_ID = 0
MIN_POLYGON_POINTS = 3  # YOLO needs at least a triangle


def clip01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def normalize_polygon(
    flat_xy: list[float],
    img_w: int,
    img_h: int,
) -> list[float] | None:
    """
    COCO flat polygon [x1,y1,x2,y2,...] -> YOLO normalized [x1,y1,...] in [0,1].

    Returns None if the polygon is unusable after validation.
    """
    if img_w <= 0 or img_h <= 0:
        return None
    if len(flat_xy) < MIN_POLYGON_POINTS * 2:
        return None
    if len(flat_xy) % 2 != 0:
        # Odd-length coordinate list cannot form (x,y) pairs
        return None

    out: list[float] = []
    for i in range(0, len(flat_xy), 2):
        x = clip01(float(flat_xy[i]) / img_w)
        y = clip01(float(flat_xy[i + 1]) / img_h)
        out.extend([x, y])

    # Degenerate after clip (all points collapsed) -> reject
    xs = out[0::2]
    ys = out[1::2]
    if max(xs) - min(xs) <= 0.0 or max(ys) - min(ys) <= 0.0:
        return None
    return out


def polygon_to_yolo_line(norm_xy: list[float], class_id: int = YOLO_CLASS_ID) -> str:
    coords = " ".join(f"{v:.6f}" for v in norm_xy)
    return f"{class_id} {coords}"


def extract_polygons(segmentation: Any) -> list[list[float]]:
    """
    COCO segmentation variants:
      - polygon: [[x1,y1,x2,y2,...], ...]   <- ARCADE uses this
      - RLE dict: {"counts": ..., "size": ...}  <- skip (usually iscrowd=1)
    """
    if segmentation is None:
        return []
    if isinstance(segmentation, dict):
        # RLE / compressed mask — not converted here
        return []
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return []

    # Rare malformed case: flat list without outer nesting
    if segmentation and isinstance(segmentation[0], (int, float)):
        return [list(map(float, segmentation))]

    polys: list[list[float]] = []
    for poly in segmentation:
        if isinstance(poly, list) and poly and isinstance(poly[0], (int, float)):
            polys.append(list(map(float, poly)))
    return polys


def convert_coco_to_yolo_seg_labels(
    json_path: Path,
    target_coco_id: int = TARGET_COCO_ID,
    yolo_class_id: int = YOLO_CLASS_ID,
) -> dict[str, list[str]]:
    """
    Returns: {image_file_name: [yolo_seg_line, ...]}
    Empty list means background image (valid for YOLO).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_dict = {
        img["id"]: {
            "file_name": img["file_name"],
            "w": int(img["width"]),
            "h": int(img["height"]),
        }
        for img in coco["images"]
    }

    # Ensure every image gets a label file (including negatives)
    yolo_labels: dict[str, list[str]] = {
        img["file_name"]: [] for img in coco["images"]
    }

    stats = defaultdict(int)

    for ann in coco["annotations"]:
        if ann.get("category_id") != target_coco_id:
            stats["skipped_wrong_category"] += 1
            continue

        # Crowd / RLE masks are not reliable polygon sources
        if ann.get("iscrowd", 0) == 1:
            stats["skipped_iscrowd"] += 1
            continue

        image_id = ann["image_id"]
        if image_id not in img_dict:
            stats["skipped_missing_image"] += 1
            continue

        info = img_dict[image_id]
        file_name = info["file_name"]
        img_w, img_h = info["w"], info["h"]

        polys = extract_polygons(ann.get("segmentation"))
        if not polys:
            stats["skipped_empty_or_rle"] += 1
            continue

        wrote_any = False
        for poly in polys:
            norm = normalize_polygon(poly, img_w, img_h)
            if norm is None:
                stats["skipped_invalid_polygon"] += 1
                continue
            yolo_labels[file_name].append(polygon_to_yolo_line(norm, yolo_class_id))
            stats["polygons_written"] += 1
            wrote_any = True

        if wrote_any:
            stats["instances_kept"] += 1

    n_empty = sum(1 for v in yolo_labels.values() if len(v) == 0)
    print(f"[{json_path.name}]")
    print(f"  images              : {len(yolo_labels)}")
    print(f"  stenosis instances  : {stats['instances_kept']}")
    print(f"  polygons written    : {stats['polygons_written']}")
    print(f"  empty label files   : {n_empty}")
    for k in (
        "skipped_wrong_category",
        "skipped_iscrowd",
        "skipped_empty_or_rle",
        "skipped_invalid_polygon",
        "skipped_missing_image",
    ):
        if stats[k]:
            print(f"  {k}: {stats[k]}")

    return yolo_labels


def write_labels(
    yolo_labels: dict[str, list[str]],
    label_dir: Path,
    dry_run: bool = False,
) -> int:
    label_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for file_name, lines in yolo_labels.items():
        txt_path = label_dir / (Path(file_name).stem + ".txt")
        if dry_run:
            n += 1
            continue
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            # Keep trailing newline only when there is content (optional)
            if lines:
                f.write("\n")
        n += 1
    return n


def mirror_labels_to_pipelines(
    src_split_dir: Path,
    out_root: Path,
    pipelines: list[str],
    split_name: str,
    dry_run: bool = False,
) -> None:
    """
    Seg labels are geometry-only (independent of CLAHE/gamma pixels),
    so copy the same .txt into each pipeline folder.
    """
    for pipe in pipelines:
        dst = out_root / pipe / "labels" / split_name
        if dry_run:
            print(f"  [dry-run] would copy -> {dst}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for txt in src_split_dir.glob("*.txt"):
            shutil.copy2(txt, dst / txt.name)
        print(f"  copied labels -> {dst} ({len(list(dst.glob('*.txt')))} files)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert ARCADE COCO polygon segmentation to YOLO-seg labels"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "data"),
        help="ARCADE data root (contains arcade/ and will receive yolo_seg/)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root for YOLO-seg dataset (default: <data-dir>/yolo_seg)",
    )
    p.add_argument(
        "--pipelines",
        nargs="+",
        default=["original", "clahe", "clahe_gamma"],
        help="Pipeline folder names that will receive identical seg labels",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse/validate only; do not write label files",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """parse_known_args: ignore Jupyter kernel argv (-f kernel.json)."""
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(argv)
    return args


def run_conversion(
    data_dir: Path | None = None,
    out_root: Path | None = None,
    pipelines: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Notebook-safe entrypoint. Prefer this over main() inside Jupyter."""
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data"
    out_root = Path(out_root or (data_dir / "yolo_seg"))
    pipelines = pipelines or ["original", "clahe", "clahe_gamma"]

    train_json = data_dir / "arcade/arcade/stenosis/train/annotations/train.json"
    val_json = data_dir / "arcade/arcade/stenosis/val/annotations/val.json"

    print("=== Path check ===")
    print(f"DATA_DIR   : {data_dir}  exists={data_dir.exists()}")
    print(f"OUT_ROOT   : {out_root}")
    print(f"TRAIN_JSON : {train_json}  exists={train_json.exists()}")
    print(f"VAL_JSON   : {val_json}  exists={val_json.exists()}")
    print(f"PIPELINES  : {pipelines}")
    print(f"DRY_RUN    : {dry_run}")

    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(
            "COCO JSON not found. Fix data_dir or verify ARCADE stenosis paths."
        )

    staging = out_root / "_labels_shared"
    for split_name, json_path in (("train", train_json), ("val", val_json)):
        print()
        labels = convert_coco_to_yolo_seg_labels(json_path)
        split_dir = staging / split_name
        n = write_labels(labels, split_dir, dry_run=dry_run)
        print(f"  wrote/planned label files: {n} -> {split_dir}")
        mirror_labels_to_pipelines(
            split_dir, out_root, pipelines, split_name, dry_run=dry_run
        )

    print("\nDone.")
    print(
        "Next (Step 2): create data.yaml under each yolo_seg/{pipeline}/ "
        "pointing images/ + labels/, then visually spot-check a few masks."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_conversion(
        data_dir=args.data_dir,
        out_root=args.out_root,
        pipelines=args.pipelines,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    import sys

    # %run inside Jupyter injects: -f .../kernel-....json
    launched_by_jupyter = any(
        (a == "-f") or a.endswith("kernel.json") or ("ipykernel" in a)
        for a in sys.argv
    )
    if launched_by_jupyter:
        print(
            "Detected Jupyter kernel argv. Skipping CLI parse.\n"
            "Notebook에서는 main()/parse_args()/%run 대신 아래처럼 호출하세요:\n"
            "  run_conversion(dry_run=True)\n"
        )
        run_conversion(dry_run=True)
    else:
        main()
