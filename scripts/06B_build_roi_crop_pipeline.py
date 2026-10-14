"""
Path C: ROI-cluster crop pipeline builder.

GT 병변(YOLO-seg 폴리곤) 중 서로 가까이 붙어있는 것들을 하나의 "클러스터"로 묶고,
그 클러스터 주변을 패딩 포함해서 크롭 -> 640x640으로 리사이즈 -> 라벨을 크롭 좌표계로
재정규화해서 새로운 학습 파이프라인(roi_crop)을 만든다.

사용 예 (노트북에서 importlib로 로드 후):
    build_roi_crop_split(
        src_img_dir=".../original/images/train",
        src_label_dir=".../original/labels/train",
        out_img_dir=".../roi_crop/images/train",
        out_label_dir=".../roi_crop/labels/train",
        dry_run=False,
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ----------------------------- 라벨 IO -----------------------------

def load_yolo_seg_labels(label_path: Path, W: int, H: int):
    """YOLO-seg .txt -> [(class_id, poly_xy(np.ndarray[N,2] in pixel coords))]"""
    items = []
    if not label_path.exists():
        return items
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = np.array(coords[0::2]) * W
        ys = np.array(coords[1::2]) * H
        poly = np.stack([xs, ys], axis=1)
        items.append((class_id, poly))
    return items


def poly_bbox(poly: np.ndarray):
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    return x1, y1, x2, y2


# ----------------------------- 클러스터링 -----------------------------

def boxes_close(b1, b2, margin_ratio: float, min_margin_px: float) -> bool:
    """두 bbox를 각각 margin만큼 확장했을 때 겹치면 '가깝다'고 판단."""
    def expand(b):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        mx = max(w * margin_ratio, min_margin_px)
        my = max(h * margin_ratio, min_margin_px)
        return x1 - mx, y1 - my, x2 + mx, y2 + my

    ex1, ey1, ex2, ey2 = expand(b1)
    fx1, fy1, fx2, fy2 = expand(b2)
    return not (ex2 < fx1 or fx2 < ex1 or ey2 < fy1 or fy2 < ey1)


def box_gap(b1, b2) -> float:
    """두 bbox 사이의 최단 edge-to-edge 거리 (겹치면 0)."""
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def cluster_lesions(boxes, margin_ratio=0.4, min_margin_px=20.0, max_extent_px=200.0):
    """근접 쌍부터 순서대로 병합하되, 병합 결과 bbox가 max_extent_px를 넘으면 스킵.
    (single-linkage 체인 병합으로 클러스터가 원본 이미지 전체로 커지는 것을 방지)
    boxes: list[(x1,y1,x2,y2)] -> list[list[idx]]
    """
    n = len(boxes)
    parent = list(range(n))
    cluster_bbox = {i: boxes[i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def merged_bbox(b1, b2):
        return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))

    def extent_ok(b):
        return (b[2] - b[0]) <= max_extent_px and (b[3] - b[1]) <= max_extent_px

    candidate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_close(boxes[i], boxes[j], margin_ratio, min_margin_px):
                candidate_pairs.append((box_gap(boxes[i], boxes[j]), i, j))
    candidate_pairs.sort(key=lambda t: t[0])

    for _gap, i, j in candidate_pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        new_bbox = merged_bbox(cluster_bbox[ri], cluster_bbox[rj])
        if extent_ok(new_bbox):
            parent[ri] = rj
            cluster_bbox[rj] = new_bbox
        # extent 초과 시 병합 스킵 -> 두 클러스터는 분리 유지

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ----------------------------- 크롭 + 라벨 재정규화 -----------------------------

def compute_crop_region(cluster_boxes, W, H, pad_ratio=0.5, min_crop_px=160):
    xs1 = min(b[0] for b in cluster_boxes)
    ys1 = min(b[1] for b in cluster_boxes)
    xs2 = max(b[2] for b in cluster_boxes)
    ys2 = max(b[3] for b in cluster_boxes)
    w, h = xs2 - xs1, ys2 - ys1
    pad_x = max(w * pad_ratio, (min_crop_px - w) / 2 if w < min_crop_px else 0)
    pad_y = max(h * pad_ratio, (min_crop_px - h) / 2 if h < min_crop_px else 0)
    cx1, cy1, cx2, cy2 = xs1 - pad_x, ys1 - pad_y, xs2 + pad_x, ys2 + pad_y

    # 정사각형으로 맞춰주기 (짧은 축을 늘림) - YOLO 정사각 리사이즈시 왜곡 최소화
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw > ch:
        diff = cw - ch
        cy1 -= diff / 2
        cy2 += diff / 2
    elif ch > cw:
        diff = ch - cw
        cx1 -= diff / 2
        cx2 += diff / 2

    # 이미지 경계로 클램프 (경계에서 잘리면 정사각이 깨질 수 있음 -> 허용)
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(W, cx2), min(H, cy2)
    return cx1, cy1, cx2, cy2


def remap_labels_to_crop(all_items, crop_region, min_visible_ratio=0.5):
    """crop_region=(cx1,cy1,cx2,cy2) pixel coords. 반환: [(class_id, poly_norm[N,2] in 0..1)]"""
    cx1, cy1, cx2, cy2 = crop_region
    cw, ch = cx2 - cx1, cy2 - cy1
    out = []
    for class_id, poly in all_items:
        x1, y1, x2, y2 = poly_bbox(poly)
        # bbox 영역 기준 crop과의 교집합 비율로 "충분히 보이는지" 판단
        ix1, iy1 = max(x1, cx1), max(y1, cy1)
        ix2, iy2 = min(x2, cx2), min(y2, cy2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter_area = iw * ih
        orig_area = max((x2 - x1) * (y2 - y1), 1e-6)
        if inter_area / orig_area < min_visible_ratio:
            continue
        # 폴리곤 좌표를 crop 영역으로 clamp (경계 밖 점만 clamp, 나머지는 그대로)
        clamped = poly.copy()
        clamped[:, 0] = np.clip(clamped[:, 0], cx1, cx2)
        clamped[:, 1] = np.clip(clamped[:, 1], cy1, cy2)
        # crop-local 좌표로 이동 후 정규화
        norm = np.empty_like(clamped)
        norm[:, 0] = (clamped[:, 0] - cx1) / cw
        norm[:, 1] = (clamped[:, 1] - cy1) / ch
        norm = np.clip(norm, 0.0, 1.0)
        # 면적이 0에 가까우면(클램프로 눌린 경우) 스킵
        area = 0.5 * abs(
            np.dot(norm[:, 0], np.roll(norm[:, 1], -1)) - np.dot(norm[:, 1], np.roll(norm[:, 0], -1))
        )
        if area < 1e-6:
            continue
        out.append((class_id, norm))
    return out


def write_yolo_seg_labels(items, out_path: Path):
    lines = []
    for class_id, norm in items:
        coords = " ".join(f"{v:.6f}" for xy in norm for v in xy)
        lines.append(f"{class_id} {coords}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ----------------------------- 메인 빌드 루프 -----------------------------

def build_roi_crop_split(
    src_img_dir: Path,
    src_label_dir: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    target_size: int = 640,
    margin_ratio: float = 0.4,
    min_margin_px: float = 20.0,
    max_extent_px: float = 150.0,
    pad_ratio: float = 0.3,
    min_crop_px: int = 160,
    min_visible_ratio: float = 0.5,
    dry_run: bool = False,
):
    src_img_dir, src_label_dir = Path(src_img_dir), Path(src_label_dir)
    out_img_dir, out_label_dir = Path(out_img_dir), Path(out_label_dir)
    if not dry_run:
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

    n_crops, n_skipped_empty, n_skipped_no_gt = 0, 0, 0

    for label_path in sorted(src_label_dir.glob("*.txt")):
        img_path = src_img_dir / (label_path.stem + ".png")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        all_items = load_yolo_seg_labels(label_path, W, H)
        if not all_items:
            n_skipped_no_gt += 1
            continue  # 배경(negative) 이미지는 roi_crop 대상 아님 (원본 파이프라인에 이미 포함됨)

        boxes = [poly_bbox(poly) for _, poly in all_items]
        clusters = cluster_lesions(
            boxes, margin_ratio=margin_ratio, min_margin_px=min_margin_px, max_extent_px=max_extent_px
        )

        for ci, idxs in enumerate(clusters):
            cluster_boxes = [boxes[i] for i in idxs]
            crop_region = compute_crop_region(
                cluster_boxes, W, H, pad_ratio=pad_ratio, min_crop_px=min_crop_px
            )
            cx1, cy1, cx2, cy2 = [int(round(v)) for v in crop_region]
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                n_skipped_empty += 1
                continue

            remapped = remap_labels_to_crop(
                all_items, (cx1, cy1, cx2, cy2), min_visible_ratio=min_visible_ratio
            )
            if not remapped:
                n_skipped_empty += 1
                continue

            crop_img = img[cy1:cy2, cx1:cx2]
            crop_resized = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            out_name = f"{label_path.stem}_c{ci}"
            if not dry_run:
                cv2.imwrite(str(out_img_dir / f"{out_name}.png"), crop_resized)
                write_yolo_seg_labels(remapped, out_label_dir / f"{out_name}.txt")
            n_crops += 1

    print(
        f"[{src_img_dir.name}] 생성된 크롭: {n_crops}, "
        f"빈크롭 스킵: {n_skipped_empty}, GT없어 스킵: {n_skipped_no_gt}"
    )
    return n_crops


# ----------------------------- CLI / 노트북 엔트리 -----------------------------

def run_build(
    yolo_seg_dir: str,
    src_pipeline: str = "original",
    out_pipeline: str = "roi_crop",
    splits=("train", "val"),
    **kwargs,
):
    yolo_seg_dir = Path(yolo_seg_dir)
    for split in splits:
        build_roi_crop_split(
            src_img_dir=yolo_seg_dir / src_pipeline / "images" / split,
            src_label_dir=yolo_seg_dir / src_pipeline / "labels" / split,
            out_img_dir=yolo_seg_dir / out_pipeline / "images" / split,
            out_label_dir=yolo_seg_dir / out_pipeline / "labels" / split,
            dry_run=kwargs.get("dry_run", False),
            **{k: v for k, v in kwargs.items() if k != "dry_run"},
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build ROI-cluster crop pipeline (Path C)")
    parser.add_argument("--yolo-seg-dir", required=True)
    parser.add_argument("--src-pipeline", default="original")
    parser.add_argument("--out-pipeline", default="roi_crop")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--dry-run", action="store_true")
    args, _unknown = parser.parse_known_args(argv)

    run_build(
        yolo_seg_dir=args.yolo_seg_dir,
        src_pipeline=args.src_pipeline,
        out_pipeline=args.out_pipeline,
        splits=args.splits,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
