"""
Path C: 추론 시점 2단계 ROI-crop 캐스케이드.

Stage1(전체 이미지 검출) -> 예측 박스 클러스터링 -> 클러스터 주변 크롭
-> Stage2(크롭 재검출, roi_crop 도메인 특화 모델) -> 좌표 역변환 -> Stage1과 병합(NMS)

06B_build_roi_crop_pipeline.py의 cluster_lesions / compute_crop_region을 재사용한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    t_boxes = torch.tensor(boxes, dtype=torch.float32)
    t_scores = torch.tensor(scores, dtype=torch.float32)
    keep = tv_nms(t_boxes, t_scores, iou_thresh)
    return keep.numpy()


def cascade_predict(
    img_path,
    model_stage1,
    model_stage2,
    cluster_lesions_fn,
    compute_crop_region_fn,
    conf1: float = 0.35,
    conf2: float = 0.25,
    margin_ratio: float = 0.4,
    min_margin_px: float = 20.0,
    max_extent_px: float = 150.0,
    pad_ratio: float = 0.3,
    min_crop_px: int = 160,
    merge_iou: float = 0.5,
    imgsz: int = 640,
):
    """
    반환: final_boxes (N,4) xyxy in original image pixel coords, final_scores (N,)
    """
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    res1 = model_stage1.predict(source=str(img_path), conf=conf1, imgsz=imgsz, verbose=False)[0]
    if res1.boxes is None or len(res1.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), 0

    boxes1 = res1.boxes.xyxy.cpu().numpy()
    scores1 = res1.boxes.conf.cpu().numpy()

    box_tuples = [tuple(b) for b in boxes1]
    clusters = cluster_lesions_fn(
        box_tuples, margin_ratio=margin_ratio, min_margin_px=min_margin_px, max_extent_px=max_extent_px
    )

    stage2_boxes, stage2_scores = [], []
    n_crops = 0
    for idxs in clusters:
        cluster_boxes = [box_tuples[i] for i in idxs]
        cx1, cy1, cx2, cy2 = compute_crop_region_fn(
            cluster_boxes, W, H, pad_ratio=pad_ratio, min_crop_px=min_crop_px
        )
        cx1i, cy1i, cx2i, cy2i = int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))
        if cx2i - cx1i < 8 or cy2i - cy1i < 8:
            continue
        crop = img[cy1i:cy2i, cx1i:cx2i]
        crop_resized = cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        n_crops += 1

        res2 = model_stage2.predict(source=crop_resized, conf=conf2, imgsz=imgsz, verbose=False)[0]
        if res2.boxes is None or len(res2.boxes) == 0:
            continue

        cw, ch = cx2i - cx1i, cy2i - cy1i
        scale_x, scale_y = cw / float(imgsz), ch / float(imgsz)
        b2 = res2.boxes.xyxy.cpu().numpy()
        s2 = res2.boxes.conf.cpu().numpy()
        for (bx1, by1, bx2, by2), sc in zip(b2, s2):
            ox1 = cx1i + bx1 * scale_x
            oy1 = cy1i + by1 * scale_y
            ox2 = cx1i + bx2 * scale_x
            oy2 = cy1i + by2 * scale_y
            stage2_boxes.append([ox1, oy1, ox2, oy2])
            stage2_scores.append(float(sc))

    stage2_boxes_arr = np.array(stage2_boxes) if stage2_boxes else np.zeros((0, 4))
    stage2_scores_arr = np.array(stage2_scores) if stage2_scores else np.zeros((0,))

    if len(stage2_boxes_arr):
        all_boxes = np.concatenate([boxes1, stage2_boxes_arr], axis=0)
        all_scores = np.concatenate([scores1, stage2_scores_arr], axis=0)
    else:
        all_boxes, all_scores = boxes1, scores1

    keep = nms_boxes(all_boxes, all_scores, iou_thresh=merge_iou)
    return all_boxes[keep], all_scores[keep], n_crops


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_tp_fp_fn(pred_boxes, gt_boxes, iou_thresh: float = 0.5):
    matched_gt, matched_pred = set(), set()
    order = np.argsort(-np.array([0.0] * len(pred_boxes))) if len(pred_boxes) else []
    for pi in range(len(pred_boxes)):
        best_iou, best_gi = 0.0, -1
        for gi, gbox in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = iou_xyxy(pred_boxes[pi], gbox)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def load_gt_boxes(label_path: Path, W: int, H: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        coords = list(map(float, line.split()[1:]))
        xs = np.array(coords[0::2]) * W
        ys = np.array(coords[1::2]) * H
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return boxes
