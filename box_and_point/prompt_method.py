#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import Tuple, Dict, Any
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ========== 通用工具 ==========

def _clip_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in b]
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2)); y2 = max(0, min(H - 1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _predict_one(
    predictor: SAM2ImagePredictor,
    box_xyxy: np.ndarray,
    point_xy: np.ndarray | None = None
) -> Tuple[np.ndarray, float]:
    """对单个 box 分割，可选 1 个点提示；返回 (mask_uint8_0_1, score_float)。"""
    if point_xy is not None:
        point_coords = np.array([point_xy], dtype=np.int32)
        point_labels = np.array([1], dtype=np.int32)  # 前景
        masks, scores, _ = predictor.predict(
            box=box_xyxy[None, :].astype(np.float32),
            point_coords=point_coords,
            point_labels=point_labels
        )
    else:
        masks, scores, _ = predictor.predict(
            box=box_xyxy[None, :].astype(np.float32)
        )
    best = int(np.argmax(scores))
    return masks[best].astype(np.uint8), float(scores[best])

def _predict_one_point(
    predictor: SAM2ImagePredictor,
    point_xy: np.ndarray
) -> Tuple[np.ndarray, float]:
    """仅用单点提示进行分割（不提供框）。"""
    point_coords = np.array([point_xy], dtype=np.int32)
    point_labels = np.array([1], dtype=np.int32)  # 前景
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels
    )
    best = int(np.argmax(scores))
    return masks[best].astype(np.uint8), float(scores[best])

# ========== 基础推理入口（三种） ==========

def _get_masks_scores_box(image_bgr, boxes_xyxy, predictor):
    """方式1：仅 box 提示。"""
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        m, s = _predict_one(predictor, b, None)
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

def _get_masks_scores_box_center(image_bgr, boxes_xyxy, predictor):
    """方式2：box + center 点提示（点取框中心）。"""
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        cx = int(round((b[0] + b[2]) / 2.0)); cy = int(round((b[1] + b[3]) / 2.0))
        m, s = _predict_one(predictor, b, np.array([cx, cy], np.int32))
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

def _get_masks_scores_point_center(image_bgr, boxes_xyxy, predictor):
    """
    方式3：纯中心点提示（不传 box）。
    点坐标取每个输入框的中心。
    """
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        cx = int(round((b[0] + b[2]) / 2.0)); cy = int(round((b[1] + b[3]) / 2.0))
        m, s = _predict_one_point(predictor, np.array([cx, cy], np.int32))
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

# ========== 工厂（注册） ==========

def get_prompt_func(name: str, params: Dict[str, Any] | None = None):
    """
    支持：
      - "box"
      - "box+center" / "box+centre"
      - "point_center" / "point+center" / "center_only"
    """
    name = (name or "").lower().strip()
    if name == "box":
        return _get_masks_scores_box
    if name in ("box+center", "box+centre"):
        return _get_masks_scores_box_center
    if name in ("point_center", "point+center", "center_only"):
        return _get_masks_scores_point_center
    raise ValueError(f"Unknown prompt method: {name}")
