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

# ========== 现有功能（保留） ==========

def _kmeans_brightest_centroid_in_box(image_bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, W, H).astype(int)
    if x2 <= x1 or y2 <= y1:
        return np.array([(x1 + x2) // 2, (y1 + y2) // 2], dtype=np.int32)
    patch = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    flat = V.reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, centers = cv2.kmeans(flat, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(V.shape)
    bright_id = int(np.argmax(centers.flatten()))
    ys, xs = np.where(labels == bright_id)
    if len(xs) == 0:
        return np.array([(x1 + x2) // 2, (y1 + y2) // 2], dtype=np.int32)
    cx_local = int(np.mean(xs)); cy_local = int(np.mean(ys))
    cx = np.clip(x1 + cx_local, 0, W - 1); cy = np.clip(y1 + cy_local, 0, H - 1)
    return np.array([int(cx), int(cy)], dtype=np.int32)

def _hsv_cc_centroid_in_box_param(
    image_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    params: Dict[str, Any] | None = None
) -> np.ndarray:
    if params is None: params = {}
    ch = str(params.get("channel", "V")).upper()
    blur_ksize = int(params.get("blur_ksize", 0))
    thr_method = str(params.get("thr_method", "otsu")).lower()
    thr_offset = float(params.get("thr_offset", 0.0))
    invert = bool(params.get("invert", False))
    min_area_ratio = float(params.get("min_area_ratio", 0.001))

    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, W, H).astype(int)
    if x2 <= x1 or y2 <= y1:
        return np.array([(x1 + x2) // 2, (y1 + y2) // 2], dtype=np.int32)

    patch = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if ch == "H": ch_img = hsv[:, :, 0]
    elif ch == "S": ch_img = hsv[:, :, 1]
    else: ch_img = hsv[:, :, 2]

    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        ch_img = cv2.GaussianBlur(ch_img, (blur_ksize, blur_ksize), 0)

    if thr_method == "fixed":
        t = np.clip(int(thr_offset), 0, 255)
        _, bin_local = cv2.threshold(ch_img, t, 255, cv2.THRESH_BINARY)
    elif thr_method == "mean":
        bin_local = cv2.adaptiveThreshold(ch_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, int(thr_offset))
    elif thr_method == "gauss":
        bin_local = cv2.adaptiveThreshold(ch_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, int(thr_offset))
    else:  # otsu
        _, bin_local = cv2.threshold(ch_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert: bin_local = 255 - bin_local

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_local, connectivity=8)
    if num <= 1:
        return np.array([(x1 + x2) // 2, (y1 + y2) // 2], dtype=np.int32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    area_th = max(1, int((x2 - x1) * (y2 - y1) * min_area_ratio))
    keep_ids = [i + 1 for i, a in enumerate(areas) if a >= area_th]
    if not keep_ids: keep_ids = [int(np.argmax(areas)) + 1]
    k = max([(stats[i, cv2.CC_STAT_AREA], i) for i in keep_ids])[1]
    cx_local, cy_local = centroids[k]
    cx = int(np.clip(x1 + cx_local, 0, W - 1))
    cy = int(np.clip(y1 + cy_local, 0, H - 1))
    return np.array([cx, cy], dtype=np.int32)

def _hsv_prior_mask_in_box(image_bgr: np.ndarray, box_xyxy: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    if params is None: params = {}
    ch = str(params.get("channel", "V")).upper()
    blur_ksize = int(params.get("blur_ksize", 0))
    thr_method = str(params.get("thr_method", "otsu")).lower()
    thr_offset = float(params.get("thr_offset", 0.0))
    invert = bool(params.get("invert", False))

    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, W, H).astype(int)
    prior = np.zeros((H, W), np.uint8)
    if x2 <= x1 or y2 <= y1: return prior

    patch = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if ch == "H": ch_img = hsv[:, :, 0]
    elif ch == "S": ch_img = hsv[:, :, 1]
    else: ch_img = hsv[:, :, 2]

    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        ch_img = cv2.GaussianBlur(ch_img, (blur_ksize, blur_ksize), 0)

    if thr_method == "fixed":
        t = np.clip(int(thr_offset), 0, 255)
        _, bin_local = cv2.threshold(ch_img, t, 1, cv2.THRESH_BINARY)
    elif thr_method == "mean":
        bin_local = cv2.adaptiveThreshold(ch_img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, int(thr_offset))
    elif thr_method == "gauss":
        bin_local = cv2.adaptiveThreshold(ch_img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, int(thr_offset))
    else:
        _, bin_local = cv2.threshold(ch_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert: bin_local = 1 - bin_local
    prior[y1:y2, x1:x2] = bin_local.astype(np.uint8)
    return prior

def _refine_mask_morphology(mask01: np.ndarray, image_bgr: np.ndarray, box_xyxy: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    if params is None: params = {}
    min_area_ratio = float(params.get("min_area_ratio", 0.001))
    close_iter = int(params.get("close_iter", 1))
    open_iter  = int(params.get("open_iter", 1))
    prior_dilate_iter = int(params.get("prior_dilate_iter", 1))

    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, W, H).astype(int)
    if mask01.dtype != np.uint8: mask01 = (mask01 > 0).astype(np.uint8)

    prior = _hsv_prior_mask_in_box(image_bgr, box_xyxy, params)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if prior_dilate_iter > 0: prior = cv2.dilate(prior, k3, iterations=prior_dilate_iter)

    refined = cv2.bitwise_and(mask01, prior)

    area_min = max(1, int((x2 - x1) * (y2 - y1) * min_area_ratio))
    if area_min > 1:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        keep = np.zeros_like(refined)
        for k in range(1, num):
            if stats[k, cv2.CC_STAT_AREA] >= area_min: keep[labels == k] = 1
        refined = keep

    if close_iter > 0: refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k3, iterations=close_iter)
    if open_iter  > 0: refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN,  k3, iterations=open_iter)

    if x2 > x1 and y2 > y1:
        sub = np.zeros((y2 - y1 + 2, x2 - x1 + 2), np.uint8)
        sub[1:-1, 1:-1] = refined[y1:y2, x1:x2].copy()
        h_, w_ = sub.shape
        mask_ff = np.zeros((h_ + 2, w_ + 2), np.uint8)
        cv2.floodFill(sub, mask_ff, (0, 0), 1)
        holes = 1 - sub[1:-1, 1:-1]
        refined[y1:y2, x1:x2] = cv2.bitwise_or(refined[y1:y2, x1:x2], holes)

    return (refined > 0).astype(np.uint8)

# ========== GrabCut（保留并修正小错字） ==========

GRABCUT_DEFAULTS = {
    "band_px": 6,
    "erode_fg": 2,
    "erode_bg": 2,
    "iter": 3,
    "post_open": 0,
    "post_close": 1,
}

def _grabcut_refine(image_bgr: np.ndarray, init_mask01: np.ndarray, box_xyxy: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    ps = dict(GRABCUT_DEFAULTS)
    if params: ps.update(params)
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in _clip_xyxy(box_xyxy, W, H)]
    band = int(ps["band_px"])
    xb1, yb1 = max(0, x1 - band), max(0, y1 - band)
    xb2, yb2 = min(W, x2 + band), min(H, y2 + band)
    sub_img = image_bgr[yb1:yb2, xb1:xb2]
    sub_mask0 = init_mask01[yb1:yb2, xb1:xb2].astype(np.uint8)

    gc = np.full(sub_mask0.shape, cv2.GC_PR_BGD, np.uint8)
    rx1, ry1 = (x1 - xb1), (y1 - yb1)
    rx2, ry2 = (x2 - xb1), (y2 - yb1)
    rect = np.zeros_like(gc, np.uint8)
    rect[ry1:ry2, rx1:rx2] = 1
    gc[rect == 0] = cv2.GC_PR_BGD

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv2.erode(sub_mask0, k, iterations=int(ps["erode_fg"]))
    sure_bg = cv2.erode((1 - sub_mask0), k, iterations=int(ps["erode_bg"]))

    gc[sure_fg > 0] = cv2.GC_FGD
    gc[sure_bg > 0] = cv2.GC_BGD
    gc[(sub_mask0 > 0) & (sure_fg == 0) & (gc != cv2.GC_BGD)] = cv2.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(sub_img, gc, None, bgdModel, fgdModel, iterCount=int(ps["iter"]), mode=cv2.GC_INIT_WITH_MASK)

    sub_ref = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)

    if ps["post_close"] > 0: sub_ref = cv2.morphologyEx(sub_ref, cv2.MORPH_CLOSE, k, iterations=int(ps["post_close"]))
    if ps["post_open"]  > 0: sub_ref = cv2.morphologyEx(sub_ref, cv2.MORPH_OPEN,  k, iterations=int(ps["post_open"]))

    out = np.zeros((H, W), np.uint8)
    out[yb1:yb2, xb1:xb2] = sub_ref  # 修正：原先误写为 yb1:y2b
    return out

# ========== 基础推理入口（保留） ==========

def _get_masks_scores_box(image_bgr, boxes_xyxy, predictor):
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
    纯中心点提示：仅用点（不传 box）调用 SAM2。
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

def _get_masks_scores_box_kmeans_centroid(image_bgr, boxes_xyxy, predictor):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        pt = _kmeans_brightest_centroid_in_box(image_bgr, b)
        m, s = _predict_one(predictor, b, pt)
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

def _get_masks_scores_box_hsv_cc_centroid(image_bgr, boxes_xyxy, predictor, params: Dict[str, Any] | None = None):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        pt = _hsv_cc_centroid_in_box_param(image_bgr, b, params)
        m, s = _predict_one(predictor, b, pt)
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

def _get_masks_scores_box_hsv_cc_centroid_refine(image_bgr, boxes_xyxy, predictor, params: Dict[str, Any] | None = None):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        pt = _hsv_cc_centroid_in_box_param(image_bgr, b, params)
        m, s = _predict_one(predictor, b, pt)
        m = _refine_mask_morphology(m, image_bgr, b, params)
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

# --- GrabCut 版本（保留） ---
def _get_masks_scores_box_grabcut(image_bgr, boxes_xyxy, predictor, params: Dict[str, Any] | None = None):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        m, s = _predict_one(predictor, b, None)
        m_ref = _grabcut_refine(image_bgr, m, b, params)
        masks.append(m_ref); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

def _get_masks_scores_box_center_grabcut(image_bgr, boxes_xyxy, predictor, params: Dict[str, Any] | None = None):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        cx = int(round((b[0] + b[2]) / 2.0)); cy = int(round((b[1] + b[3]) / 2.0))
        m, s = _predict_one(predictor, b, np.array([cx, cy], np.int32))
        m_ref = _grabcut_refine(image_bgr, m, b, params)
        masks.append(m_ref); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

def _get_masks_scores_box_hsv_cc_centroid_grabcut(image_bgr, boxes_xyxy, predictor, params: Dict[str, Any] | None = None):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        pt = _hsv_cc_centroid_in_box_param(image_bgr, b, params)
        m, s = _predict_one(predictor, b, pt)
        m_ref = _grabcut_refine(image_bgr, m, b, params)
        masks.append(m_ref); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

# ========== 这里开始：快速“基于传统图像处理的框矫正” ==========

FAST_CORRECT_DEFAULTS: Dict[str, Any] = {
    "expand_ratio": 0.15,
    "min_area_ratio": 0.02,
    "close_iter": 1,
    "open_iter": 0,
    "hsv_use": "V",
    "hsv_blur_ksize": 3,
    "hsv_thr": "otsu",
    "hsv_thr_offset": 0.0,
    "hsv_invert": False,
    "edge_channel": "V",
    "edge_canny_low": 50,
    "edge_canny_high": 150,
    "edge_dilate_iter": 1,
    "pad_ratio": 0.03,
}

def _expand_window(box_xyxy: np.ndarray, W: int, H: int, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, W, H)
    w = x2 - x1; h = y2 - y1
    ex = w * ratio; ey = h * ratio
    return _clip_xyxy(np.array([x1 - ex, y1 - ey, x2 + ex, y2 + ey], np.float32), W, H)

def _largest_rect_from_mask(mask: np.ndarray, min_area: int) -> Tuple[bool, Tuple[int,int,int,int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1: return False, (0,0,0,0)
    cand = [(stats[i, cv2.CC_STAT_AREA], tuple(stats[i, :4])) for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not cand: return False, (0,0,0,0)
    _, (x, y, w, h) = max(cand, key=lambda t: t[0])
    return True, (x, y, w, h)

def _fast_adaptive_box_correct(image_bgr: np.ndarray, box_xyxy: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    ps = dict(FAST_CORRECT_DEFAULTS)
    if params: ps.update(params)

    H, W = image_bgr.shape[:2]
    win = _expand_window(box_xyxy, W, H, float(ps["expand_ratio"]))
    x1, y1, x2, y2 = win.astype(int)
    if x2 <= x1 or y2 <= y1: return _clip_xyxy(box_xyxy, W, H)

    patch = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    ch_map = {"H":0, "S":1, "V":2}
    ch = ch_map.get(str(ps["hsv_use"]).upper(), 2)
    ch_img = hsv[:, :, ch]
    if ps["hsv_blur_ksize"] >= 3 and ps["hsv_blur_ksize"] % 2 == 1:
        ch_img = cv2.GaussianBlur(ch_img, (int(ps["hsv_blur_ksize"]), int(ps["hsv_blur_ksize"])), 0)

    thr = str(ps["hsv_thr"]).lower()
    if thr == "fixed":
        t = np.clip(int(ps["hsv_thr_offset"]), 0, 255)
        _, bin_local = cv2.threshold(ch_img, t, 255, cv2.THRESH_BINARY)
    elif thr == "mean":
        bin_local = cv2.adaptiveThreshold(ch_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, int(ps["hsv_thr_offset"]))
    elif thr == "gauss":
        bin_local = cv2.adaptiveThreshold(ch_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, int(ps["hsv_thr_offset"]))
    else:
        _, bin_local = cv2.threshold(ch_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if bool(ps["hsv_invert"]): bin_local = 255 - bin_local

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if int(ps["close_iter"]) > 0: bin_local = cv2.morphologyEx(bin_local, cv2.MORPH_CLOSE, k3, iterations=int(ps["close_iter"]))
    if int(ps["open_iter"])  > 0: bin_local = cv2.morphologyEx(bin_local, cv2.MORPH_OPEN,  k3, iterations=int(ps["open_iter"]))

    ok, rect = _largest_rect_from_mask(bin_local, min_area=int((x2-x1)*(y2-y1)*float(ps["min_area_ratio"])))
    if ok:
        bx, by, bw, bh = rect
        px = bw * float(ps["pad_ratio"]); py = bh * float(ps["pad_ratio"])
        out = _clip_xyxy(np.array([x1+bx-px, y1+by-py, x1+bx+bw+px, y1+by+bh+py], np.float32), W, H)
        return out

    edge_ch = ch_map.get(str(ps["edge_channel"]).upper(), 2)
    e_img = hsv[:, :, edge_ch]
    e_img = cv2.GaussianBlur(e_img, (3,3), 0)
    edges = cv2.Canny(e_img, threshold1=float(ps["edge_canny_low"]), threshold2=float(ps["edge_canny_high"]))
    if int(ps["edge_dilate_iter"]) > 0:
        edges = cv2.dilate(edges, k3, iterations=int(ps["edge_dilate_iter"]))
    ok, rect = _largest_rect_from_mask(edges, min_area=int((x2-x1)*(y2-y1)*float(ps["min_area_ratio"])*0.5))
    if ok:
        bx, by, bw, bh = rect
        px = bw * float(ps["pad_ratio"]); py = bh * float(ps["pad_ratio"])
        out = _clip_xyxy(np.array([x1+bx-px, y1+by-py, x1+bx+bw+px, y1+by+bh+py], np.float32), W, H)
        return out

    return _clip_xyxy(box_xyxy, W, H)

# ========== 两个高层入口：只跑一次 SAM ==========

def _get_masks_scores_box_fast_correct(image_bgr, boxes_xyxy, predictor, params: Dict[str, Any] | None = None):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b2 = _fast_adaptive_box_correct(image_bgr, _clip_xyxy(b, W, H), params)
        m, s = _predict_one(predictor, b2, None)
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

def _get_masks_scores_box_hsv_cc_centroid_fast_correct(image_bgr, boxes_xyxy, predictor, params: Dict[str, Any] | None = None):
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores = [], []
    for b in boxes_xyxy:
        b2 = _fast_adaptive_box_correct(image_bgr, _clip_xyxy(b, W, H), params)
        pt = _hsv_cc_centroid_in_box_param(image_bgr, b2, params)
        m, s = _predict_one(predictor, b2, pt)
        masks.append(m); scores.append(s)
    if not masks: return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

# ========== Scale-aware erosion（新增） ==========

SCALE_ERODE_DEFAULTS: Dict[str, Any] = {
    # —— 如何划分大小尺度 ——
    # thr_type = "ratio": 使用面积比阈值；"pixels": 使用绝对像素阈值
    "thr_type": "ratio",
    # ratio_base = "box": 以每个框的面积为基准；"image": 以整幅图像面积为基准
    "ratio_base": "box",
    # 使用面积比时的阈值（小于 small_thr -> small；介于 small_thr 和 large_thr -> medium；否则 large）
    "small_thr": 0.10,   # 0.10 表示 <10% 归为 small
    "large_thr": 0.30,   # 0.30 表示 10%~30% 归为 medium，>=30% 归为 large

    # 使用像素阈值时（当 thr_type="pixels"）：
    "small_thr_px": 32 * 32,
    "large_thr_px": 96 * 96,

    # —— 腐蚀参数（按尺度） ——
    # 内核形状固定为椭圆3x3/5x5/7x7，可调迭代次数
    "erode_small_kernel": 3,   # small 实例使用的核尺寸（3/5/7等奇数）
    "erode_small_iter":  2,
    "erode_medium_kernel": 3,
    "erode_medium_iter":  1,
    "erode_large_kernel": 3,
    "erode_large_iter":   0,   # 大目标默认不腐蚀

    # —— 可选的先/后形态学（稳定性）——
    "pre_close_iter": 0,
    "post_open_iter": 0,

    # —— 过滤超小噪声（按框面积占比）——
    "min_keep_area_ratio": 0.001,  # 小于该占比的连通域删除（以框面积为基准）
}

def _scale_class_from_mask(
    mask01: np.ndarray,
    box_xyxy: np.ndarray,
    image_shape: Tuple[int, int],
    ps: Dict[str, Any]
) -> str:
    """根据配置，判定该实例为 'small' / 'medium' / 'large'。"""
    H, W = image_shape
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, W, H).astype(int)
    box_area = max(1, (x2 - x1) * (y2 - y1))
    img_area = H * W

    # 实例像素面积
    if mask01.dtype != np.uint8:
        m = (mask01 > 0).astype(np.uint8)
    else:
        m = mask01
    inst_area = int(m.sum())

    if str(ps.get("thr_type", "ratio")).lower() == "pixels":
        small_thr_px = int(ps.get("small_thr_px", 32 * 32))
        large_thr_px = int(ps.get("large_thr_px", 96 * 96))
        if inst_area < small_thr_px:
            return "small"
        elif inst_area < large_thr_px:
            return "medium"
        else:
            return "large"
    else:
        ratio_base = str(ps.get("ratio_base", "box")).lower()
        base_area = box_area if ratio_base == "box" else img_area
        ratio = inst_area / float(max(1, base_area))
        small_thr = float(ps.get("small_thr", 0.10))
        large_thr = float(ps.get("large_thr", 0.30))
        if ratio < small_thr:
            return "small"
        elif ratio < large_thr:
            return "medium"
        else:
            return "large"

def _apply_scale_erosion(
    mask01: np.ndarray,
    box_xyxy: np.ndarray,
    image_shape: Tuple[int, int],
    ps: Dict[str, Any]
) -> np.ndarray:
    """按照实例尺度进行差异化腐蚀，并做轻量连通域清理与可选形态学处理。"""
    H, W = image_shape
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, W, H).astype(int)
    if mask01.dtype != np.uint8:
        mask01 = (mask01 > 0).astype(np.uint8)

    # 可选：先闭运算（填小孔、连断裂）
    pre_close_iter = int(ps.get("pre_close_iter", 0))
    if pre_close_iter > 0:
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, k3, iterations=pre_close_iter)

    # 尺度判断
    scale = _scale_class_from_mask(mask01, box_xyxy, (H, W), ps)

    # 选择核尺寸与迭代
    if scale == "small":
        ksz = int(ps.get("erode_small_kernel", 3))
        itn = int(ps.get("erode_small_iter", 2))
    elif scale == "medium":
        ksz = int(ps.get("erode_medium_kernel", 3))
        itn = int(ps.get("erode_medium_iter", 1))
    else:
        ksz = int(ps.get("erode_large_kernel", 3))
        itn = int(ps.get("erode_large_iter", 0))

    # 保证奇数尺寸
    if ksz < 1: ksz = 1
    if ksz % 2 == 0: ksz += 1

    if itn > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask01 = cv2.erode(mask01, k, iterations=itn)

    # 清除框内过小连通域（相对框面积）
    min_keep_ratio = float(ps.get("min_keep_area_ratio", 0.001))
    area_min = max(1, int((x2 - x1) * (y2 - y1) * min_keep_ratio))
    if area_min > 1:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
        keep = np.zeros_like(mask01)
        for k_id in range(1, num):
            if stats[k_id, cv2.CC_STAT_AREA] >= area_min:
                keep[labels == k_id] = 1
        mask01 = keep

    # 可选：后开运算（去毛刺、平滑边界）
    post_open_iter = int(ps.get("post_open_iter", 0))
    if post_open_iter > 0:
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask01 = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, k3, iterations=post_open_iter)

    return (mask01 > 0).astype(np.uint8)

def _get_masks_scores_box_scale_erode(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    predictor: SAM2ImagePredictor,
    params: Dict[str, Any] | None = None
):
    """
    box 提示跑 SAM2 → 按实例尺度进行差异化腐蚀（small/medium/large）。
    尺度与腐蚀策略可通过 params（SCALE_ERODE_DEFAULTS）调参。
    """
    ps = dict(SCALE_ERODE_DEFAULTS)
    if params:
        ps.update(params)

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    masks, scores = [], []
    for b in boxes_xyxy:
        b = _clip_xyxy(b, W, H)
        # 1) 先用纯 box 提示得到实例掩码
        m, s = _predict_one(predictor, b, None)
        # 2) 按尺度腐蚀
        m2 = _apply_scale_erosion(m, b, (H, W), ps)
        masks.append(m2)
        scores.append(s)

    if not masks:
        return np.zeros((0, H, W), np.uint8), np.zeros((0,), np.float32)
    return np.stack(masks, 0), np.asarray(scores, np.float32)

# ========== 工厂（注册） ==========

def get_prompt_func(name: str, params: Dict[str, Any] | None = None):
    """
    name 支持：
      - "box"
      - "box+center"
      - "point_center" / "point+center" / "center_only"   ← 仅点提示（不传框）
      - "box+kmeans_centroid"
      - "box+hsv_cc_centroid"
      - "box+hsv_cc_centroid+refine"
      - "box+grabcut" | "box+center+grabcut" | "box+hsv_cc_centroid+grabcut"
      - "box+fast_correct"
      - "box+hsv_cc_centroid+fast_correct"
      - "box+scale_erode"（新增：按实例尺度腐蚀）
    """
    name = (name or "").lower().strip()
    if name == "box":
        return _get_masks_scores_box
    if name in ("box+center", "box+centre"):
        return _get_masks_scores_box_center

    # 纯中心点提示（不传 box）
    if name in ("point_center", "point+center", "center_only"):
        return _get_masks_scores_point_center

    if name in ("box+kmeans_centroid", "box+kmeans"):
        return _get_masks_scores_box_kmeans_centroid
    if name in ("box+hsv_cc_centroid", "box+hsv"):
        return lambda img, boxes, pred: _get_masks_scores_box_hsv_cc_centroid(img, boxes, pred, params)
    if name in ("box+hsv_cc_centroid+refine", "box+hsv+refine", "box+hsv_cc+refine"):
        return lambda img, boxes, pred: _get_masks_scores_box_hsv_cc_centroid_refine(img, boxes, pred, params)

    # GrabCut
    if name in ("box+grabcut",):
        return lambda img, boxes, pred: _get_masks_scores_box_grabcut(img, boxes, pred, params)
    if name in ("box+center+grabcut", "box+centre+grabcut"):
        return lambda img, boxes, pred: _get_masks_scores_box_center_grabcut(img, boxes, pred, params)
    if name in ("box+hsv_cc_centroid+grabcut", "box+hsv+grabcut", "box+hsv_cc+grabcut"):
        return lambda img, boxes, pred: _get_masks_scores_box_hsv_cc_centroid_grabcut(img, boxes, pred, params)

    # 快速矫正
    if name in ("box+fast_correct", "box+fast"):
        return lambda img, boxes, pred: _get_masks_scores_box_fast_correct(img, boxes, pred, params)
    if name in ("box+hsv_cc_centroid+fast_correct", "box+hsv+fast"):
        return lambda img, boxes, pred: _get_masks_scores_box_hsv_cc_centroid_fast_correct(img, boxes, pred, params)

    # —— 新增：按实例尺度腐蚀 ——
    if name in ("box+scale_erode", "box+size_erode", "box+scale"):
        return lambda img, boxes, pred: _get_masks_scores_box_scale_erode(img, boxes, pred, params)

    raise ValueError(f"Unknown prompt method: {name}")
