#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import glob
import json
import time
import shutil
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from prompt_method import get_prompt_func

# ---------- 通用工具 ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def yolo_box_to_xyxy_auto(b: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = [float(x) for x in b]
    normalized = max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5
    if normalized:
        x1 = (cx - bw / 2.0) * w; y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w; y2 = (cy + bh / 2.0) * h
    else:
        x1 = cx - bw / 2.0; y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0; y2 = cy + bh / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def polygon_to_mask_auto(poly_xy: List[float], w: int, h: int) -> np.ndarray:
    pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 2)
    normalized = (pts.max() <= 1.5) and (pts.min() >= -0.5)
    if normalized:
        pts_px = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1)
    else:
        pts_px = pts
    pts_px[:, 0] = np.clip(pts_px[:, 0], 0, w - 1)
    pts_px[:, 1] = np.clip(pts_px[:, 1], 0, h - 1)
    pts_px = pts_px.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts_px) >= 3:
        cv2.fillPoly(mask, [pts_px], 1)
    return mask

def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union + 1e-9)

def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(u8, cv2.MORPH_GRADIENT, kernel)
    return (grad > 0).astype(np.uint8)

def boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tol: int) -> float:
    eps = 1e-9
    b_pred = _binary_boundary(pred_mask)
    b_gt   = _binary_boundary(gt_mask)
    tol = max(1, int(tol))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    b_gt_d   = cv2.dilate(b_gt, se)
    b_pred_d = cv2.dilate(b_pred, se)
    tp_p = np.logical_and(b_pred, b_gt_d).sum()
    tp_r = np.logical_and(b_gt, b_pred_d).sum()
    P = tp_p / (b_pred.sum() + eps)
    R = tp_r / (b_gt.sum() + eps)
    return float(2 * P * R / (P + R + eps))

# ====== AP 计算：COCO 101-point 版本 ======
_COCO_RECALL_POINTS = np.linspace(0.0, 1.0, 101)

def ap_101_point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    ap_sum, cnt = 0.0, 0
    for r in _COCO_RECALL_POINTS:
        inds = np.where(mrec >= r)[0]
        p = float(np.max(mpre[inds])) if inds.size > 0 else 0.0
        ap_sum += p
        cnt += 1
    return ap_sum / max(1, cnt)

def read_gt_polygons_yolo(file_path: str) -> List[Dict[str, Any]]:
    anns = []
    if not os.path.exists(file_path):
        return anns
    with open(file_path, "r") as f:
        for line in f:
            ps = line.strip().split()
            if len(ps) < 6:
                continue
            cls = int(float(ps[0]))
            coords = list(map(float, ps[1:]))
            anns.append({"cls": cls, "poly": coords})
    return anns

def read_pred_bboxes_yolo(file_path: str) -> List[Dict[str, Any]]:
    dets = []
    if not os.path.exists(file_path):
        return dets
    with open(file_path, "r") as f:
        for line in f:
            ps = line.strip().split()
            if len(ps) < 5:
                continue
            cls = int(float(ps[0]))
            if len(ps) == 6:
                tail = float(ps[-1])
                if 0.0 <= tail <= 1.0:
                    box = list(map(float, ps[1:5])); conf = tail
                else:
                    box = list(map(float, ps[1:5])); conf = 1.0
            elif len(ps) >= 7:
                maybe_conf = float(ps[1])
                if 0.0 <= maybe_conf <= 1.0:
                    conf = maybe_conf; box = list(map(float, ps[2:6]))
                else:
                    box = list(map(float, ps[1:5])); conf = float(ps[5])
            else:
                continue
            dets.append({"cls": cls, "conf": conf, "box": box})
    return dets

# ---------- 评估器（移除 Mask-SoftNMS & WBF 后处理） ----------
class SAM2SegEvaluator:
    def __init__(
        self,
        image_dir: str,
        gt_label_dir: str,
        pred_bbox_dir: str,
        result_dir: str,
        model_cfg_name: str,      # e.g. "sam2.1/sam2.1_hiera_t"
        sam2_ckpt_path: str,      # e.g. "./checkpoints/sam2.1_hiera_tiny.pt"
        device: str = "cuda",
        iou_thresholds_coco = np.arange(0.50, 0.96, 0.05),
        class_agnostic: bool = False,
        max_eval: Optional[int] = None,
        prompt_mode: str = "box",          # "box" | "box+center" | ...
        bf_tolerance_pixels: int = 2,
        prompt_param: Optional[Dict[str, Any]] = None,
        # 检查模式
        check_mode: bool = False,
        timeout_sec: float = 0.0,
        passed_set_dir: Optional[str] = None,
        # 严格 COCO 参数
        max_dets_per_image: int = 100,     # COCO 默认 100
        use_strict_coco_ap: bool = True,   # 使用 101-point AP
        coco_filter_dt_by_area: bool = True,
        # 小目标掩膜细化（工程补救）
        refine_small_masks: bool = False,
        refine_small_area_multiplier: float = 2.0,
        refine_min_component: int = 8,
        refine_kernel: int = 3,
        # （已移除的后处理项：SoftNMS/WBF 等）
    ):
        self.image_dir = image_dir
        self.gt_label_dir = gt_label_dir
        self.pred_bbox_dir = pred_bbox_dir
        self.result_dir = result_dir
        ensure_dir(result_dir)

        self.iou_thresholds_coco = iou_thresholds_coco
        self.class_agnostic = class_agnostic
        self.max_eval = max_eval
        self.prompt_mode = prompt_mode
        self.bf_tol = int(bf_tolerance_pixels)
        self.prompt_param = prompt_param or {}
        self.check_mode = bool(check_mode)
        self.timeout_sec = float(timeout_sec) if timeout_sec else 0.0
        self.passed_set_dir = passed_set_dir

        # 严格 COCO
        self.max_dets_per_image = int(max_dets_per_image)
        self.use_strict_coco_ap = bool(use_strict_coco_ap)
        self.coco_filter_dt_by_area = bool(coco_filter_dt_by_area)

        # 小目标细化
        self.refine_small_masks = bool(refine_small_masks)
        self.refine_small_area_multiplier = float(refine_small_area_multiplier)
        self.refine_min_component = int(refine_min_component)
        self.refine_kernel = int(refine_kernel if refine_kernel % 2 == 1 else refine_kernel + 1)

        print(f"[SAM2] build model: cfg={model_cfg_name}, ckpt={sam2_ckpt_path}, device={device}")
        sam2_model = build_sam2(model_cfg_name, sam2_ckpt_path, device=device, mode="eval")
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = device

        # 提示函数
        self.get_masks_scores = get_prompt_func(prompt_mode, self.prompt_param)

        # 通过样本复制
        if self.passed_set_dir:
            ensure_dir(self.passed_set_dir)
            ensure_dir(os.path.join(self.passed_set_dir, "images"))
            ensure_dir(os.path.join(self.passed_set_dir, "labels"))

        # COCO 面积桶
        self._area_s2 = 32 * 32
        self._area_m2 = 96 * 96
        self._area_buckets = ["all", "s", "m", "l"]

    def _collect_images(self):
        im_paths = sorted(
            [p for p in glob.glob(os.path.join(self.image_dir, "*"))
             if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        )
        if self.max_eval is not None:
            im_paths = im_paths[:self.max_eval]
        return im_paths

    @staticmethod
    def _union_mask(masks: np.ndarray, H: int, W: int) -> np.ndarray:
        if masks is None or masks.shape[0] == 0:
            return np.zeros((H, W), dtype=np.uint8)
        return np.any(masks.astype(bool), axis=0).astype(np.uint8)

    def _save_passed_sample(self, img_path: str):
        if not self.passed_set_dir:
            return
        name = os.path.splitext(os.path.basename(img_path))[0]
        dst_img = os.path.join(self.passed_set_dir, "images", os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        src_lab = os.path.join(self.gt_label_dir, name + ".txt")
        if os.path.exists(src_lab):
            dst_lab = os.path.join(self.passed_set_dir, "labels", name + ".txt")
            shutil.copy2(src_lab, dst_lab)

    def _bucket_of_area(self, area: int) -> str:
        if area < self._area_s2:
            return "s"
        elif area < self._area_m2:
            return "m"
        else:
            return "l"

    # ---- AP 计算 ----
    def _ap_from_lists(self, scores_list, flags_list, gt_count) -> float:
        if len(scores_list) == 0 or gt_count == 0:
            return 0.0
        scores = np.array(scores_list, dtype=np.float32)
        flags  = np.array(flags_list, dtype=np.int32)
        order = np.argsort(-scores)
        tp = flags[order]
        fp = 1 - tp
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        recalls = tp_c / (gt_count + 1e-9)
        precisions = tp_c / (tp_c + fp_c + 1e-9)
        if self.use_strict_coco_ap:
            return ap_101_point(recalls, precisions)
        else:
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def _refine_small_pred_masks(self, pred_masks: np.ndarray, pred_areas: np.ndarray) -> np.ndarray:
        if pred_masks.size == 0:
            return pred_masks
        refined = []
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.refine_kernel, self.refine_kernel))
        small_thresh = int(self._area_s2 * self.refine_small_area_multiplier)
        for i, m in enumerate(pred_masks):
            m_bin = (m > 0).astype(np.uint8)
            if pred_areas[i] < small_thresh:
                m2 = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, k)
                num, labs, stats, _ = cv2.connectedComponentsWithStats(m2, connectivity=4)
                keep = np.zeros_like(m2)
                for cid in range(1, num):
                    if stats[cid, cv2.CC_STAT_AREA] >= self.refine_min_component:
                        keep[labs == cid] = 1
                refined.append(keep.astype(np.uint8))
            else:
                refined.append(m_bin)
        return np.stack(refined, 0)

    # ===== 主评估 =====
    def evaluate(self) -> Dict[str, Any]:
        images = self._collect_images()
        if not images:
            print("No images found:", self.image_dir)
            return {}

        # 原有（macro）统计
        per_class_scores_by_thr: Dict[int, Dict[float, list]] = {}
        per_class_flags_by_thr:  Dict[int, Dict[float, list]] = {}
        per_class_gt_cnt_by_thr: Dict[int, Dict[float, int]] = {}
        per_class_total_gt: Dict[int, int] = {}

        # 严格 COCO：分面积桶
        per_class_scores_by_thr_area: Dict[int, Dict[float, Dict[str, list]]] = {}
        per_class_flags_by_thr_area:  Dict[int, Dict[float, Dict[str, list]]] = {}
        per_class_gt_cnt_by_thr_area: Dict[int, Dict[float, Dict[str, int]]]  = {}
        per_class_total_gt_area:      Dict[int, Dict[str, int]]               = {}

        def _ensure_class(c: int):
            if c not in per_class_scores_by_thr:
                per_class_scores_by_thr[c] = {thr: [] for thr in self.iou_thresholds_coco}
                per_class_flags_by_thr[c]  = {thr: [] for thr in self.iou_thresholds_coco}
                per_class_gt_cnt_by_thr[c] = {thr: 0   for thr in self.iou_thresholds_coco}
                per_class_total_gt[c] = 0
            if c not in per_class_scores_by_thr_area:
                per_class_scores_by_thr_area[c] = {thr: {b: [] for b in self._area_buckets} for thr in self.iou_thresholds_coco}
                per_class_flags_by_thr_area[c]  = {thr: {b: [] for b in self._area_buckets} for thr in self.iou_thresholds_coco}
                per_class_gt_cnt_by_thr_area[c] = {thr: {b: 0   for b in self._area_buckets} for thr in self.iou_thresholds_coco}
                per_class_total_gt_area[c]      = {b: 0 for b in self._area_buckets}

        # per-image union 指标（论文式）
        paper_iou_list, paper_bfF_list = [], []
        n_passed, n_timeout = 0, 0

        for img_path in tqdm(images, desc=f"Evaluating ({self.prompt_mode})"):
            name = os.path.splitext(os.path.basename(img_path))[0]
            im_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if im_bgr is None:
                continue
            H, W = im_bgr.shape[:2]

            # --- GT ---
            gt_path = os.path.join(self.gt_label_dir, name + ".txt")
            gt_anns = read_gt_polygons_yolo(gt_path)
            gt_masks = [polygon_to_mask_auto(a["poly"], W, H) for a in gt_anns]
            gt_cls   = [a["cls"] for a in gt_anns]
            gt_masks = np.stack(gt_masks, 0) if gt_masks else np.zeros((0, H, W), np.uint8)
            gt_cls   = np.array(gt_cls, np.int32) if gt_cls else np.zeros((0,), np.int32)
            n_gt = gt_masks.shape[0]
            gt_areas = np.array([int(m.sum()) for m in gt_masks], dtype=np.int32)
            gt_buckets = np.array([self._bucket_of_area(a) for a in gt_areas], dtype=object)

            for c in np.unique(gt_cls):
                _ensure_class(int(c))
                per_class_total_gt[int(c)] += int((gt_cls == c).sum())
                for b in self._area_buckets:
                    if b == "all":
                        cnt_b = int((gt_cls == c).sum())
                    else:
                        cnt_b = int(((gt_cls == c) & (gt_buckets == b)).sum())
                    per_class_total_gt_area[int(c)][b] += cnt_b

            # --- 预测（det -> SAM2）---
            pred_path = os.path.join(self.pred_bbox_dir, name + ".txt")
            dets = read_pred_bboxes_yolo(pred_path)
            for d in dets:
                _ensure_class(int(d["cls"]))

            t0 = time.time()
            if len(dets) > 0:
                boxes_xyxy = np.stack([yolo_box_to_xyxy_auto(np.array(d["box"]), W, H) for d in dets], 0)
                det_scores = np.array([d["conf"] for d in dets], np.float32)
                pred_cls   = np.array([d["cls"] for d in dets], np.int32)

                if self.device == "cuda":
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pred_masks, sam_scores = self.get_masks_scores(im_bgr, boxes_xyxy, self.predictor)
                else:
                    with torch.inference_mode():
                        pred_masks, sam_scores = self.get_masks_scores(im_bgr, boxes_xyxy, self.predictor)

                # 初始融合分（可按需修改策略）
                combined_scores = (det_scores * sam_scores).astype(np.float32)

                # 已移除：SoftNMS / WBF 后处理
                # 保留原始 pred_masks / pred_cls / combined_scores
            else:
                pred_masks = np.zeros((0, H, W), np.uint8)
                pred_cls   = np.zeros((0,), np.int32)
                combined_scores = np.zeros((0,), np.float32)

            elapsed = time.time() - t0
            if self.timeout_sec > 0 and elapsed > self.timeout_sec:
                n_timeout += 1
                if self.check_mode:
                    print(f"[timeout] {name} took {elapsed:.2f}s > {self.timeout_sec:.2f}s, skipped.")
                continue
            else:
                n_passed += 1
                self._save_passed_sample(img_path)

            n_pred = pred_masks.shape[0]

            # ——（可选）小目标预测掩膜细化：仅改变 pred_masks，不改变评估口径
            pred_areas = np.array([int(m.sum()) for m in pred_masks], dtype=np.int32) if n_pred > 0 else np.zeros((0,), np.int32)
            if self.refine_small_masks and n_pred > 0:
                pred_masks = self._refine_small_pred_masks(pred_masks, pred_areas)
                pred_areas = np.array([int(m.sum()) for m in pred_masks], dtype=np.int32)

            # union（论文式）
            gt_union   = self._union_mask(gt_masks, H, W)
            pred_union = self._union_mask(pred_masks, H, W)
            if (gt_union.sum() + pred_union.sum()) > 0:
                paper_iou_list.append(mask_iou(pred_union, gt_union))
                paper_bfF_list.append(boundary_f1(pred_union, gt_union, self.bf_tol))

            # IoU matrix（本图）
            iou_matrix = np.zeros((n_pred, n_gt), np.float32)
            for i in range(n_pred):
                for j in range(n_gt):
                    if self.class_agnostic or pred_cls[i] == gt_cls[j]:
                        iou_matrix[i, j] = mask_iou(pred_masks[i].astype(bool), gt_masks[j].astype(bool))

            pred_buckets = np.array([self._bucket_of_area(a) for a in pred_areas], dtype=object) if n_pred > 0 else np.array([], dtype=object)

            # ===== 1) 原 macro（不分面积桶）=====
            order_global = np.argsort(-combined_scores) if n_pred > 0 else np.array([], dtype=int)
            for thr in self.iou_thresholds_coco:
                for c in np.unique(gt_cls):
                    per_class_gt_cnt_by_thr[int(c)][thr] += int((gt_cls == c).sum())
            classes_considered = set(list(np.unique(gt_cls)) + list(np.unique(pred_cls)))
            for c in classes_considered:
                c = int(c)
                idxs_c = [idx for idx in order_global if n_pred > 0 and pred_cls[idx] == c]
                for thr in self.iou_thresholds_coco:
                    used_gt_c = set()
                    for idx in idxs_c:
                        best_iou, best_j = 0.0, -1
                        for j in range(n_gt):
                            if gt_cls[j] != c:
                                continue
                            iou_ij = float(iou_matrix[idx, j])
                            if iou_ij > best_iou:
                                best_iou, best_j = iou_ij, j
                        if best_iou >= thr and best_j not in used_gt_c:
                            per_class_flags_by_thr[c][thr].append(1)
                            used_gt_c.add(best_j)
                        else:
                            per_class_flags_by_thr[c][thr].append(0)
                        per_class_scores_by_thr[c][thr].append(
                            float(combined_scores[idx]) if n_pred > 0 else 0.0
                        )

            # ===== 2) 严格 COCO：分面积桶 + maxDets=100 + 101-point AP =====
            for c in classes_considered:
                c = int(c)
                for thr in self.iou_thresholds_coco:
                    for b in self._area_buckets:
                        if b == "all":
                            gt_idxs_bucket = [j for j in range(n_gt) if gt_cls[j] == c]
                        else:
                            gt_idxs_bucket = [j for j in range(n_gt) if gt_cls[j] == c and gt_buckets[j] == b]
                        per_class_gt_cnt_by_thr_area[c][thr][b] += len(gt_idxs_bucket)

                    for b in self._area_buckets:
                        if b == "all":
                            pred_idxs_bucket = [idx for idx in range(n_pred) if pred_cls[idx] == c]
                        else:
                            if self.coco_filter_dt_by_area:
                                pred_idxs_bucket = [idx for idx in range(n_pred)
                                                    if pred_cls[idx] == c and pred_buckets[idx] == b]
                            else:
                                pred_idxs_bucket = [idx for idx in range(n_pred) if pred_cls[idx] == c]

                        if len(pred_idxs_bucket) > 0:
                            order = np.argsort(-combined_scores[pred_idxs_bucket])
                            pred_idxs_bucket = [pred_idxs_bucket[i] for i in order]

                        if self.max_dets_per_image > 0 and len(pred_idxs_bucket) > self.max_dets_per_image:
                            pred_idxs_bucket = pred_idxs_bucket[:self.max_dets_per_image]

                        if b == "all":
                            gt_idxs_bucket = [j for j in range(n_gt) if gt_cls[j] == c]
                        else:
                            gt_idxs_bucket = [j for j in range(n_gt) if gt_cls[j] == c and gt_buckets[j] == b]

                        used_gt = set()
                        for idx in pred_idxs_bucket:
                            best_iou, best_j = 0.0, -1
                            for j in gt_idxs_bucket:
                                iou_ij = float(iou_matrix[idx, j])
                                if iou_ij > best_iou:
                                    best_iou, best_j = iou_ij, j
                            if best_iou >= thr and best_j not in used_gt:
                                per_class_flags_by_thr_area[c][thr][b].append(1)
                                used_gt.add(best_j)
                            else:
                                per_class_flags_by_thr_area[c][thr][b].append(0)
                            per_class_scores_by_thr_area[c][thr][b].append(float(combined_scores[idx]) if n_pred > 0 else 0.0)

        # ===== 原 macro 指标 =====
        per_class_AP50, per_class_mAP = {}, {}
        valid_classes = [c for c, total in per_class_total_gt.items() if total > 0]
        for c in per_class_scores_by_thr.keys():
            ap_list = []
            ap50_c = 0.0
            for thr in self.iou_thresholds_coco:
                gt_cnt_c = per_class_gt_cnt_by_thr[c][thr]
                scores_c = per_class_scores_by_thr[c][thr]
                flags_c  = per_class_flags_by_thr[c][thr]
                ap_c_thr = self._ap_from_lists(scores_c, flags_c, gt_cnt_c)
                ap_list.append(ap_c_thr)
                if abs(thr - 0.5) < 1e-6:
                    ap50_c = ap_c_thr
            per_class_AP50[c] = ap50_c
            per_class_mAP[c]  = float(np.mean(ap_list)) if len(ap_list) else 0.0

        macro_AP50 = float(np.mean([per_class_AP50[c] for c in valid_classes])) if valid_classes else 0.0
        macro_mAP  = float(np.mean([per_class_mAP[c]  for c in valid_classes])) if valid_classes else 0.0

        # ===== 严格 COCO 桶指标 =====
        def _compute_bucket_maps(bucket: str):
            classes_valid = [c for c, totals in per_class_total_gt_area.items() if totals[bucket] > 0]
            per_class_mAP_bucket: Dict[int, float] = {}
            per_class_AP50_bucket: Dict[int, float] = {}
            per_class_AP75_bucket: Dict[int, float] = {}
            for c in per_class_scores_by_thr_area.keys():
                ap_list = []
                ap50_c = 0.0
                ap75_c = 0.0
                for thr in self.iou_thresholds_coco:
                    gt_cnt = per_class_gt_cnt_by_thr_area[c][thr][bucket]
                    scores = per_class_scores_by_thr_area[c][thr][bucket]
                    flags  = per_class_flags_by_thr_area[c][thr][bucket]
                    ap_thr = self._ap_from_lists(scores, flags, gt_cnt)
                    ap_list.append(ap_thr)
                    if abs(thr - 0.5) < 1e-6:
                        ap50_c = ap_thr
                    if abs(thr - 0.75) < 1e-6:
                        ap75_c = ap_thr
                per_class_mAP_bucket[c]   = float(np.mean(ap_list)) if len(ap_list) else 0.0
                per_class_AP50_bucket[c]  = ap50_c
                per_class_AP75_bucket[c]  = ap75_c
            mAP_bucket  = float(np.mean([per_class_mAP_bucket[c]  for c in classes_valid])) if classes_valid else 0.0
            AP50_bucket = float(np.mean([per_class_AP50_bucket[c] for c in classes_valid])) if classes_valid else 0.0
            AP75_bucket = float(np.mean([per_class_AP75_bucket[c] for c in classes_valid])) if classes_valid else 0.0
            return mAP_bucket, AP50_bucket, AP75_bucket

        coco_mAP_all, coco_AP50_all, coco_AP75_all = _compute_bucket_maps("all")
        coco_mAP_s, _, _ = _compute_bucket_maps("s")
        coco_mAP_m, _, _ = _compute_bucket_maps("m")
        coco_mAP_l, _, _ = _compute_bucket_maps("l")

        # ===== 诊断：各桶 GT/DT/TP@50 =====
        bucket_diag = {b: {"gt": 0, "dt": 0, "tp@50": 0} for b in self._area_buckets}
        for c, totals in per_class_total_gt_area.items():
            for b in self._area_buckets:
                bucket_diag[b]["gt"] += int(totals[b])
        if 0.50 in self.iou_thresholds_coco:
            thr50 = 0.50
        else:
            thr50 = float(min(self.iou_thresholds_coco, key=lambda t: abs(t - 0.50)))
        for b in self._area_buckets:
            dt_count = 0
            tp_count = 0
            for c in per_class_scores_by_thr_area.keys():
                scores = per_class_scores_by_thr_area[c][thr50][b]
                flags  = per_class_flags_by_thr_area[c][thr50][b]
                dt_count += len(scores)
                tp_count += int(np.sum(flags))
            bucket_diag[b]["dt"]    = int(dt_count)
            bucket_diag[b]["tp@50"] = int(tp_count)

        summary = {
            "images_evaluated": len(images),
            "prompt_mode": self.prompt_mode,
            # 原有（macro）
            "macro_AP50": macro_AP50,
            "macro_mAP50_95": macro_mAP,
            "mIoU_paper_style": float(np.mean(paper_iou_list)) if len(paper_iou_list) else 0.0,
            "BF_F1_paper_style": float(np.mean(paper_bfF_list)) if len(paper_bfF_list) else 0.0,
            # 严格 COCO / MMDet 命名
            "segm_mAP": coco_mAP_all,        # 0.50:0.95
            "segm_mAP_50": coco_AP50_all,    # AP@0.50
            "segm_mAP_75": coco_AP75_all,    # AP@0.75
            "segm_mAP_s": coco_mAP_s,
            "segm_mAP_m": coco_mAP_m,
            "segm_mAP_l": coco_mAP_l,
            # 记录严格参数 + 诊断信息
            "strict_coco": {
                "maxDets": self.max_dets_per_image,
                "use_101_point_AP": self.use_strict_coco_ap,
                "coco_filter_dt_by_area": self.coco_filter_dt_by_area,
                "area_ranges": {"s": f"[0,{self._area_s2})", "m": f"[{self._area_s2},{self._area_m2})", "l": f"[{self._area_m2},+inf)"},
            },
            "diag_area_buckets": bucket_diag,
            "refine_small_masks": {
                "enabled": self.refine_small_masks,
                "small_area_multiplier": self.refine_small_area_multiplier,
                "min_component": self.refine_min_component,
                "kernel": self.refine_kernel,
            },
            #（已移除 postproc 配置回显）
        }

        if self.check_mode or self.timeout_sec > 0:
            summary.update({
                "timeout_sec": self.timeout_sec,
                "num_passed": int(n_passed),
                "num_timeout_skipped": int(n_timeout),
                "passed_set_dir": self.passed_set_dir if self.passed_set_dir else None,
            })

        return summary
