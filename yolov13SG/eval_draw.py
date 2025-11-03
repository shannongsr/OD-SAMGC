#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a pre-split subset (with its own data.yaml where test: points to subset).
Now also supports generating color overlays of predicted segmentation masks
(overlaid only on foreground; background unchanged).

Usage example:
python eval_subset.py \
  --weights runs/segment/SRYOLO_coffee-seg/weights/last.pt \
  --data /content/drive/MyDrive/yolov13-coffeev2/merged_without_part_003/data.yaml \
  --project runs/test_eval --name yolov13_coffee2 \
  --imgsz 640 --batch 4 --conf 0.001 --iou 0.6 \
  --out-label-dir pred_labels_yolov13_test \
  --overlay-dir overlays --alpha 0.6 --color-mode class
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------- utils -----------------
def _resolve(p: Path) -> Path:
    return Path(p).expanduser().resolve()

def _read_lines(p: Path) -> list[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _gather_test_images(test_entry: str | Path) -> list[Path]:
    """Accept a folder path or a .txt list file."""
    test_entry = _resolve(Path(test_entry))
    imgs: list[Path] = []
    if test_entry.is_file() and test_entry.suffix.lower() == ".txt":
        for line in _read_lines(test_entry):
            ip = _resolve(Path(line))
            if ip.is_file() and ip.suffix.lower() in IMG_EXTS:
                imgs.append(ip)
    elif test_entry.is_dir():
        for ext in IMG_EXTS:
            imgs.extend(test_entry.rglob(f"*{ext}"))
    else:
        print(f"[Warn] test entry not found or invalid: {test_entry}")
    imgs = sorted(set(imgs))
    print(f"[Info] Found {len(imgs)} test images.")
    return imgs

def _default_class_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    colors = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    return colors

def _instance_color(idx: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(idx * 9176 + 23)
    c = rng.integers(0, 255, size=(3,), dtype=np.uint8)
    return int(c[0]), int(c[1]), int(c[2])

def _overlay_foreground_only(image_bgr: np.ndarray, mask: np.ndarray,
                             color: tuple[int,int,int], alpha: float) -> np.ndarray:
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    color_img[mask.astype(bool)] = color
    blended = image_bgr.copy()
    fg = mask.astype(bool)
    blended[fg] = (alpha * color_img[fg] + (1 - alpha) * blended[fg]).astype(np.uint8)
    return blended

def _parse_seg_line_flexible(line: str, W: int, H: int):
    """
    Robust parser for Ultralytics segmentation prediction lines.

    Supported formats:
      A) cls conf xc yc w h x1 y1 x2 y2 ...
      B) cls xc yc w h x1 y1 x2 y2 ...   (no conf)
    Returns:
      (cls_id:int, conf:float, polygon_pts: (N,2) float32 in pixel coords) or None if invalid.
    """
    parts = line.strip().split()
    if len(parts) < 6:
        # Not enough to even have bbox
        return None

    try:
        cls_id = int(float(parts[0]))
    except Exception:
        return None

    nums = list(map(float, parts[1:]))

    def _denorm_poly(vec):
        if len(vec) < 6:  # need >= 3 points
            return None
        if len(vec) % 2 != 0:
            vec = vec[:-1]
        pts = np.array(vec, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0] * W, 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1] * H, 0, H - 1)
        return pts

    # Try format A: with conf
    # nums[0]=conf, nums[1:5]=bbox, nums[5:]=poly
    conf = 1.0
    poly = None

    if len(nums) >= 11:  # 1(conf)+4(bbox)+6(poly)=11 at least
        conf_cand = nums[0]
        poly_cand = _denorm_poly(nums[5:])
        if poly_cand is not None:
            conf = float(conf_cand)
            poly = poly_cand

    # Fallback: format B (no conf)
    if poly is None and len(nums) >= 10:  # 4(bbox)+6(poly)=10
        poly_cand = _denorm_poly(nums[4:])
        if poly_cand is not None:
            conf = 1.0
            poly = poly_cand

    if poly is None or poly.shape[0] < 3:
        return None

    return cls_id, conf, poly

def _load_names_from_yaml(data_yaml: Path) -> list[str] | None:
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        names = d.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        if isinstance(names, list):
            return names
    except Exception:
        pass
    return None

def _build_label_lookup(label_root: Path) -> dict[str, Path]:
    lut = {}
    if not label_root.is_dir():
        return lut
    for p in label_root.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt":
            lut[p.stem] = p
    return lut

def _make_overlay_for_image(img_path: Path,
                            label_path: Path | None,
                            overlay_dir: Path,
                            color_mode: str,
                            alpha: float,
                            class_palette: np.ndarray | None,
                            stats: dict):
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"[Warn] cannot read image: {img_path}")
        stats["img_read_fail"] += 1
        return
    H, W = image_bgr.shape[:2]

    if label_path is None or (not label_path.exists()):
        out_p = overlay_dir / img_path.name
        cv2.imwrite(str(out_p), image_bgr)
        stats["no_label_txt"] += 1
        return

    over = image_bgr.copy()
    lines = []
    try:
        lines = _read_lines(label_path)
    except Exception:
        pass

    inst_idx = 0
    valid_poly_cnt = 0
    for line in lines:
        parsed = _parse_seg_line_flexible(line, W, H)
        if parsed is None:
            # could be "box-only" or malformed seg
            stats["invalid_or_box_only"] += 1
            continue
        cls_id, conf, poly = parsed
        if poly.shape[0] < 3:
            stats["too_few_points"] += 1
            continue

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        if color_mode == "class" and class_palette is not None:
            c = tuple(int(v) for v in class_palette[cls_id % len(class_palette)][::-1])  # BGR
        else:
            c = _instance_color(inst_idx)
        inst_idx += 1

        over = _overlay_foreground_only(over, mask, c, alpha)
        valid_poly_cnt += 1

    out_p = overlay_dir / img_path.name
    cv2.imwrite(str(out_p), over)

    if valid_poly_cnt == 0:
        stats["images_no_valid_poly"] += 1

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Evaluate one subset (data.yaml with test pointing to subset)")
    ap.add_argument("--weights", required=True, type=str, help="Path to .pt weights")
    ap.add_argument("--data", required=True, type=str, help="Path to the subset's data.yaml")
    ap.add_argument("--project", type=str, default="runs/test_eval", help="Ultralytics project dir")
    ap.add_argument("--name", type=str, default="subset_eval", help="Ultralytics run name")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--out-label-dir", type=str, default="pred_labels_subset",
                    help="Where to collect predicted labels")
    ap.add_argument("--no-clear-out", action="store_true",
                    help="Do not clear OUT_LABEL_DIR before copy")
    # --- overlay options ---
    ap.add_argument("--overlay-dir", type=str, default=None,
                    help="If set, write color overlays of predicted masks to this folder")
    ap.add_argument("--alpha", type=float, default=0.5, help="Mask opacity (0~1)")
    ap.add_argument("--color-mode", type=str, default="class", choices=["class", "instance"],
                    help="Color by class palette or per-instance random")
    args = ap.parse_args()

    weights = _resolve(Path(args.weights))
    data_yaml = _resolve(Path(args.data))
    project = _resolve(Path(args.project))
    out_label_dir = _resolve(Path(args.out_label_dir))
    overlay_dir = _resolve(Path(args.overlay_dir)) if args.overlay_dir else None

    # 读 data.yaml
    test_entry = None
    names = None
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        test_entry = d.get("test")
        print("[Info] 子集 data.yaml 加载成功。test 指向：", test_entry)
        names = _load_names_from_yaml(data_yaml)
        if names:
            print(f"[Info] names: {names}")
    except Exception as e:
        print("[Warn] 子集 data.yaml 读取失败，但继续评估：", e)

    # 评估（并生成 labels）
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        save_txt=True,
        save_conf=True,
        project=str(project),
        name=args.name,
        verbose=True,
    )

    # 兼容不同版本 metrics
    metrics_dict = {}
    try:
        metrics_dict = results.results_dict
    except Exception:
        try:
            m = results.metrics
            metrics_dict = {
                "mAP50-95": getattr(m, "map", None),
                "mAP50": getattr(m, "map50", None),
                "mAP75": getattr(m, "map75", None),
                "P": getattr(m, "mp", None),
                "R": getattr(m, "mr", None),
            }
        except Exception:
            pass

    print("\n===== Evaluation Metrics (this subset) =====")
    print(metrics_dict)

    # 收集预测 labels
    pred_label_src = project / args.name / "labels"
    print(f"\nPredicted label txt files are at: {pred_label_src}")

    if pred_label_src.is_dir():
        out_label_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_clear_out:
            for f in out_label_dir.iterdir():
                if f.is_file():
                    f.unlink()
        copied = 0
        for f in pred_label_src.iterdir():
            if f.is_file() and f.suffix.lower() == ".txt":
                shutil.copy2(f, out_label_dir / f.name)
                copied += 1
        print(f"Copied {copied} label files to: {out_label_dir}")
    else:
        print("WARNING: labels folder not found. Make sure save_txt=True and split='test' worked.")

    # 生成掩码叠加图（只在前景区域着色）
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # 解析 test 集图片清单
        img_list: list[Path] = []
        if test_entry is not None:
            img_list = _gather_test_images(test_entry)
        else:
            print("[Warn] `test` not found in data.yaml; overlay generation may be incomplete.")

        label_lut = _build_label_lookup(out_label_dir)
        if args.color_mode == "class":
            num_classes = len(names) if names else 80
            class_palette = _default_class_palette(num_classes)
        else:
            class_palette = None

        stats = {
            "no_label_txt": 0,
            "invalid_or_box_only": 0,
            "too_few_points": 0,
            "images_no_valid_poly": 0,
            "img_read_fail": 0,
        }

        for img_p in img_list:
            lbl_p = label_lut.get(img_p.stem)
            _make_overlay_for_image(
                img_path=img_p,
                label_path=lbl_p,
                overlay_dir=overlay_dir,
                color_mode=args.color_mode,
                alpha=float(args.alpha),
                class_palette=class_palette,
                stats=stats
            )

        print(f"[Overlay] Done. Wrote overlays to: {overlay_dir}")
        print(f"[Overlay] Stats: missing label txt = {stats['no_label_txt']}, "
              f"invalid_or_box_only lines = {stats['invalid_or_box_only']}, "
              f"too_few_points = {stats['too_few_points']}, "
              f"images with no valid polygon = {stats['images_no_valid_poly']}, "
              f"img read fail = {stats['img_read_fail']}")

    print("\n[Done] Subset evaluation completed.")

if __name__ == "__main__":
    main()
