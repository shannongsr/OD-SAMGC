#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-image inference (YOLO / RT-DETR via Ultralytics).
Given a weights file and an image path, run inference and save an annotated image.

Usage:
python infer_one.py \
  --weights runs/detect/SRYOLO_coffee4/weights/best.pt \
  --image /path/to/your.jpg \
  --out out_dir/annot.png \
  --engine auto \
  --imgsz 640 --conf 0.25 --iou 0.6 --device 0
"""

import argparse
from pathlib import Path
import re

# ultralytics
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from ultralytics import RTDETR
except Exception:
    RTDETR = None


def _resolve(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _infer_engine_from_weights(weights_path: Path) -> str:
    n = weights_path.name.lower()
    return "rtdetr" if ("rtdetr" in n or re.search(r"\brt[-_]?detr\b", n)) else "yolo"


def _load_model(engine: str, weights: Path):
    if engine == "auto":
        engine = _infer_engine_from_weights(weights)
    if engine == "rtdetr":
        if RTDETR is None:
            raise RuntimeError("未检测到 RTDETR 类，请升级 ultralytics：pip install -U ultralytics")
        print(f"[Info] Engine=RT-DETR | weights={weights}")
        return RTDETR(str(weights))
    elif engine == "yolo":
        if YOLO is None:
            raise RuntimeError("未检测到 YOLO 类，请安装/升级 ultralytics：pip install -U ultralytics")
        print(f"[Info] Engine=YOLO    | weights={weights}")
        return YOLO(str(weights))
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def main():
    ap = argparse.ArgumentParser("Single image inference & save annotated result")
    ap.add_argument("--weights", required=True, type=str, help="Path to .pt weights (YOLO/RT-DETR)")
    ap.add_argument("--image", required=True, type=str, help="Path to a single image")
    ap.add_argument("--out", type=str, default="runs/infer_one/annot.png", help="Output annotated image path")
    ap.add_argument("--engine", type=str, default="auto", choices=["auto", "yolo", "rtdetr"],
                    help="auto=根据权重名推断; 也可强制指定")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    ap.add_argument("--device", type=str, default=None, help="device id like 0 or 'cpu'")
    args = ap.parse_args()

    weights = _resolve(args.weights)
    image = _resolve(args.image)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.engine, weights)

    # 统一的 Ultralytics 推理接口：predict
    # 返回一个 Results 列表（单张图 -> len==1）
    results = model.predict(
        source=str(image),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=True,
        save=False,       # 不使用内部保存，改为手动保存到 --out
        boxes=True
    )

    if not results:
        raise RuntimeError("No results returned from model.predict")

    r0 = results[0]

    # r0.plot(): 返回带框标注的 BGR ndarray（适配不同版本）
    annot = r0.plot()

    # 用 OpenCV 保存（避免额外依赖，这里做个兜底）
    try:
        import cv2
        cv2.imwrite(str(out_path), annot)
    except Exception:
        # 兜底用 PIL 保存（需要转换 BGR->RGB）
        try:
            from PIL import Image
            import numpy as np
            rgb = annot[..., ::-1]  # BGR->RGB
            Image.fromarray(rgb).save(str(out_path))
        except Exception as e:
            raise RuntimeError(f"Failed to save annotated image: {e}")

    print(f"[Done] Saved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
