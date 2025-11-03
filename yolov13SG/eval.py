#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a pre-split subset (with its own data.yaml where test: points to subset).
No index range needed — just pass the subset's data.yaml.

Usage:
python eval_subset.py \
  --weights runs/detect/SRYOLO_coffee4/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --project runs/test_eval --name part_001 \
  --imgsz 640 --batch 4 --conf 0.001 --iou 0.6 \
  --out-label-dir pred_labels_part_001
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def _resolve(p: Path) -> Path:
    return Path(p).expanduser().resolve()

def _write_yaml(obj: dict, p: Path):
    p = _resolve(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

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
    args = ap.parse_args()

    weights = _resolve(Path(args.weights))
    data_yaml = _resolve(Path(args.data))
    project = _resolve(Path(args.project))
    out_label_dir = _resolve(Path(args.out_label_dir))

    # 读一下 data.yaml，给出提示（可选）
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        print("[Info] 子集 data.yaml 加载成功。test 指向：", d.get("test"))
    except Exception as e:
        print("[Warn] 子集 data.yaml 读取失败，但继续评估：", e)

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

    # 兼容不同版本字段
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

    print("\n[Done] Subset evaluation completed.")

if __name__ == "__main__":
    main()
