#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a pre-split subset (YOLO/RT-DETR, Ultralytics API).
- Works with YOLO(.pt) or RT-DETR(.pt) weights from ultralytics.
- The subset's data.yaml should have test: pointing to that subset.

Usage (YOLO or RT-DETR are both fine):
python eval_subset_ultra.py \
  --weights runs/detect/SRYOLO_coffee4/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --project runs/test_eval --name part_001 \
  --imgsz 640 --batch 4 --conf 0.001 --iou 0.6 \
  --out-label-dir pred_labels_part_001 \
  --engine auto  # auto|yolo|rtdetr
"""

import os
import re
import shutil
import argparse
from pathlib import Path
import yaml

# --- ultralytics imports (YOLO + RTDETR) ---
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    # Ultralytics >=8.1 提供 RTDETR 类
    from ultralytics import RTDETR
except Exception:
    RTDETR = None


def _resolve(p: Path) -> Path:
    return Path(p).expanduser().resolve()


def _write_yaml(obj: dict, p: Path):
    p = _resolve(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _infer_engine_from_weights(weights_path: Path) -> str:
    """
    粗略根据权重名推断：含 'rtdetr' 优先认为是 RT-DETR；否则默认 YOLO。
    仅在 --engine=auto 时生效。
    """
    n = weights_path.name.lower()
    if "rtdetr" in n or re.search(r"\brt[-_]?detr\b", n):
        return "rtdetr"
    return "yolo"


def _load_model(engine: str, weights: Path, task: str = "detect"):
    """
    根据 engine 加载模型：
      - 'yolo'  : YOLO(...)
      - 'rtdetr': RTDETR(...)
      - 'auto'  : 按权重名推断
    """
    eng = engine
    if engine == "auto":
        eng = _infer_engine_from_weights(weights)

    if eng == "rtdetr":
        if RTDETR is None:
            raise RuntimeError("ultralytics 未提供 RTDETR 类，或版本过旧。请升级：pip install -U ultralytics")
        print(f"[Info] Engine=RT-DETR  | weights={weights}")
        return RTDETR(str(weights))
    elif eng == "yolo":
        if YOLO is None:
            raise RuntimeError("未找到 YOLO 类。请安装 ultralytics：pip install -U ultralytics")
        print(f"[Info] Engine=YOLO     | weights={weights}")
        return YOLO(str(weights))
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def _extract_metrics(results):
    """
    兼容不同 ultralytics 版本，尽可能抽取常见指标。
    """
    md = {}
    # v8 常见：results.metrics 或 results.results_dict
    for path in ("results_dict",):
        if hasattr(results, path):
            try:
                d = getattr(results, path)
                if isinstance(d, dict) and d:
                    md.update(d)
            except Exception:
                pass

    if not md and hasattr(results, "metrics"):
        m = results.metrics
        # 这些属性在不同版本上有不同命名，逐个兼容
        def _get(obj, *names):
            for nm in names:
                if hasattr(obj, nm):
                    return getattr(obj, nm)
            return None

        md = {
            "mAP50-95": _get(m, "map", "mAP50_95", "mAP"),
            "mAP50"   : _get(m, "map50", "mAP50"),
            "mAP75"   : _get(m, "map75", "mAP75"),
            "P"       : _get(m, "mp", "precision"),
            "R"       : _get(m, "mr", "recall"),
            "cls-P"   : getattr(m, "ap_class_index", None),  # optional
        }
    return md


def _find_labels_dir(run_dir: Path) -> Path | None:
    """
    在一次 val 运行目录下寻找预测 labels 的常见位置。
    典型：<project>/<name>/labels
    某些版本可能放在 predictions/labels、preds/labels 等，做多路径兜底。
    """
    candidates = [
        run_dir / "labels",
        run_dir / "predictions" / "labels",
        run_dir / "preds" / "labels",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def main():
    ap = argparse.ArgumentParser("Evaluate one subset (YOLO/RT-DETR; data.yaml with test pointing to subset)")
    ap.add_argument("--weights", required=True, type=str, help="Path to .pt weights")
    ap.add_argument("--data", required=True, type=str, help="Path to the subset's data.yaml")
    ap.add_argument("--project", type=str, default="runs/test_eval", help="Ultralytics project dir")
    ap.add_argument("--name", type=str, default="subset_eval", help="Ultralytics run name")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--out-label-dir", type=str, default="pred_labels_subset",
                    help="Where to collect predicted labels (txt)")
    ap.add_argument("--no-clear-out", action="store_true",
                    help="Do not clear OUT_LABEL_DIR before copy")
    ap.add_argument("--engine", type=str, default="auto", choices=["auto", "yolo", "rtdetr"],
                    help="Which model loader to use (auto infer from weights name)")
    ap.add_argument("--task", type=str, default="detect", choices=["detect", "pose", "seg"],
                    help="Ultralytics task (mostly 'detect')")
    args = ap.parse_args()

    weights = _resolve(Path(args.weights))
    data_yaml = _resolve(Path(args.data))
    project = _resolve(Path(args.project))
    out_label_dir = _resolve(Path(args.out_label_dir))

    # 读一下 data.yaml，给出提示（可选）
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        print("[Info] 子集 data.yaml 加载成功。test 指向：", d.get("test"))
    except Exception as e:
        print("[Warn] 子集 data.yaml 读取失败，但继续评估：", e)

    # 加载模型（兼容 YOLO / RT-DETR）
    model = _load_model(args.engine, weights, task=args.task)

    # 统一调用 val（Ultralytics 对 YOLO/RT-DETR 的 val 接口一致）
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        save_txt=True,       # 关键：保存 txt labels
        save_conf=True,
        project=str(project),
        name=args.name,
        verbose=True,
    )

    # 抽取指标（尽量兼容不同版本）
    metrics_dict = _extract_metrics(results)
    print("\n===== Evaluation Metrics (this subset) =====")
    if metrics_dict:
        for k, v in metrics_dict.items():
            print(f"{k}: {v}")
    else:
        print("(no metrics parsed; ultralytics version fields may differ)")

    # 收集预测 labels（txt）
    run_dir = project / args.name
    pred_label_src = _find_labels_dir(run_dir) or (run_dir / "labels")
    print(f"\nPredicted label txt folder (guess): {pred_label_src}")

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
        print("WARNING: labels folder not found. Ensure save_txt=True and split='test' worked; "
              "check the run dir for where labels are saved.")

    print("\n[Done] Subset evaluation completed.")


if __name__ == "__main__":
    main()
