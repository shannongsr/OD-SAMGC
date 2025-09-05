#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from sam2_seg_evaluator import SAM2SegEvaluator, ensure_dir

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SAM2 (macro & COCO-style metrics) on YOLO-style data with three prompt modes."
    )
    # 基本路径
    p.add_argument("--image-dir", required=True, help="Directory of images")
    p.add_argument("--gt-label-dir", required=True, help="Directory of GT polygon labels (YOLO-style polygons)")
    p.add_argument("--pred-bbox-dir", required=True, help="Directory of predicted det bboxes (YOLO-style boxes)")
    p.add_argument("--result-dir", required=True, help="Directory to save results (summary.json)")

    # 模型
    p.add_argument("--model-cfg", default="sam2.1/sam2.1_hiera_t", help="SAM2 config name")
    p.add_argument("--ckpt", required=True, help="Checkpoint path of SAM2")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-eval", type=int, default=None, help="Limit number of images to evaluate")
    p.add_argument("--class-agnostic", action="store_true", help="Ignore class when matching masks")

    # 提示与评估设置（仅三种）
    p.add_argument(
        "--prompt-mode",
        default="box",
        choices=[
            "box",          # 仅 box
            "box+center",   # box + 中心点
            "point_center", # 仅中心点（不传 box）
        ],
        help="Prompting strategy for SAM2"
    )
    p.add_argument("--bf-tol", type=int, default=2, help="Boundary F1 tolerance in pixels")

    # 仅保留通用 prompt 参数（JSON，可选）
    p.add_argument("--prompt-param", type=str, default="{}", help="Prompt params (JSON)")

    # 运行控制
    p.add_argument("--check-mode", action="store_true", help="Print diagnostics (may skip on timeout)")
    p.add_argument("--timeout-sec", type=float, default=0.0, help="Per-image timeout; >0 enables skipping")
    p.add_argument("--passed-set-dir", type=str, default=None, help="If set, copy passed images/labels here")

    return p.parse_args()

def _loads_json(s: str, name: str):
    try:
        return json.loads(s) if s else {}
    except Exception as e:
        print(f"[warn] {name} JSON 解析失败，已忽略。错误：{e}")
        return {}

def main():
    args = parse_args()
    ensure_dir(args.result_dir)

    # ---- 仅保留基础 prompt 参数 ----
    prompt_param = _loads_json(args.prompt_param, "prompt-param")

    # ---- 创建评估器（接口不变，去除无关参数）----
    evaluator = SAM2SegEvaluator(
        image_dir=args.image_dir,
        gt_label_dir=args.gt_label_dir,
        pred_bbox_dir=args.pred_bbox_dir,
        result_dir=args.result_dir,
        model_cfg_name=args.model_cfg,
        sam2_ckpt_path=args.ckpt,
        device=args.device,
        class_agnostic=args.class_agnostic,
        max_eval=args.max_eval,
        prompt_mode=args.prompt_mode,
        bf_tolerance_pixels=args.bf_tol,
        prompt_param=prompt_param,
        check_mode=args.check_mode,
        timeout_sec=args.timeout_sec,
        passed_set_dir=args.passed_set_dir,
    )

    summary = evaluator.evaluate()
    out_path = os.path.join(args.result_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()
