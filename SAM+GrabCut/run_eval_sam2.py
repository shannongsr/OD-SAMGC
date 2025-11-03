#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from sam2_seg_evaluator import SAM2SegEvaluator, ensure_dir

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SAM2 (macro & COCO-style metrics) with different prompt methods on YOLO-style data."
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

    # 提示与评估设置
    p.add_argument(
        "--prompt-mode",
        default="box",
        choices=[
            "box", "box+center", "point_center",
            "box+hsv_cc_centroid", "box+hsv_cc_centroid+refine",
            "box+kmeans_centroid", "box+hsv_cc_centroid+fast_correct",
            "box+grabcut", "box+center+grabcut", "box+hsv_cc_centroid+grabcut",
            # —— 新增的按尺度腐蚀 ——（与工厂函数保持一致的别名）
            "box+scale_erode", "box+size_erode", "box+scale",
        ],
        help="Prompting strategy for SAM2"
    )
    p.add_argument("--bf-tol", type=int, default=2, help="Boundary F1 tolerance in pixels")

    # JSON 参数（按需合并）
    p.add_argument("--prompt-param", type=str, default="{}", help="Base prompt params (JSON)")
    p.add_argument("--fast-correct-param", type=str, default="{}", help="Params for fast_correct (JSON)")
    p.add_argument("--hsv-cc-param", type=str, default="{}", help="Params for hsv_cc_centroid (JSON)")
    # —— 新增：按尺度腐蚀参数 ——（仅在 *scale* 模式下会被并入）
    p.add_argument("--scale-erode-param", type=str, default="{}", help="Params for scale-aware erosion (JSON)")

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

    # ---- 合并 prompt 参数（分模块、可叠加）----
    base_param        = _loads_json(args.prompt_param, "prompt-param")
    fast_param        = _loads_json(args.fast_correct_param, "fast-correct-param")
    hsvcc_param       = _loads_json(args.hsv_cc_param, "hsv-cc-param")
    scale_erode_param = _loads_json(args.scale_erode_param, "scale-erode-param")

    prompt_param = {}
    # 通用基础参数先入
    prompt_param.update(base_param)
    # fast_correct 相关（仅在含有 fast_correct 的模式下）
    if "fast_correct" in args.prompt_mode or args.prompt_mode.endswith("+fast"):
        prompt_param.update(fast_param)
    # HSV 连通域质心相关（仅在含有 hsv_cc_centroid 的模式下）
    if "hsv_cc_centroid" in args.prompt_mode or args.prompt_mode in ("box+hsv", "box+hsv+refine", "box+hsv+fast"):
        prompt_param.update(hsvcc_param)
    # 按尺度腐蚀（仅在 *scale* 模式下）
    if args.prompt_mode in ("box+scale_erode", "box+size_erode", "box+scale"):
        prompt_param.update(scale_erode_param)

    # ---- 创建评估器 ----
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
