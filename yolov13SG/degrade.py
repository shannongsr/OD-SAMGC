#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build 9 degraded YOLO datasets from a base dataset:
- Fog (beta):   0.05, 0.10, 0.15
- Rain (density): 200, 400, 600
- Dark (gamma): 0.8, 0.6, 0.4

Directory structure expected for base dataset (MERGED):
MERGED/
  images/{train,val,test}/*.jpg|png|...
  labels/{train,val,test}/*.txt
  (optional) data.yaml  # used to inherit nc/names

Each degraded dataset will be written under OUT_ROOT with name like:
  <base_name>_fog_light/
  <base_name>_rain_heavy/
  <base_name>_dark_medium/

Usage:
    python build_degraded_datasets.py \
        --input "/content/drive/MyDrive/yolov13-coffeev2/merged_without_part_003" \
        --out-root "/content/drive/MyDrive/yolov13-coffeev2/degraded_sets"

Requirements:
    pip install opencv-python numpy pyyaml
"""

import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from math import cos, sin, radians

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------------
# I/O utils
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(d: Path):
    if not d.is_dir():
        return []
    return [
        p for p in sorted(d.iterdir(), key=lambda x: x.name.lower())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

def imread_any(path: Path):
    # Robust to non-ascii paths (Windows)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def imwrite_any(path: Path, img):
    ensure_dir(path.parent)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(str(path))

def copy_or_empty_label(src_label: Path, dst_label: Path):
    ensure_dir(dst_label.parent)
    if src_label.is_file():
        shutil.copy2(src_label, dst_label)
    else:
        # Create empty label file to be safe (some tools expect it)
        dst_label.write_text("", encoding="utf-8")

def read_label_ids(lbl: Path):
    ids = set()
    if not lbl.is_file():
        return ids
    try:
        for line in lbl.read_text(encoding="utf-8").splitlines():
            tok = line.strip().split()
            if tok and tok[0].lstrip("-").isdigit():
                ids.add(int(tok[0]))
    except Exception:
        pass
    return ids

# -----------------------------
# Dark (gamma) degradation
# -----------------------------
def apply_gamma(img_bgr, gamma: float):
    inv = 1.0 / max(gamma, 1e-6)  # gamma<1 -> darker
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((x ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)

# -----------------------------
# Fog degradation (approx. scattering)
# I = J*t + A*(1-t),  t = exp(-beta*depth)
# depth ≈ vertical gradient + low-freq noise
# -----------------------------
def smooth_noise(h, w, strength=0.5, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.random((h, w)).astype(np.float32)
    sigma = max(h, w) * 0.03
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    noise = cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)
    return noise * strength

def apply_fog(img_bgr, beta: float, airlight=1.0, seed=None):
    h, w = img_bgr.shape[:2]
    J = img_bgr.astype(np.float32) / 255.0
    y_grad = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    y_grad = np.repeat(y_grad, w, axis=1)
    noise = smooth_noise(h, w, strength=0.5, seed=seed)
    depth = cv2.normalize(0.6 * y_grad + 0.4 * noise, None, 0, 1, cv2.NORM_MINMAX)
    t = np.exp(-beta * depth)
    t3 = np.dstack([t, t, t])
    I = J * t3 + airlight * (1.0 - t3)
    return np.clip(I * 255.0, 0, 255).astype(np.uint8)

# -----------------------------
# Rain degradation
# -----------------------------
def make_motion_kernel(ksize: int, angle_deg: float):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cv2.line(kernel, (ksize//2, 0), (ksize//2, ksize-1), 1.0, 1)
    M = cv2.getRotationMatrix2D((ksize/2 - 0.5, ksize/2 - 0.5), angle_deg, 1.0)
    rotated = cv2.warpAffine(kernel, M, (ksize, ksize))
    rotated /= (rotated.sum() + 1e-6)
    return rotated

def apply_rain(img_bgr, density: int, angle_deg: float=-25.0, length_px=(15, 35),
               thickness=(1, 2), motion_ksize: int=15, rain_alpha: float=0.3, seed=None):
    h, w = img_bgr.shape[:2]
    rng = np.random.default_rng(seed)
    rain = np.zeros((h, w), dtype=np.uint8)

    for _ in range(int(density)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        ln = int(rng.integers(length_px[0], length_px[1]))
        th = int(rng.integers(thickness[0], thickness[1] + 1))
        a = radians(angle_deg)
        dx = int(ln * cos(a))
        dy = int(ln * sin(a))
        x2 = np.clip(x + dx, 0, w - 1)
        y2 = np.clip(y + dy, 0, h - 1)
        cv2.line(rain, (x, y), (x2, y2), 255, th)

    k = make_motion_kernel(motion_ksize, angle_deg)
    rain_blur = cv2.filter2D(rain.astype(np.float32) / 255.0, -1, k)

    base = img_bgr.astype(np.float32) / 255.0
    rain3 = np.dstack([rain_blur]*3)
    out = np.clip(base + rain_alpha * rain3, 0, 1)
    return (out * 255.0).astype(np.uint8)

# -----------------------------
# Dataset build helpers
# -----------------------------
def inherit_nc_names(base_root: Path):
    data_yaml = base_root / "data.yaml"
    if data_yaml.is_file():
        try:
            y = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            nc = int(y.get("nc")) if "nc" in y else None
            names = y.get("names")
            if isinstance(names, list) and nc is not None and len(names) == nc:
                return nc, names
        except Exception:
            pass
    # Fallback: infer from labels
    all_ids = set()
    for split in ["train", "val", "valid", "test"]:
        lbl_dir = base_root / "labels" / ("valid" if split == "val" else split)
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            all_ids |= read_label_ids(lbl)
    if all_ids:
        nc = max(all_ids) + 1
    else:
        nc = 1
    names = [str(i) for i in range(nc)]
    return nc, names

def write_data_yaml(root: Path, nc: int, names: list):
    y = {
        "train": str((root / "images" / "train").resolve()),
        "val":   str((root / "images" / "valid").resolve()),
        "test":  str((root / "images" / "test").resolve()),
        "nc": int(nc),
        "names": list(names)
    }
    (root / "data.yaml").write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False), encoding="utf-8")

def ensure_non_empty_train_val(ds_root: Path):
    """
    If train or valid has no images, copy one sample from test to fill it.
    """
    img_train = ds_root / "images" / "train"
    img_valid = ds_root / "images" / "valid"
    img_test  = ds_root / "images" / "test"
    lbl_train = ds_root / "labels" / "train"
    lbl_valid = ds_root / "labels" / "valid"
    lbl_test  = ds_root / "labels" / "test"

    ensure_dir(img_train); ensure_dir(img_valid)
    ensure_dir(lbl_train); ensure_dir(lbl_valid)

    def copy_one_from_test(dst_img_dir: Path, dst_lbl_dir: Path):
        test_imgs = list_images(img_test)
        if not test_imgs:
            return
        src_img = test_imgs[0]
        dst_img = dst_img_dir / src_img.name
        shutil.copy2(src_img, dst_img)
        src_lbl = lbl_test / (src_img.stem + ".txt")
        dst_lbl = dst_lbl_dir / (src_img.stem + ".txt")
        copy_or_empty_label(src_lbl, dst_lbl)

    if len(list_images(img_train)) == 0:
        copy_one_from_test(img_train, lbl_train)
    if len(list_images(img_valid)) == 0:
        copy_one_from_test(img_valid, lbl_valid)

def process_split(base_root: Path, out_root: Path, split: str, op_name: str, op_fn):
    """
    Apply op_fn to all images under split; copy labels.
    """
    split_std = "valid" if split == "val" else split  # accept 'val' alias
    in_img_dir = base_root / "images" / split_std
    in_lbl_dir = base_root / "labels" / split_std
    out_img_dir = out_root / "images" / split_std
    out_lbl_dir = out_root / "labels" / split_std
    ensure_dir(out_img_dir); ensure_dir(out_lbl_dir)

    imgs = list_images(in_img_dir)
    for img_path in imgs:
        img = imread_any(img_path)
        if img is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue
        out_img = op_fn(img)
        out_path = out_img_dir / img_path.name
        imwrite_any(out_path, out_img)

        lbl_in  = in_lbl_dir / f"{img_path.stem}.txt"
        lbl_out = out_lbl_dir / f"{img_path.stem}.txt"
        copy_or_empty_label(lbl_in, lbl_out)

    print(f"[{op_name}] {split_std}: {len(imgs)} images processed.")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Base dataset root, e.g. merged_without_part_003")
    parser.add_argument("--out-root", type=str, required=True,
                        help="Where to write the 9 degraded datasets")
    parser.add_argument("--base-name", type=str, default=None,
                        help="Base name prefix for datasets; default is input folder name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_root = Path(args.input)
    out_root  = Path(args.out_root)
    ensure_dir(out_root)

    base_name = args.base_name or base_root.name
    nc, names = inherit_nc_names(base_root)

    # Define 9 variants
    fog_levels  = [("light", 1), ("medium", 1.5), ("heavy", 2)]
    rain_levels = [("light", 300),  ("medium", 600),  ("heavy", 900)]
    dark_levels = [("light", 0.6),  ("medium", 0.45),  ("heavy", 0.3)]

    rng_seed = args.seed

    # Build fog datasets
    for sev_name, beta in fog_levels:
        ds_name = f"{base_name}_fog_{sev_name}"
        ds_root = out_root / ds_name
        print(f"\n=== Building {ds_name} (beta={beta}) ===")
        # Define operator
        def op(img, b=beta, s=rng_seed):
            return apply_fog(img, beta=b, airlight=1.0, seed=s)
        for split in ["train", "valid", "test"]:
            process_split(base_root, ds_root, split, f"fog_{sev_name}", op)
        ensure_non_empty_train_val(ds_root)
        write_data_yaml(ds_root, nc, names)

    # Build rain datasets
    for sev_name, dens in rain_levels:
        ds_name = f"{base_name}_rain_{sev_name}"
        ds_root = out_root / ds_name
        print(f"\n=== Building {ds_name} (density={dens}) ===")
        def op(img, d=dens, s=rng_seed):
            return apply_rain(img, density=d, angle_deg=-25.0, seed=s)
        for split in ["train", "valid", "test"]:
            process_split(base_root, ds_root, split, f"rain_{sev_name}", op)
        ensure_non_empty_train_val(ds_root)
        write_data_yaml(ds_root, nc, names)

    # Build dark (gamma) datasets
    for sev_name, gamma in dark_levels:
        ds_name = f"{base_name}_dark_{sev_name}"
        ds_root = out_root / ds_name
        print(f"\n=== Building {ds_name} (gamma={gamma}) ===")
        def op(img, g=gamma):
            return apply_gamma(img, gamma=g)
        for split in ["train", "valid", "test"]:
            process_split(base_root, ds_root, split, f"dark_{sev_name}", op)
        ensure_non_empty_train_val(ds_root)
        write_data_yaml(ds_root, nc, names)

    print("\nAll degraded datasets are built under:", out_root.resolve())

if __name__ == "__main__":
    main()
