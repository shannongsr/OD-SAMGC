# OD-SAMGC

This repository contains two main components:

1. **SAM+GrabCut Segmentation Pipeline (SAMGC)** — a set of advanced SAM2 prompting and mask‑refinement scripts.
2. **YOLOv13SG** — a modified YOLOv13 model configuration supporting the proposed SG (Structure‑Guided) variant.

---

## 📁 Repository Structure

```
OD-SAMGC/
│
├── SAM+GrabCut/          # SAMGC scripts (prompting, refinement, evaluation)
├── box_and_point/        # Additional prompting utilities
├── yolov13SG/            # YOLOv13SG model config (yaml)
│
├── LICENSE
└── README.md
```

---

# 1. SAMGC (SAM + GrabCut + Prompting Engine)

SAMGC is a **mask‑refinement framework built on top of Meta-SAM2**, designed to take detection boxes (e.g., YOLO outputs) and generate high‑quality segmentation masks.

It includes:

### ✔ Multiple Prompting Modes
Provided by `prompt_method.py`:
- `box` — pure SAM2 box prompting  
- `box+center` — add a center point to stabilize segmentation  
- `point_center` — point‑only SAM inference  
- `box+kmeans_centroid` — use KMeans cluster centroid as point prompt  
- `box+hsv_cc_centroid` — use HSV connected‑component centroid  
- `box+hsv_cc_centroid+refine` — centroid + morphology  
- **GrabCut variants**  
  - `box+grabcut`
  - `box+center+grabcut`
  - `box+hsv_cc_centroid+grabcut`
- **Scale-aware erosion prompting**  
  - `box+scale_erode`, `box+size_erode`, `box+scale`

### ✔ Mask Refinement Modules
Implemented inside `prompt_method.py`:
- Morphological refinement  
- HSV prior filtering  
- GrabCut two‑stage refinement  
- Scale‑aware erosion (small/medium/large masks processed differently)

### ✔ COCO‑style Evaluation Engine
`s`
- Computes:
  - AP50 / AP75 / mAP  
  - COCO 101‑point AP  
  - Small / Medium / Large instance AP  
  - Boundary‑F1  
  - Mask‑IoU (union‑based)

---

# 2. How to Use SAMGC

## Step 1. Prepare environment

Clone SAM2 official repository:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy SAMGC scripts into the **main directory of SAM2**:

```
segment-anything-2/
│
├── sam2_seg_evaluator.py   ← copy from this repo
├── run_eval_sam2.py        ← copy from this repo
├── prompt_method.py        ← copy from this repo
└── ...
```

## Step 2. Prepare input directories

- `--image-dir` : directory of testing images  
- `--gt-label-dir` : YOLO polygon labels (`cls x1 y1 x2 y2 ...`)  
- `--pred-bbox-dir` : YOLO detection results (`cls conf cx cy w h`)  

Example:

```
dataset/
    images/
    labels/
detections/
```

## Step 3. Run evaluation

```bash
python run_eval_sam2.py   --image-dir dataset/images   --gt-label-dir dataset/labels   --pred-bbox-dir detections   --result-dir results_samgc   --ckpt sam2.1_hiera_tiny.pt   --model-cfg sam2.1/sam2.1_hiera_t   --prompt-mode box+grabcut   --bf-tol 2
```

### Optional parameters
- `--max-eval N` : limit number of images  
- `--prompt-param '{}'` : JSON tuning params  
- `--passed-set-dir path` : copy evaluated images  
- `--device cpu/cuda`  

### Output
A `summary.json`:

```
{
  "mAP": 0.789,
  "AP50": 0.901,
  "AP75": 0.812,
  "AP_small": 0.602,
  "BF1": 0.874,
  ...
}
```

---

# 3. YOLOv13SG

Located in:

```
yolov13SG/
    YOLOv13SG.yaml
```

This folder contains:
- A **modified YOLOv13 model configuration** integrating:
  - Structure‑Guided design  
  - Additional feature enhancement modules  

## Training YOLOv13SG

Simply run:

```bash
yolo train   model=yolov13SG/YOLOv13SG.yaml   data=your_dataset.yaml   epochs=300   imgsz=640
```

YOLOv13SG can also be used as a detection backbone before running SAMGC.

---

# 4. Citation

If you use this repository or the SAMGC pipeline, please cite our paper (to be updated).

---

# 5. License

See `LICENSE` for details.

