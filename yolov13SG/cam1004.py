import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os, shutil, cv2, torch
import numpy as np
from PIL import Image
from tqdm import trange

from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

np.random.seed(0)

# -------------------------- 工具 --------------------------
def letterbox(im, new_shape=(640, 640), color=(114,114,114), auto=False, scaleFill=True, scaleup=True, stride=32):
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right  = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

# === 自适应坐标解码：统一返回像素 xyxy（用于预测回退方案） ===
def _decode_boxes_auto(boxes: torch.Tensor, shape_hw):
    H, W = shape_hw
    b = boxes.clone()
    if b.numel() == 0:
        return b

    mx = float(b.max().detach().cpu())
    if mx <= 1.5:
        b = xywh2xyxy(b)
        b[:, [0,2]] *= W
        b[:, [1,3]] *= H
    else:
        b_try = xywh2xyxy(b)
        frac_bad = ((b_try[:, 2] < b_try[:, 0]) | (b_try[:, 3] < b_try[:, 1])).float().mean().item() if b_try.numel() else 0.0
        b = b if frac_bad < 0.1 else b_try

    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H - 1)
    return b

def mask_cam_by_bboxes(grayscale_cam: np.ndarray, boxes_xyxy: np.ndarray):
    """将 CAM 在所有 bbox 的并集内保留，框外全部置零；不做任何归一化。"""
    H, W = grayscale_cam.shape
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return grayscale_cam * 0.0
    mask = np.zeros((H, W), dtype=np.float32)
    for x1, y1, x2, y2 in boxes_xyxy.astype(np.int64):
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return grayscale_cam * mask

def imgpath_to_labelpath(img_path: str) -> str:
    """
    将 .../images/<split>/name.jpg -> .../labels/<split>/name.txt
    若目录名不是 images/labels，可按需修改这段。
    """
    dirpath, fname = os.path.split(img_path)
    # 将路径段中的 'images' 替换为 'labels'
    parts = dirpath.split(os.sep)
    for i, p in enumerate(parts):
        if p.lower() == 'images':
            parts[i] = 'labels'
            break
    label_dir = os.sep.join(parts)
    stem = os.path.splitext(fname)[0]
    return os.path.join(label_dir, stem + '.txt')

def load_seg_bboxes_from_label(label_path: str, img_size: int) -> np.ndarray:
    """
    读取 YOLO 分割标签，格式：cls x1 y1 x2 y2 ...（点为归一化坐标）。
    以每行多边形的最小外接矩形作为 bbox（像素坐标，基于 scaleFill 到 img_size×img_size）。
    返回形状 (N,4) 的 np.array，列为 [x1,y1,x2,y2]。
    """
    if not os.path.isfile(label_path):
        return np.zeros((0, 4), dtype=np.float32)

    bboxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            if len(toks) < 6:
                continue
            # 第一个是类别，其余为成对的 x y
            try:
                coords = list(map(float, toks[1:]))
            except ValueError:
                continue
            if len(coords) % 2 != 0:
                coords = coords[:-1]  # 容错：丢弃尾部孤立值
            if len(coords) < 6:
                continue
            xs = np.array(coords[0::2], dtype=np.float32) * img_size
            ys = np.array(coords[1::2], dtype=np.float32) * img_size
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
            # 防止退化
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])

    if not bboxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(bboxes, dtype=np.float32)

# -------------------------- 反传目标 --------------------------
class YoloV8Target(torch.nn.Module):
    def __init__(self, output_type='all', conf=0.25, top_ratio=1.0):
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.top_ratio = top_ratio

    def forward(self, model_outputs):
        out = model_outputs
        if isinstance(out, (list, tuple)):
            out = out[0]
        preds = out[0] if out.dim() == 3 else out  # (N, 5+nc)

        if preds.ndim != 2 or preds.size(1) < 6:
            return preds.sum() * 0.0

        boxes = preds[:, 0:4]
        obj = preds[:, 4:5]
        cls = preds[:, 5:]
        cls_max, _ = cls.max(dim=1, keepdim=True)
        score = (obj * cls_max).squeeze(1)

        keep = score > self.conf
        if keep.sum() == 0:
            return (score.sum() + cls_max.sum())

        score_sorted, idx = torch.sort(score[keep], descending=True)
        k = max(1, int(len(score_sorted) * self.top_ratio))
        idx_sel = keep.nonzero(as_tuple=False).squeeze(1)[idx[:k]]

        terms = []
        if self.output_type in ('class', 'all'):
            terms.append(cls_max[idx_sel].sum())
        if self.output_type in ('box', 'all'):
            terms.append(boxes[idx_sel].sum())
        return sum(terms) if len(terms) else score_sorted[:k].sum()

# -------------------------- Grad-CAM 执行类 --------------------------
class YoloV8Heatmap:
    def __init__(self, weight, device, method, layer_idx_list, backward_type, conf_threshold, ratio):
        self.device = torch.device(device)
        self.yolo = YOLO(weight)
        self.model = self.yolo.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.names = self.yolo.names if hasattr(self.yolo, 'names') else {i: str(i) for i in range(1000)}
        self.target_layers = [self.model.model[i] for i in layer_idx_list]

        cam_cls = eval(method)
        self.cam = cam_cls(self.model, target_layers=self.target_layers)

        self.target = YoloV8Target(output_type=backward_type, conf=conf_threshold, top_ratio=ratio)
        self.conf_threshold = conf_threshold

    @torch.no_grad()
    def forward_raw(self, tensor):
        out = self.model(tensor)[0] if isinstance(self.model(tensor), (list, tuple)) else self.model(tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out  # (B, N, 5+nc)

    def post_process(self, pred, shape_hw):
        H, W = shape_hw
        if pred.ndim != 2 or pred.size(1) < 6:
            return np.zeros((0,6), dtype=np.float32)

        boxes = pred[:, 0:4].clone()
        obj = pred[:, 4:5]
        cls = pred[:, 5:]
        cls_max, cls_id = cls.max(dim=1, keepdim=True)
        score = (obj * cls_max).squeeze(1)

        keep = score > self.conf_threshold
        if keep.sum() == 0:
            return np.zeros((0,6), dtype=np.float32)

        boxes = boxes[keep]
        cls_id = cls_id[keep].squeeze(1)
        score = score[keep]

        boxes_xyxy = _decode_boxes_auto(boxes, (H, W))
        out = torch.cat([boxes_xyxy, score[:, None], cls_id[:, None].float()], dim=1).cpu().numpy()
        return out

    def _fallback_top1_det(self, raw_pred, shape_hw):
        """在 det 为空时，从 raw_pred 中取 top1 置信度做兜底框。"""
        H, W = shape_hw
        if raw_pred.ndim != 2 or raw_pred.size(1) < 6 or raw_pred.shape[0] == 0:
            return np.zeros((0,6), dtype=np.float32)
        boxes = raw_pred[:, :4]
        obj = raw_pred[:, 4:5]
        cls = raw_pred[:, 5:]
        cls_max, cls_id = cls.max(dim=1, keepdim=True)
        score = (obj * cls_max).squeeze(1)
        i = int(torch.argmax(score))
        boxes_xyxy = _decode_boxes_auto(boxes[[i]], (H, W))
        det = torch.cat([boxes_xyxy, score[[i]][:, None], cls_id[[i]].float()], dim=1).cpu().numpy()
        return det

    def _get_boxes_from_seg_labels(self, img_path: str, img_size: int):
        """优先从 seg 标签计算 bbox（像素坐标，img_size 基准）。"""
        label_path = imgpath_to_labelpath(img_path)
        seg_boxes = load_seg_bboxes_from_label(label_path, img_size)  # (N,4)
        return seg_boxes  # 可能为空

    def process_one(self, img_path, save_path, img_size=640):
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f'[WARN] read fail: {img_path}'); return
        oh, ow = bgr.shape[:2]
        lb, _, _ = letterbox(bgr.copy(), (img_size, img_size), auto=False, scaleFill=True)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        rgb_f = (rgb.astype(np.float32) / 255.0)
        tensor = torch.from_numpy(np.transpose(rgb_f, (2,0,1))).unsqueeze(0).to(self.device)

        # ===== 计算 CAM（库内部完成前/后向）=====
        with torch.set_grad_enabled(True):
            grayscale_cam = self.cam(input_tensor=tensor, targets=[self.target])
        grayscale_cam = grayscale_cam[0]  # (H, W), in [0,1]

        # === 1) 优先用 seg 标签生成 bbox ===
        seg_boxes = self._get_boxes_from_seg_labels(img_path, img_size)  # (N,4)

        boxes_for_mask = None
        if seg_boxes is not None and len(seg_boxes) > 0:
            boxes_for_mask = seg_boxes
        else:
            # === 2) 回退：用预测的 top-1 框（保证可视化不空） ===
            with torch.no_grad():
                raw_pred = self.forward_raw(tensor)[0]
            det = self.post_process(raw_pred, shape_hw=(img_size, img_size))
            if det.shape[0] == 0:
                det = self._fallback_top1_det(raw_pred, shape_hw=(img_size, img_size))
            boxes_for_mask = det[:, :4] if det is not None and det.shape[0] > 0 else None

        # === 只显示 bbox 内的 CAM：框外置零；不画框；不做归一化 ===
        cam_in_boxes = mask_cam_by_bboxes(grayscale_cam, boxes_for_mask)
        vis_img = show_cam_on_image(rgb_f, cam_in_boxes, use_rgb=True)

        vis_img = cv2.resize(vis_img, (ow, oh))
        Image.fromarray(vis_img).save(save_path)

    def __call__(self, img_path, save_dir):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        if os.path.isdir(img_path):
            files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]
            for f in trange(len(files)):
                p = os.path.join(img_path, files[f])
                self.process_one(p, os.path.join(save_dir, files[f]))
        else:
            self.process_one(img_path, os.path.join(save_dir, 'result.png'))

# -------------------------- 辅助：打印层索引 --------------------------
def print_model_layers(weight, device='cpu'):
    y = YOLO(weight)
    m = y.model.to(device)
    print('== Model layers (index : class) ==')
    for i, layer in enumerate(m.model):
        print(f'{i:>3d}: {layer.__class__.__name__}')

# -------------------------- 参数 --------------------------
def get_params():
    return {
        'weight': '/content/drive/MyDrive/yolov13-coffeev2/runs/detect/SRYOLO_coffee4/weights/last.pt',
        'device': 'cuda:0',
        'method': 'EigenCAM',   # 可切换: 'GradCAMPlusPlus' / 'LayerCAM' / 'EigenCAM' ...
        'layer_idx_list': [24, 28],
        'backward_type': 'all',      # 'class' / 'box' / 'all'
        'conf_threshold': 0.25,
        'ratio': 0.25
    }

if __name__ == '__main__':
    print_model_layers('/content/drive/MyDrive/yolov13-coffeev2/runs/detect/SRYOLO_coffee4/weights/last.pt', device='cpu')

    p = get_params()
    cam = YoloV8Heatmap(
        weight=p['weight'],
        device=p['device'],
        method=p['method'],
        layer_idx_list=p['layer_idx_list'],
        backward_type=p['backward_type'],
        conf_threshold=p['conf_threshold'],
        ratio=p['ratio'],
    )
    cam('/content/drive/MyDrive/yolov13-coffeev2/merged_without_part_003/images/test',
        'result_cam_segBBox_only')
