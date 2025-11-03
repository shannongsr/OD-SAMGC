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
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

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
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def draw_box_label(img, box, color=(0,0,255), label_str="", thickness=3):
    x1,y1,x2,y2 = list(map(int, box))
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label_str:
        cv2.putText(img, label_str, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return img

def renormalize_cam_in_bboxes(boxes, image_float_np, grayscale_cam):
    """将 CAM 在每个 bbox 内独立归一化到[0,1]，框外置零"""
    renorm = np.zeros_like(grayscale_cam, dtype=np.float32)
    H, W = grayscale_cam.shape
    for x1,y1,x2,y2 in boxes:
        x1 = max(0, min(W-1, int(x1))); x2 = max(0, min(W-1, int(x2)))
        y1 = max(0, min(H-1, int(y1))); y2 = max(0, min(H-1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = grayscale_cam[y1:y2, x1:x2].copy()
        renorm[y1:y2, x1:x2] = scale_cam_image(patch)
    renorm = scale_cam_image(renorm)
    vis = show_cam_on_image(image_float_np, renorm, use_rgb=True)
    return vis

# === 自适应坐标解码：统一返回像素 xyxy ===
def _decode_boxes_auto(boxes: torch.Tensor, shape_hw):
    H, W = shape_hw
    b = boxes.clone()
    if b.numel() == 0:
        return b

    mx = float(b.max().detach().cpu())
    if mx <= 1.5:
        # 归一化 xywh ∈ [0,1]
        b = xywh2xyxy(b)
        b[:, [0,2]] *= W
        b[:, [1,3]] *= H
    else:
        # 像素尺度：猜测是 xyxy 还是 xywh
        b_try = xywh2xyxy(b)
        frac_bad = ((b_try[:, 2] < b_try[:, 0]) | (b_try[:, 3] < b_try[:, 1])).float().mean().item() if b_try.numel() else 0.0
        b = b if frac_bad < 0.1 else b_try

    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H - 1)
    return b

# -------------------------- 目标（反传所用的“损失”） --------------------------
class YoloV8Target(torch.nn.Module):
    """
    将 YOLOv8 的原始预测转换为一个用于反向传播的标量。
    - 若通过 conf 过滤后为空，回退到 obj*cls_max 的总和，保证梯度非零（GradCAM++/LayerCAM 必须有梯度）。
    """
    def __init__(self, output_type='all', conf=0.25, top_ratio=1.0):
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.top_ratio = top_ratio

    def forward(self, model_outputs):
        # model_outputs: (B, N, 5+nc) 或 [tensor,...]
        out = model_outputs
        if isinstance(out, (list, tuple)):
            out = out[0]
        preds = out[0] if out.dim() == 3 else out  # (N, 5+nc)

        if preds.ndim != 2 or preds.size(1) < 6:
            return preds.sum() * 0.0  # 保底

        boxes = preds[:, 0:4]
        obj = preds[:, 4:5]
        cls = preds[:, 5:]
        cls_max, _ = cls.max(dim=1, keepdim=True)
        score = (obj * cls_max).squeeze(1)

        keep = score > self.conf
        if keep.sum() == 0:
            # 保底目标，确保有梯度
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
    def __init__(self, weight, device, method, layer_idx_list, backward_type, conf_threshold, ratio, show_box, renormalize):
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
        self.show_box = show_box
        self.renormalize = renormalize
        self.colors = np.random.uniform(0, 255, size=(len(self.names), 3)).astype(np.int64)

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
        """在 det 为空时，从 raw_pred 中取 top1 置信度做兜底框。返回 np.array Nx6 或空。"""
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
        grayscale_cam = grayscale_cam[0]  # (H, W)

        # 推理结果（未NMS张量）用于画框与renormalize
        with torch.no_grad():
            raw_pred = self.forward_raw(tensor)[0]
        det = self.post_process(raw_pred, shape_hw=(img_size, img_size))

        # 若无检测，启用 top-1 兜底（保证 show_box / renormalize 可用）
        if det.shape[0] == 0:
            det = self._fallback_top1_det(raw_pred, shape_hw=(img_size, img_size))

        # --- 可视化 ---
        vis_img = show_cam_on_image(rgb_f, grayscale_cam, use_rgb=True)

        # 框内归一化（renormalize）
        if self.renormalize and det.shape[0] > 0:
            vis_img = renormalize_cam_in_bboxes(det[:, :4], rgb_f, grayscale_cam)

        # 画框（show_box）
        if self.show_box and det.shape[0] > 0:
            for x1,y1,x2,y2, sc, cid in det.astype(np.float32):
                name = self.names.get(int(cid), str(int(cid)))
                color = (0, 0, 255)  # 纯红色更显眼
                vis_img = draw_box_label(vis_img, (x1,y1,x2,y2), color, f'{name} {sc:.2f}', thickness=3)

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
        'method': 'EigenCAM',   # 可切换: 'GradCAMPlusPlus' / 'LayerCAM' / 'EigenCAM'
        'layer_idx_list': [24, 28],
        'backward_type': 'all',      # 'class' / 'box' / 'all'
        'conf_threshold': 0.25,
        'ratio': 0.25,
        'show_box': False,
        'renormalize': False
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
        show_box=p['show_box'],
        renormalize=p['renormalize']
    )
    cam('/content/drive/MyDrive/yolov13-coffeev2/merged_without_part_003/images/test', 'result_cam_24_28_EigenCAM_SG1004')
