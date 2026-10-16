"""Stage 1 - SCRFD face detector (ONNX, InsightFace weights) + the full detect -> crop -> landmark -> remap pipeline."""
import os
import zipfile
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .common import crop_affine, invert_affine, transform_points, warp, square_from_box, normalize_image

# InsightFace model packs (https://github.com/deepinsight/insightface/tree/master/python-package)
PACKS = {
    "det_500m": ("buffalo_sc", "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"),  # SCRFD-500M-KPS
    "det_10g": ("buffalo_l", "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"),     # SCRFD-10G-KPS
}


def download_scrfd(name="det_500m", root="/content/models"):
    """Returns the path to <name>.onnx, downloading the InsightFace pack if needed."""
    pack, url = PACKS[name]
    os.makedirs(root, exist_ok=True)
    onnx_path = os.path.join(root, f"{name}.onnx")
    if os.path.exists(onnx_path):
        return onnx_path
    zpath = os.path.join(root, f"{pack}.zip")
    if not os.path.exists(zpath):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        member = [m for m in z.namelist() if m.endswith(f"{name}.onnx")][0]
        with z.open(member) as src, open(onnx_path, "wb") as dst:
            dst.write(src.read())
    return onnx_path


class SCRFD:
    """Minimal re-implementation of InsightFace's SCRFD post-processing (strides 8/16/32, 2 anchors, 5 kps)."""

    def __init__(self, onnx_path, input_size=640, providers=None):
        import onnxruntime as ort
        if providers is None:
            avail = ort.get_available_providers()
            want = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            providers = [p for p in want if p in avail]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = input_size
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.use_kps = len(self.output_names) == 9
        self._centers = {}

    def _anchor_centers(self, h, w, stride):
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32) * stride   # (h,w,2) as [x,y]
            c = c.reshape(-1, 2)
            c = np.stack([c] * self.num_anchors, axis=1).reshape(-1, 2)
            self._centers[key] = c
        return self._centers[key]

    @staticmethod
    def _nms(dets, thresh):
        x1, y1, x2, y2, s = dets.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= thresh)[0] + 1]
        return keep

    def detect(self, img_bgr, score_thresh=0.5, nms_thresh=0.4):
        """Returns boxes (N,5) [x1,y1,x2,y2,score] and kps (N,5,2) in original image coordinates."""
        S = self.input_size
        h0, w0 = img_bgr.shape[:2]
        scale = S / max(h0, w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        resized = cv2.resize(img_bgr, (nw, nh))
        canvas = np.zeros((S, S, 3), np.uint8); canvas[:nh, :nw] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (S, S), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        fmc = len(self.strides)
        scores_l, boxes_l, kps_l = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            bbox = outs[i + fmc].reshape(-1, 4) * stride
            h, w = S // stride, S // stride
            ac = self._anchor_centers(h, w, stride)
            keep = np.where(scores >= score_thresh)[0]
            if keep.size == 0:
                continue
            ac, bbox, scores = ac[keep], bbox[keep], scores[keep]
            boxes = np.stack([ac[:, 0] - bbox[:, 0], ac[:, 1] - bbox[:, 1], ac[:, 0] + bbox[:, 2], ac[:, 1] + bbox[:, 3]], 1)
            scores_l.append(scores); boxes_l.append(boxes)
            if self.use_kps:
                kps = outs[i + 2 * fmc].reshape(-1, 5, 2)[keep] * stride + ac[:, None, :]
                kps_l.append(kps)
        if not scores_l:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        scores = np.concatenate(scores_l); boxes = np.concatenate(boxes_l) / scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        order = scores.argsort()[::-1]
        dets = dets[order]
        keep = self._nms(dets, nms_thresh)
        kps = (np.concatenate(kps_l) / scale)[order][keep] if self.use_kps else np.zeros((len(keep), 5, 2), np.float32)
        return dets[keep], kps.astype(np.float32)


class FacePipeline:
    """SCRFD -> square crop (crop_scale x box) -> ARMOR landmark net -> landmarks back on the original image."""

    def __init__(self, detector: SCRFD, model, cfg, device, tta=None):
        self.det, self.model, self.cfg, self.device, self.tta = detector, model, cfg, device, tta
        self.model.eval()

    @torch.no_grad()
    def run(self, img_bgr, score_thresh=0.5, max_faces=32):
        boxes, _ = self.det.detect(img_bgr, score_thresh)
        boxes = boxes[:max_faces]
        if len(boxes) == 0:
            return []
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        S = self.cfg["crop_size"]
        crops, Ms = [], []
        for b in boxes:
            center, side = square_from_box(b, self.cfg["crop_scale"])
            M = crop_affine(center, side, S)
            crops.append(normalize_image(warp(rgb, M, S)))
            Ms.append(M)
        x = torch.from_numpy(np.stack(crops)).to(self.device)
        out = self.tta(x) if self.tta is not None else self.model(x)
        coords = out["coords"].float().cpu().numpy()
        vis = torch.sigmoid(out["vis_logit"].float()).cpu().numpy()
        alpha = out["alpha"].float().cpu().numpy().reshape(-1, 16, 16)
        results = []
        for i, b in enumerate(boxes):
            Minv = invert_affine(Ms[i])
            results.append(dict(box=b[:4], score=float(b[4]), landmarks=transform_points(coords[i], Minv),
                                visibility=vis[i], reliability=1 - alpha[i]))
        return results

    @staticmethod
    def draw(img_bgr, results, vis_thresh=0.5):
        out = img_bgr.copy()
        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r["box"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (x, y), v in zip(r["landmarks"], r["visibility"]):
                color = (0, 255, 255) if v >= vis_thresh else (0, 0, 255)       # red = predicted non-visible
                cv2.circle(out, (int(round(x)), int(round(y))), 2, color, -1)
        return out


def roundtrip_test(n=200, seed=0):
    """Unit test: source -> crop -> source must be exact (<1e-3 px) for random similarity crops."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        box = rng.uniform(0, 500, 2); box = np.concatenate([box, box + rng.uniform(20, 300, 2)])
        center, side = square_from_box(box, 1.3)
        M = crop_affine(center, side, 128, rng.uniform(-45, 45), bool(rng.random() < 0.5))
        pts = rng.uniform(-100, 700, (98, 2))
        back = transform_points(transform_points(pts, M), invert_affine(M))
        worst = max(worst, float(np.abs(back - pts).max()))
    return worst
