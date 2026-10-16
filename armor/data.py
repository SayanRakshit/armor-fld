"""WFLW loading, face-crop cache and the training / evaluation Dataset."""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .common import (WFLW_FLIP, crop_affine, transform_points, warp, square_from_box,
                     render_landmark_heatmaps, render_edge_heatmaps, normalize_image, N_EDGES)
from .degrade import DegradationBank, visibility_from_unreliability, DEG_TYPES


# ----------------------------------------------------------------------------------------------
# Annotation parsing
# ----------------------------------------------------------------------------------------------
def parse_wflw_list(list_path):
    """Each line: 196 coords, 4 rect values (x_min y_min x_max y_max), 6 attribute flags, image name."""
    items = []
    with open(list_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 207:
                continue
            pts = np.array(p[:196], dtype=np.float32).reshape(98, 2)
            rect = np.array(p[196:200], dtype=np.float32)
            attrs = np.array(p[200:206], dtype=np.int64)
            items.append(dict(pts=pts, rect=rect, attrs=attrs, name=p[206]))
    return items


def wflw_lists(data_root):
    base = os.path.join(data_root, "WFLW_annotations", "list_98pt_rect_attr_train_test")
    return (os.path.join(base, "list_98pt_rect_attr_train.txt"),
            os.path.join(base, "list_98pt_rect_attr_test.txt"))


# ----------------------------------------------------------------------------------------------
# Face-crop cache (decode each JPEG once; training then runs from a uint8 numpy array)
# ----------------------------------------------------------------------------------------------
def build_cache(items, img_root, out_prefix, cache_size=160, cache_scale=1.6, limit=None, verbose=True):
    """Writes <out_prefix>_img.npy (N,S,S,3 uint8 RGB), _pts.npy (N,98,2 crop coords), _attrs.npy (N,6),
    _meta.npy (N,2,3 affine: crop coords -> original image coords)."""
    if limit:
        items = items[:limit]
    N = len(items)
    if os.path.exists(out_prefix + "_img.npy") and len(np.load(out_prefix + "_attrs.npy", mmap_mode="r")) == N:
        if verbose:
            print(f"cache exists: {out_prefix} ({N} faces)")
        return load_cache(out_prefix)
    imgs = np.lib.format.open_memmap(out_prefix + "_img.npy", mode="w+", dtype=np.uint8, shape=(N, cache_size, cache_size, 3))
    pts_all = np.zeros((N, 98, 2), np.float32)
    attrs = np.zeros((N, 6), np.int64)
    meta = np.zeros((N, 2, 3), np.float32)
    try:
        from tqdm.auto import tqdm
        it = tqdm(range(N), desc=f"caching {os.path.basename(out_prefix)}") if verbose else range(N)
    except ImportError:
        it = range(N)
    last_path, img = None, None
    for i in it:
        d = items[i]
        path = os.path.join(img_root, d["name"])
        if path != last_path:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            last_path = path
        # square around the annotation rect (same convention as detector boxes at inference)
        center, side = square_from_box(d["rect"], cache_scale)
        M = crop_affine(center, side, cache_size)
        imgs[i] = warp(img, M, cache_size)
        pts_all[i] = transform_points(d["pts"], M)
        attrs[i] = d["attrs"]
        meta[i] = cv2.invertAffineTransform(M)
    imgs.flush()
    np.save(out_prefix + "_pts.npy", pts_all)
    np.save(out_prefix + "_attrs.npy", attrs)
    np.save(out_prefix + "_meta.npy", meta)
    return load_cache(out_prefix)


def load_cache(out_prefix):
    return dict(img=np.load(out_prefix + "_img.npy", mmap_mode="r"),
                pts=np.load(out_prefix + "_pts.npy"),
                attrs=np.load(out_prefix + "_attrs.npy"),
                meta=np.load(out_prefix + "_meta.npy"))


# ----------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------
class FaceDataset(Dataset):
    """Returns 128x128 crops (clean + degraded) with all supervision signals.

    train=True : random similarity augmentation + flip + colour jitter + degradation bank.
    train=False: deterministic centre crop; `fixed_deg=(type_name, severity)` turns the test set
                 into one cell of the WFLW-C benchmark (deterministic per index).
    """

    def __init__(self, cache, cfg, train=True, fixed_deg=None, return_clean_targets=True):
        self.c = cache
        self.cfg = cfg
        self.train = train
        self.fixed_deg = fixed_deg
        self.return_clean_targets = return_clean_targets
        self.S = cfg["crop_size"]
        self.H = cfg["heatmap_size"]
        self.bank = DegradationBank(grid=16)
        self.n = len(cache["attrs"])

    def __len__(self):
        return self.n

    # geometry of the crop inside the cached image --------------------------------------------
    def _affine(self, rng):
        cs, cache_size = self.cfg, self.cfg["cache_size"]
        # base crop = crop_scale * box, box = cache_size / cache_scale
        base_side = cache_size * cs["crop_scale"] / cs["cache_scale"]
        center = np.array([(cache_size - 1) / 2.0] * 2)
        rot, scale, flip = 0.0, 1.0, False
        if self.train:
            rot = float(rng.uniform(-cs["aug_rot"], cs["aug_rot"]))
            scale = float(1.0 + rng.uniform(-cs["aug_scale"], cs["aug_scale"]))
            center = center + rng.uniform(-cs["aug_shift"], cs["aug_shift"], 2) * base_side
            flip = bool(rng.random() < cs["aug_flip"])
        return crop_affine(center, base_side * scale, self.S, rot, flip), flip

    def _color_jitter(self, img, rng):
        a = self.cfg["aug_color"]
        if a <= 0:
            return img
        x = img.astype(np.float32)
        x = x * float(rng.uniform(1 - a, 1 + a)) + float(rng.uniform(-a, a)) * 64.0
        x = (x - x.mean()) * float(rng.uniform(1 - a, 1 + a)) + x.mean()
        return np.clip(x, 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else (10_000_003 * i + 17))
        M, flip = self._affine(rng)
        img = warp(np.asarray(self.c["img"][i]), M, self.S)
        pts = transform_points(self.c["pts"][i], M)
        if flip:
            pts = pts[WFLW_FLIP]
        if self.train:
            img = self._color_jitter(img, rng)
        clean = img

        # ---- degradation bank -> unreliability / type / visibility targets
        u = np.zeros((self.S, self.S), np.float32)
        tmap = np.zeros((self.S, self.S), np.int64)
        deg = clean
        if self.fixed_deg is not None:
            name, sev = self.fixed_deg
            if name != "clean":
                deg, u = self.bank.apply(clean, name, sev, rng)
                tmap = np.where(u >= 0.05, DEG_TYPES.index(name), 0)
        elif self.train and rng.random() < self.cfg["p_degrade"]:
            deg, u, tmap = self.bank.random(clean, rng)
        u_grid, t_grid = self.bank.to_grid(u, tmap)
        vis = visibility_from_unreliability(u, pts)

        out = dict(
            img=torch.from_numpy(normalize_image(deg)),
            pts=torch.from_numpy(pts.astype(np.float32)),            # (98,2) in crop pixels
            unrel=torch.from_numpy(u_grid.reshape(-1)),             # (256,) alpha target
            dtype=torch.from_numpy(t_grid.reshape(-1)),             # (256,) type target
            vis=torch.from_numpy(vis),                              # (98,)
            attrs=torch.from_numpy(np.asarray(self.c["attrs"][i])),
            index=i,
        )
        if self.return_clean_targets:
            s = self.H / self.S
            out["img_clean"] = torch.from_numpy(normalize_image(clean))
            out["lmk_hm"] = torch.from_numpy(render_landmark_heatmaps(pts * s, self.H, self.cfg["lmk_sigma"]))
            out["edge_hm"] = torch.from_numpy(render_edge_heatmaps(pts * s, self.H, self.cfg["edge_sigma"]))
        return out


def make_loader(ds, bs, train, cfg):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train, drop_last=train,
                                       num_workers=cfg["num_workers"], pin_memory=True,
                                       persistent_workers=cfg["num_workers"] > 0)


# ----------------------------------------------------------------------------------------------
# Synthetic stand-in (lets the whole notebook run even before WFLW is downloaded)
# ----------------------------------------------------------------------------------------------
def make_synthetic_cache(n, cache_size=160, seed=0):
    """Cartoon faces with a deterministic 98-point layout: only for smoke-testing the pipeline."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, cache_size, cache_size, 3), np.uint8)
    pts = np.zeros((n, 98, 2), np.float32)
    base = _canonical_face_98()
    for i in range(n):
        s = cache_size * rng.uniform(0.45, 0.6)
        c = cache_size / 2 + rng.uniform(-6, 6, 2)
        ang = np.radians(rng.uniform(-15, 15))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        p = (base - 0.5) @ R.T * s + c
        img = np.full((cache_size, cache_size, 3), rng.integers(40, 200, 3), np.uint8)
        skin = tuple(int(v) for v in rng.integers(120, 240, 3))
        cv2.fillPoly(img, [np.round(p[:33]).astype(np.int32)], skin)
        cv2.fillPoly(img, [np.round(np.vstack([p[:33], p[42:51][::-1]])).astype(np.int32)], skin)
        for grp in (p[33:42], p[42:51], p[60:68], p[68:76], p[76:88]):
            cv2.polylines(img, [np.round(grp).astype(np.int32)], True, (20, 20, 20), 2)
        cv2.polylines(img, [np.round(p[51:60]).astype(np.int32)], False, (60, 40, 40), 2)
        for q in (p[96], p[97]):
            cv2.circle(img, (int(q[0]), int(q[1])), 2, (0, 0, 0), -1)
        imgs[i] = img
        pts[i] = p
    attrs = rng.integers(0, 2, (n, 6)).astype(np.int64)
    meta = np.tile(np.array([[1, 0, 0], [0, 1, 0]], np.float32), (n, 1, 1))
    return dict(img=imgs, pts=pts, attrs=attrs, meta=meta)


def _canonical_face_98():
    """A plausible 98-point template in [0,1]^2 (used for the synthetic smoke-test data only)."""
    p = np.zeros((98, 2), np.float32)
    t = np.linspace(np.pi, 2 * np.pi, 33)                               # face contour (chin arc)
    p[:33, 0] = 0.5 + 0.42 * np.cos(t); p[:33, 1] = 0.45 - 0.5 * np.sin(t)
    def loop(idx, cx, cy, rx, ry):
        # WFLW order: start at the left-most point, run along the upper lid/edge to the right-most point,
        # then back along the lower edge (clockwise in image coordinates) - this makes the flip map consistent
        a = np.pi + np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
        p[idx, 0] = cx + rx * np.cos(a); p[idx, 1] = cy + ry * np.sin(a)
    loop(list(range(33, 42)), 0.32, 0.30, 0.13, 0.03)                  # left brow
    loop(list(range(42, 51)), 0.68, 0.30, 0.13, 0.03)                  # right brow
    p[51:55] = np.stack([np.full(4, 0.5), np.linspace(0.40, 0.58, 4)], 1)   # nose bridge
    p[55:60] = np.stack([np.linspace(0.42, 0.58, 5), [0.62, 0.64, 0.65, 0.64, 0.62]], 1)
    loop(list(range(60, 68)), 0.34, 0.42, 0.08, 0.04)                  # left eye
    loop(list(range(68, 76)), 0.66, 0.42, 0.08, 0.04)                  # right eye
    a = np.linspace(np.pi, 0, 7); p[76:83] = np.stack([0.5 + 0.17 * np.cos(a), 0.78 - 0.04 * np.sin(a)], 1)
    a = np.linspace(0, np.pi, 7)[1:-1]; p[83:88] = np.stack([0.5 + 0.17 * np.cos(a), 0.78 + 0.07 * np.sin(a)], 1)
    a = np.linspace(np.pi, 0, 5); p[88:93] = np.stack([0.5 + 0.10 * np.cos(a), 0.785 - 0.01 * np.sin(a)], 1)
    a = np.linspace(0, np.pi, 5)[1:-1]; p[93:96] = np.stack([0.5 + 0.10 * np.cos(a), 0.785 + 0.03 * np.sin(a)], 1)
    p[96] = (0.34, 0.42); p[97] = (0.66, 0.42)
    return p
