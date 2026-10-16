"""Degradation Bank with *reliability supervision*.

Every degradation returns, besides the corrupted image, a per-pixel **unreliability map** u in [0,1]
(how little a local patch can be trusted) and the degradation **type**.  The maps are pooled onto the
ORFormer token grid (16x16) and become supervision for the per-patch reliability score alpha and for the
degradation-type head - this is what turns ORFormer's implicit "occlusion" notion into an explicit,
multi-type *reliability* notion (contribution 1 of ARMOR-FLD).

The same bank, driven deterministically by (type, severity, seed), defines the WFLW-C benchmark.
"""
import numpy as np
import cv2

DEG_TYPES = ["clean", "occlusion", "blur", "lowlight", "weather", "compression"]
TYPE_ID = {n: i for i, n in enumerate(DEG_TYPES)}


def _soft_mask(mask_uint8, k=7):
    m = cv2.GaussianBlur(mask_uint8.astype(np.float32), (k | 1, k | 1), 0)
    return np.clip(m, 0, 1)


def _smooth_field(rng, size, scale=4):
    """Low-frequency random field in [0,1]."""
    small = rng.random((scale, scale)).astype(np.float32)
    f = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)
    return f


class DegradationBank:
    """All methods take/return HxWx3 uint8 RGB images and return (img, unreliability_map)."""

    def __init__(self, grid=16):
        self.grid = grid

    # ------------------------------------------------------------------ occlusion
    def occlusion(self, img, sev, rng):
        H, W = img.shape[:2]
        out = img.copy()
        mask = np.zeros((H, W), np.uint8)
        n = 1 if sev <= 2 else rng.integers(1, 3)
        for _ in range(n):
            frac = 0.12 + 0.07 * sev + rng.uniform(-0.03, 0.03)          # occluder size relative to crop
            w, h = int(W * frac * rng.uniform(0.7, 1.4)), int(H * frac * rng.uniform(0.7, 1.4))
            w, h = max(8, min(w, W - 2)), max(8, min(h, H - 2))
            cx, cy = rng.integers(w // 2, W - w // 2), rng.integers(h // 2, H - h // 2)
            m = np.zeros((H, W), np.uint8)
            shape = rng.integers(0, 3)
            if shape == 0:    # rectangle
                cv2.rectangle(m, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), 1, -1)
            elif shape == 1:  # ellipse
                cv2.ellipse(m, (int(cx), int(cy)), (w // 2, h // 2), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            else:             # hand-like blob = union of ellipses
                for _ in range(4):
                    cv2.ellipse(m, (int(cx + rng.integers(-w // 3, w // 3 + 1)), int(cy + rng.integers(-h // 3, h // 3 + 1))),
                                (max(3, w // 3), max(3, h // 2)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            fill = rng.integers(0, 3)
            if fill == 0:     # solid random colour
                tex = np.empty_like(img); tex[:] = rng.integers(0, 256, 3)
            elif fill == 1:   # noise texture
                tex = rng.integers(0, 256, img.shape).astype(np.uint8)
            else:             # copy-paste of a (flipped) patch of the same image = realistic self-occluder
                tex = np.ascontiguousarray(img[::-1, ::-1])
                tex = np.roll(tex, (int(rng.integers(0, H)), int(rng.integers(0, W))), axis=(0, 1))
            out[m > 0] = tex[m > 0]
            mask |= m
        u = _soft_mask(mask, 5)
        return out, u

    # ------------------------------------------------------------------ blur
    def blur(self, img, sev, rng):
        H, W = img.shape[:2]
        k = int(3 + 4 * sev)                                       # 7 .. 23 px kernels at 128 px
        if rng.random() < 0.6:                                     # motion blur
            kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0
            Mrot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
            kern = cv2.warpAffine(kern, Mrot, (k, k)); kern /= kern.sum() + 1e-8
            blurred = cv2.filter2D(img, -1, kern)
        else:                                                      # defocus
            sigma = 0.6 * sev + 0.4
            blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        strength = min(1.0, (k - 3) / 20.0)
        if rng.random() < 0.5:                                     # regional blur (soft mask)
            field = _smooth_field(rng, H, 3)
            region = (field > np.quantile(field, 0.45)).astype(np.uint8)
            m = _soft_mask(region, 15)[..., None]
            out = (blurred * m + img * (1 - m)).astype(np.uint8)
            u = strength * m[..., 0]
        else:
            out, u = blurred, np.full((H, W), strength, np.float32)
        return out, u

    # ------------------------------------------------------------------ low light + sensor noise
    def lowlight(self, img, sev, rng):
        H, W = img.shape[:2]
        x = img.astype(np.float32) / 255.0
        gamma = 1.0 + 0.6 * sev
        gain = 1.0 - 0.12 * sev
        vign = 1.0
        if rng.random() < 0.5:                                     # vignette: darker at the borders
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
            vign = np.clip(1.2 - 0.6 * r, 0.3, 1.0)[..., None]
        x = (x ** gamma) * gain * vign
        noise_sigma = (4 + 6 * sev) / 255.0
        x = x + rng.normal(0, noise_sigma, x.shape).astype(np.float32)            # read noise
        x = x + rng.normal(0, 1, x.shape).astype(np.float32) * noise_sigma * np.sqrt(np.clip(x, 0, 1))  # shot noise
        out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        base = min(0.9, 0.15 * sev)
        u = np.full((H, W), base, np.float32)
        if np.ndim(vign):
            u = u * (1.4 - vign[..., 0])                                            # darker corners = less reliable
        return out, np.clip(u, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------ weather: haze + rain
    def weather(self, img, sev, rng):
        H, W = img.shape[:2]
        x = img.astype(np.float32)
        field = _smooth_field(rng, H, 4)
        t = 1.0 - (0.10 + 0.14 * sev) * (0.5 + 0.5 * field)          # transmission
        A = rng.uniform(200, 255)
        out = x * t[..., None] + A * (1 - t[..., None])
        u = (1 - t).astype(np.float32)
        if rng.random() < 0.6:                                     # rain streaks
            rain = np.zeros((H, W), np.uint8)
            ang = rng.uniform(-20, 20)
            for _ in range(int(15 * sev)):
                x0, y0 = rng.integers(0, W), rng.integers(0, H)
                L = rng.integers(8, 20)
                x1 = int(x0 + L * np.sin(np.radians(ang))); y1 = int(y0 + L * np.cos(np.radians(ang)))
                cv2.line(rain, (int(x0), int(y0)), (x1, y1), 1, 1)
            rm = cv2.GaussianBlur(rain.astype(np.float32), (3, 3), 0)[..., None]
            out = out * (1 - 0.6 * rm) + 230 * 0.6 * rm
            u = np.maximum(u, 0.08 * sev * (rm[..., 0] > 0.05))
        return np.clip(out, 0, 255).astype(np.uint8), np.clip(u, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------ compression / pixelation
    def compression(self, img, sev, rng):
        H, W = img.shape[:2]
        q = int(max(3, 42 - 8 * sev))
        f = 1.0 + 0.5 * sev
        small = cv2.resize(img, (max(8, int(W / f)), max(8, int(H / f))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out = cv2.resize(dec, (W, H), interpolation=cv2.INTER_LINEAR)
        u = np.full((H, W), min(0.85, 0.10 + 0.12 * sev), np.float32)
        return out, u

    # ------------------------------------------------------------------ composition API
    def apply(self, img, type_name, sev, rng):
        fn = getattr(self, type_name)
        return fn(img, int(sev), rng)

    def random(self, img, rng, p_clean=0.15, max_types=2):
        """Random composition of 1-2 degradations. Returns img, u(HxW), type_map(HxW int)."""
        H, W = img.shape[:2]
        u_total = np.zeros((H, W), np.float32)
        type_map = np.zeros((H, W), np.int64)
        best_u = np.zeros((H, W), np.float32)
        if rng.random() < p_clean:
            return img, u_total, type_map
        names = DEG_TYPES[1:]
        k = 1 if rng.random() < 0.65 else max_types
        chosen = rng.choice(len(names), size=k, replace=False)
        out = img
        for c in chosen:
            name = names[c]
            sev = int(rng.integers(1, 6))
            out, u = self.apply(out, name, sev, rng)
            u_total = 1 - (1 - u_total) * (1 - u)
            upd = u > best_u
            type_map[upd] = TYPE_ID[name]
            best_u = np.maximum(best_u, u)
        type_map[u_total < 0.05] = 0
        return out, u_total, type_map

    def to_grid(self, u, type_map):
        """Pool pixel maps onto the token grid: mean unreliability + majority type (0 if reliable)."""
        g = self.grid
        H, W = u.shape
        u_grid = u.reshape(g, H // g, g, W // g).mean(axis=(1, 3)).astype(np.float32)
        t = type_map.reshape(g, H // g, g, W // g).transpose(0, 2, 1, 3).reshape(g, g, -1)
        t_grid = np.zeros((g, g), np.int64)
        for i in range(g):
            for j in range(g):
                cnt = np.bincount(t[i, j], minlength=len(DEG_TYPES))
                cnt[0] = 0
                t_grid[i, j] = int(cnt.argmax()) if cnt.max() > 0 and u_grid[i, j] >= 0.05 else 0
        return u_grid, t_grid


def visibility_from_unreliability(u, pts):
    """Per-landmark visibility target = 1 - u sampled (bilinear) at the landmark location."""
    H, W = u.shape
    x = np.clip(pts[:, 0], 0, W - 1.001); y = np.clip(pts[:, 1], 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    v = (u[y0, x0] * (1 - fx) * (1 - fy) + u[y0, x0 + 1] * fx * (1 - fy)
         + u[y0 + 1, x0] * (1 - fx) * fy + u[y0 + 1, x0 + 1] * fx * fy)
    return (1.0 - v).astype(np.float32)
