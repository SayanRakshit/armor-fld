"""Stage C - landmark network: lite stacked hourglass + reliability-gated prior fusion + visibility head."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .orformer import ARMORPrior


# ---------------------------------------------------------------------------------- building blocks
class Bottleneck(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        mid = cout // 2
        self.body = nn.Sequential(
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True), nn.Conv2d(cin, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, cout, 1, bias=False))
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.body(x)


class Hourglass(nn.Module):
    def __init__(self, depth, ch):
        super().__init__()
        self.depth = depth
        self.up1 = Bottleneck(ch, ch)
        self.low1 = Bottleneck(ch, ch)
        self.low2 = Hourglass(depth - 1, ch) if depth > 1 else Bottleneck(ch, ch)
        self.low3 = Bottleneck(ch, ch)

    def forward(self, x):
        up1 = self.up1(x)
        low = self.low3(self.low2(self.low1(F.max_pool2d(x, 2))))
        return up1 + F.interpolate(low, scale_factor=2, mode="nearest")


# ---------------------------------------------------------------------------------- soft-argmax + covariance
def heatmap_probs(hm, beta, window=9):
    """Windowed spatial softmax: softmax(beta * hm) restricted to a (window x window) box around the argmax.

    The window removes the centre bias of a global soft-argmax over 64x64 pixels while keeping the
    distribution differentiable; its covariance describes the local (anisotropic) shape of the peak.
    hm: (B,N,H,W) -> probs (B,N,H*W) float32
    """
    B, N, H, W = hm.shape
    flat = hm.float().flatten(2)
    if window and window < max(H, W):
        r = window // 2
        idx = flat.argmax(-1)
        ym, xm = idx // W, idx % W
        xs = torch.arange(W, device=hm.device); ys = torch.arange(H, device=hm.device)
        mx = (xs[None, None, :] - xm[..., None]).abs() <= r                    # (B,N,W)
        my = (ys[None, None, :] - ym[..., None]).abs() <= r                    # (B,N,H)
        mask = (my[..., :, None] & mx[..., None, :]).flatten(2)
        flat = flat.masked_fill(~mask, -1e4 / beta)
    return F.softmax(flat * beta, -1)


def softargmax_cov(hm, beta, window=9):
    """hm: (B,N,H,W) -> mu (B,N,2) [x,y] in heatmap pixels, cov (B,N,2,2), probs (B,N,H*W).

    The heatmap is turned into a distribution (see heatmap_probs); its mean is the landmark,
    its covariance feeds the STAR loss (anisotropic ambiguity) and the visibility head.
    """
    B, N, H, W = hm.shape
    p = heatmap_probs(hm, beta, window)
    xs = torch.arange(W, device=hm.device, dtype=torch.float32)
    ys = torch.arange(H, device=hm.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)                                   # (HW,) pixel coordinates
    moments = p @ torch.stack([gx, gy, gx * gx, gy * gy, gx * gy], 1)          # (B,N,5): E[x],E[y],E[x2],E[y2],E[xy]
    mx, my, mxx, myy, mxy = moments.unbind(-1)
    cxx, cyy, cxy = (mxx - mx * mx).clamp(min=0), (myy - my * my).clamp(min=0), mxy - mx * my
    mu = torch.stack([mx, my], -1)                                            # (B,N,2)
    cov = torch.stack([torch.stack([cxx, cxy], -1), torch.stack([cxy, cyy], -1)], -2)  # (B,N,2,2)
    return mu, cov, p


# ---------------------------------------------------------------------------------- the network
class ARMORLandmarkNet(nn.Module):
    def __init__(self, cfg, prior: ARMORPrior, n_landmarks=98, n_edges=13):
        super().__init__()
        self.cfg, self.N, self.E = cfg, n_landmarks, n_edges
        ch, stacks = cfg["hg_ch"], cfg["hg_stacks"]
        self.prior = prior
        if not cfg["finetune_prior"]:
            self.prior.requires_grad_(False)
        # stem: 128 -> 64 (one stride-2 conv), feature width ch
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True), Bottleneck(64, ch))
        # reliability-gated fusion of the prior heatmaps
        self.prior_proj = nn.Sequential(nn.Conv2d(n_edges, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.gate = nn.Conv2d(2 * ch, ch, 3, padding=1)
        nn.init.constant_(self.gate.bias, -2.0)                                   # gate starts almost closed (~0.12) ...
        self.kappa = nn.Parameter(torch.tensor(1.0))                              # ... and opens where the image is unreliable
        # hourglass stacks with landmark + edge heads and intermediate supervision
        self.hgs = nn.ModuleList([Hourglass(cfg["hg_depth"], ch) for _ in range(stacks)])
        self.feat = nn.ModuleList([nn.Sequential(Bottleneck(ch, ch), nn.Conv2d(ch, ch, 1, bias=False),
                                                 nn.BatchNorm2d(ch), nn.ReLU(inplace=True)) for _ in range(stacks)])
        self.out_lmk = nn.ModuleList([nn.Conv2d(ch, n_landmarks, 1) for _ in range(stacks)])
        self.out_edge = nn.ModuleList([nn.Conv2d(ch, n_edges, 1) for _ in range(stacks)])
        self.merge_f = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(stacks - 1)])
        self.merge_o = nn.ModuleList([nn.Conv2d(n_landmarks + n_edges, ch, 1) for _ in range(stacks - 1)])
        # per-landmark visibility head: sampled features + reliability + heatmap statistics
        self.vis_head = nn.Sequential(nn.Linear(ch + 4, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    # ------------------------------------------------------------------ helpers
    def run_prior(self, x):
        x64 = F.interpolate(x, size=64, mode="bilinear", align_corners=False)
        if self.cfg["finetune_prior"]:
            return self.prior(x64)
        with torch.no_grad():
            return self.prior(x64)

    def _sample(self, fmap, mu):
        """Bilinear sample fmap (B,C,H,W) at mu (B,N,2) given in fmap pixel coords -> (B,N,C)."""
        B, C, H, W = fmap.shape
        g = torch.stack([mu[..., 0] / (W - 1) * 2 - 1, mu[..., 1] / (H - 1) * 2 - 1], -1).unsqueeze(2)
        return F.grid_sample(fmap, g.to(fmap.dtype), align_corners=True).squeeze(-1).transpose(1, 2)

    # ------------------------------------------------------------------ forward
    def forward(self, x, prior_out=None):
        """x: (B,3,128,128) normalised crop. Returns dict with heatmaps, coords (input pixels), visibility."""
        B = x.shape[0]
        po = prior_out if prior_out is not None else self.run_prior(x)
        H_rec = po["hm"].to(x.dtype)                                              # (B,E,64,64)
        r = (1 - po["alpha"]).view(B, 1, 16, 16).to(x.dtype)
        r64 = F.interpolate(r, size=H_rec.shape[-1], mode="bilinear", align_corners=False)

        f = self.stem(x)                                                          # (B,ch,64,64)
        P = self.prior_proj(H_rec)
        g = torch.sigmoid(self.gate(torch.cat([f, P], 1)) + self.kappa * (1 - r64))
        f = f + g * P

        lmk_hms, edge_hms = [], []
        for i, hg in enumerate(self.hgs):
            y = self.feat[i](hg(f))
            lmk, edge = self.out_lmk[i](y), self.out_edge[i](y)
            lmk_hms.append(lmk); edge_hms.append(edge)
            if i < len(self.hgs) - 1:
                f = f + self.merge_f[i](y) + self.merge_o[i](torch.cat([lmk, edge], 1))

        mu, cov, probs = softargmax_cov(lmk_hms[-1], self.cfg["softargmax_beta"], self.cfg["softargmax_window"])   # heatmap pixels
        scale = x.shape[-1] / lmk_hms[-1].shape[-1]
        # visibility: features at the landmark, local reliability, peak value and spread
        fs = self._sample(y, mu)                                                  # (B,N,ch)
        rs = self._sample(r64, mu)                                                # (B,N,1)
        peak = lmk_hms[-1].flatten(2).amax(-1).unsqueeze(-1)                      # (B,N,1)
        tr = torch.log1p(cov[..., 0, 0] + cov[..., 1, 1]).unsqueeze(-1)                  # log spread
        ent = (-(probs * (probs + 1e-8).log()).sum(-1) / math.log(probs.shape[-1])).unsqueeze(-1)  # normalised entropy
        vis_logit = self.vis_head(torch.cat([fs.float(), rs.float(), peak.float(), tr, ent], -1).to(fs.dtype)).squeeze(-1)
        return dict(lmk_hms=lmk_hms, edge_hms=edge_hms, mu=mu, cov=cov, coords=mu * scale,
                    vis_logit=vis_logit, gate=g, H_rec=H_rec, alpha=po["alpha"], prior=po)

    # ------------------------------------------------------------------ parameter groups for PG-TTA
    def tta_parameters(self):
        """Normalisation affine params of the landmark branch + the fusion gate (the prior stays frozen)."""
        params = []
        for name, m in self.named_modules():
            if name == "prior" or name.startswith("prior."):      # the clean-face prior stays frozen
                continue
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                params += [p for p in m.parameters(recurse=False)]
        params += list(self.gate.parameters()) + [self.kappa]
        return params
