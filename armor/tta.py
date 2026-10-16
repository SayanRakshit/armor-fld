"""Prior-Guided Test-Time Adaptation (PG-TTA) with a VQ shift detector.

Label-free, source-free adaptation of the landmark branch on the test crops themselves:
  L_prior : the backbone's edge head must agree with the frozen clean-face prior's reconstruction H_rec
  L_flip  : landmark heatmaps must be equivariant to horizontal flipping (with the 98-pt permutation)
  L_ent   : (small) entropy minimisation of the landmark heatmap distributions
Only normalisation affine parameters + the fusion gate are updated (see ARMORLandmarkNet.tta_parameters).

The number of gradient steps is *gated by the VQ shift score*: the quantisation error of the frozen
clean-face encoder measures how far a face lies from the clean manifold.  Clean faces get 0 steps
(no cost, no risk of degradation), heavily corrupted faces get up to `tta_max_steps`.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .common import WFLW_FLIP
from .fld import heatmap_probs


class ShiftDetector:
    """Shift score of a test batch, z-scored against clean source crops.

    mode="alpha" (default): mean predicted unreliability of the R-ORFormer (its alpha head is trained on the
        degradation bank, so it is a calibrated, degradation-agnostic shift signal).
    mode="vq": quantisation error of the frozen clean-face VQ encoder. On WFLW this turned out to be a poor
        gate: blur / low-light / compression *remove* high-frequency detail and quantise as well as or better
        than clean faces (negative z), so only occlusion raised it. Kept for the ablation.
    """

    def __init__(self, prior, mode="alpha"):
        self.prior, self.mode = prior, mode
        self.mu, self.sigma = None, None

    @torch.no_grad()
    def score(self, x):
        x64 = F.interpolate(x, size=64, mode="bilinear", align_corners=False)
        if self.mode == "vq":
            return self.prior.shift_score(x64)
        was_training = self.prior.training
        self.prior.eval()
        s = self.prior(x64)["alpha"].mean(1)                                 # (B,) mean unreliability
        self.prior.train(was_training)
        return s

    @torch.no_grad()
    def fit(self, loader, device, max_batches=40):
        scores = []
        for i, b in enumerate(loader):
            scores.append(self.score(b["img_clean"].to(device, non_blocking=True)).cpu())
            if i + 1 >= max_batches:
                break
        s = torch.cat(scores)
        self.mu, self.sigma = float(s.mean()), float(s.std() + 1e-6)
        return self

    @torch.no_grad()
    def z(self, x):
        return (self.score(x) - self.mu) / self.sigma                      # (B,)

    def state_dict(self):
        return dict(mu=self.mu, sigma=self.sigma, mode=self.mode)

    def load_state_dict(self, d):
        self.mu, self.sigma, self.mode = d["mu"], d["sigma"], d.get("mode", self.mode)


def flip_heatmaps(hm):
    """Horizontally flip landmark heatmaps and permute channels so they align with the un-flipped face."""
    return hm.flip(-1)[:, WFLW_FLIP]


def tta_losses(model, x, out, cfg):
    losses = {}
    edge_pred = out["edge_hms"][-1].float()
    losses["prior"] = F.mse_loss(edge_pred, out["H_rec"].float().detach())
    if cfg["tta_w_flip"] > 0:
        out_f = model(x.flip(-1), prior_out=None)
        hm_f = flip_heatmaps(out_f["lmk_hms"][-1].float())
        losses["flip"] = F.mse_loss(out["lmk_hms"][-1].float(), hm_f)
    if cfg["tta_w_ent"] > 0:
        p = heatmap_probs(out["lmk_hms"][-1], cfg["softargmax_beta"], cfg["softargmax_window"])
        losses["ent"] = -(p * (p + 1e-8).log()).sum(-1).mean()
    total = cfg["tta_w_prior"] * losses["prior"] + cfg["tta_w_flip"] * losses.get("flip", 0) + cfg["tta_w_ent"] * losses.get("ent", 0)
    return total, losses


class PGTTA:
    """Episodic (default) or online prior-guided test-time adaptation wrapper."""

    def __init__(self, model, cfg, detector: ShiftDetector):
        self.model, self.cfg, self.det = model, cfg, detector
        self.params = model.tta_parameters()
        self._snapshot = [p.detach().clone() for p in self.params]
        self.opt = torch.optim.Adam(self.params, lr=cfg["tta_lr"])
        self.model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in self.params:
            p.requires_grad_(True)

    def reset(self):
        with torch.no_grad():
            for p, s in zip(self.params, self._snapshot):
                p.copy_(s)
        self.opt = torch.optim.Adam(self.params, lr=self.cfg["tta_lr"])

    def steps_for(self, z):
        """Shift z-score -> number of adaptation steps (per batch: use the median)."""
        if self.cfg.get("tta_force_steps") is not None:          # ablation: no shift gating
            return int(self.cfg["tta_force_steps"])
        zb = float(torch.median(z))
        return int(np.clip(round(zb / self.cfg["tta_z_per_step"]), 0, self.cfg["tta_max_steps"]))

    def __call__(self, x):
        cfg = self.cfg
        z = self.det.z(x)
        n_steps = self.steps_for(z)
        if not cfg["tta_online"]:
            self.reset()
        for _ in range(n_steps):
            with torch.enable_grad():
                out = self.model(x)
                loss, _ = tta_losses(self.model, x, out, cfg)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
        with torch.no_grad():
            out = self.model(x)
        out["tta_steps"] = n_steps
        out["shift_z"] = z
        return out
