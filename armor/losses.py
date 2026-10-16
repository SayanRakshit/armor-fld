"""Losses: STAR-V (visibility-weighted STAR), Adaptive Wing, and the stage A/B/C objectives."""
import math
import torch
import torch.nn.functional as F


def eig2x2(cov, eps=1e-6):
    """Closed-form eigen-decomposition of symmetric 2x2 matrices (...,2,2) -> lam (...,2) desc, vecs (...,2,2)."""
    a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    tr, det = a + c, a * c - b * b
    disc = torch.sqrt(torch.clamp(tr * tr / 4 - det, min=0.0) + eps)
    l1, l2 = (tr / 2 + disc).clamp(min=0.0), (tr / 2 - disc).clamp(min=0.0)   # PSD up to rounding
    # eigenvector of l1: (b, l1 - a); for (near-)isotropic heatmaps fall back to the x axis.
    # NaN-safe: no .norm() at zero vectors (its gradient is NaN and leaks through torch.where).
    v1 = torch.stack([b, l1 - a], -1)
    n2 = (v1 * v1).sum(-1, keepdim=True)
    iso = n2 < 1e-10
    v1 = torch.where(iso, torch.tensor([1.0, 0.0], device=cov.device), v1 / torch.sqrt(n2 + 1e-12))
    v2 = torch.stack([-v1[..., 1], v1[..., 0]], -1)
    return torch.stack([l1, l2], -1), torch.stack([v1, v2], -2)   # vecs[..., k, :] = k-th eigenvector


def star_loss(mu, cov, gt, weight=None, detach_lambda=True, w_vr=0.0, eps=1e-4, detach_axes=True, lam_floor=0.25):
    """STAR loss (Zhou et al., CVPR 2023) with optional per-landmark weights (STAR-V).

    mu, gt: (B,N,2) in heatmap pixels; cov: (B,N,2,2) from the predicted heatmap distribution.
    The error is projected on the heatmap's principal axes and each component is divided by the
    corresponding sqrt(eigenvalue): errors along the ambiguous (long) axis are softened.
    """
    lam, vecs = eig2x2(cov.float())
    if detach_axes:                                                  # principal axes act as constants (stable)
        vecs = vecs.detach()
    e = (mu.float() - gt.float()).unsqueeze(-2)                      # (B,N,1,2)
    proj = (e * vecs).sum(-1)                                        # (B,N,2) components along v1, v2
    lam_d = lam.detach() if detach_lambda else lam
    lam_d = lam_d.clamp(min=lam_floor)                               # a one-hot peak must not blow the loss up
    per = F.smooth_l1_loss(proj / torch.sqrt(lam_d + eps), torch.zeros_like(proj), reduction="none").sum(-1)  # (B,N)
    if weight is not None:
        per = per * weight
    loss = per.mean()
    if w_vr > 0:                                                     # eigenvalue (value) restriction
        loss = loss + w_vr * lam.sum(-1).mean()
    return loss


def adaptive_wing_loss(pred, target, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
    """Adaptive Wing loss (Wang et al., ICCV 2019) for heatmap regression."""
    pred, target = pred.float(), target.float()
    diff = (target - pred).abs()
    A = omega * (1 / (1 + (theta / epsilon) ** (alpha - target))) * (alpha - target) * \
        ((theta / epsilon) ** (alpha - target - 1)) / epsilon
    C = theta * A - omega * torch.log1p((theta / epsilon) ** (alpha - target))
    small = diff < theta
    loss = torch.where(small, omega * torch.log1p((diff / epsilon) ** (alpha - target)), A * diff - C)
    # weight foreground pixels more (standard practice)
    w = 1 + 10 * (target > 0.2).float()
    return (loss * w).mean()


def _bce(p, t, eps=1e-5):
    """Plain BCE on probabilities (written out so it is safe inside torch.autocast)."""
    p = p.clamp(eps, 1 - eps)
    return -(t * p.log() + (1 - t) * (1 - p).log()).mean()


# ---------------------------------------------------------------------------------- stage objectives
def stage_a_loss(out, edge_hm):
    l_hm = F.mse_loss(out["hm"].float(), edge_hm.float())
    return l_hm + out["vq_loss"], dict(hm=l_hm.item(), vq=out["vq_loss"].item(), ppl=out["perplexity"].item())


def stage_b_loss(out, idx_tgt, edge_hm, unrel, dtype_tgt, cfg):
    """idx_tgt (B,256) codes of the clean face; unrel (B,256) in [0,1]; dtype_tgt (B,256) long."""
    B = idx_tgt.shape[0]
    l_code = F.cross_entropy(out["logits_I"].float().reshape(-1, cfg["n_codes"]), idx_tgt.reshape(-1)) + \
             F.cross_entropy(out["logits_M"].float().reshape(-1, cfg["n_codes"]), idx_tgt.reshape(-1))
    l_hm = F.mse_loss(out["hm"].float(), edge_hm.float())
    l_rel = sum(_bce(a.float(), unrel.float()) for a in out["alphas"]) / len(out["alphas"])
    l_type = F.cross_entropy(out["type_logits"].float().reshape(-1, out["type_logits"].shape[-1]), dtype_tgt.reshape(-1))
    total = cfg["w_code"] * l_code + cfg["w_hm"] * l_hm + cfg["w_rel"] * l_rel + cfg["w_type"] * l_type
    with torch.no_grad():
        acc_I = (out["logits_I"].argmax(-1) == idx_tgt).float().mean().item()
        rel_mae = (out["alpha"] - unrel).abs().mean().item()
    return total, dict(code=l_code.item(), hm=l_hm.item(), rel=l_rel.item(), type=l_type.item(), accI=acc_I, relMAE=rel_mae)


def stage_c_loss(out, batch, cfg, star_scale=1.0):
    """STAR-V on coordinates + AWing on heatmaps (intermediate supervision) + edge aux + visibility BCE.

    star_scale ramps STAR in during the first part of training (see cfg["star_warmup"]): AWing forms the
    peaks first, STAR-V then refines them sub-pixel; STAR's local window has nothing useful to say while
    the arg-max is still wrong."""
    s = out["lmk_hms"][-1].shape[-1] / batch["img"].shape[-1]
    gt_hm_coords = batch["pts"] * s
    vis_t = batch["vis"]
    w = cfg["vis_floor"] + (1 - cfg["vis_floor"]) * vis_t                 # never 0: GT exists under occlusion too
    gate = cfg.get("star_gate_px", 4)
    if gate:
        # STAR-V is a *refinement* loss: it only acts on landmarks whose heat-map peak already lies within
        # `gate` pixels of the target; wrong peaks are AWing's job (the windowed soft-argmax has no gradient there).
        with torch.no_grad():
            hm = out["lmk_hms"][-1]
            idx = hm.flatten(2).argmax(-1)
            am = torch.stack([idx % hm.shape[-1], idx // hm.shape[-1]], -1).float()
            near = ((am - gt_hm_coords).abs().amax(-1) <= gate).float()
        w = w * near
    l_star = star_loss(out["mu"], out["cov"], gt_hm_coords, w, cfg["star_detach"], cfg["star_vr"])
    l_aw = sum(adaptive_wing_loss(h, batch["lmk_hm"]) for h in out["lmk_hms"]) / len(out["lmk_hms"])
    l_edge = sum(F.mse_loss(e.float(), batch["edge_hm"].float()) for e in out["edge_hms"]) / len(out["edge_hms"])
    l_vis = F.binary_cross_entropy_with_logits(out["vis_logit"].float(), vis_t.float())
    total = star_scale * cfg["w_star"] * l_star + cfg["w_awing"] * l_aw + cfg["w_edge"] * l_edge + cfg["w_vis"] * l_vis
    return total, dict(star=l_star.item(), awing=l_aw.item(), edge=l_edge.item(), vis=l_vis.item())
