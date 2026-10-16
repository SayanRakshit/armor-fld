"""WFLW metrics: NME (inter-ocular), FR@0.10, AUC@0.10, per-attribute subsets."""
import numpy as np
from .common import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, ATTR_NAMES


def nme_per_image(pred, gt):
    """pred, gt: (B,98,2) numpy. Returns (B,) NME normalised by the inter-ocular distance."""
    io = np.linalg.norm(gt[:, LEFT_EYE_OUTER] - gt[:, RIGHT_EYE_OUTER], axis=-1)
    err = np.linalg.norm(pred - gt, axis=-1).mean(-1)
    return err / np.maximum(io, 1e-6)


def failure_rate(nmes, thr=0.10):
    return float((nmes > thr).mean())


def auc_ced(nmes, thr=0.10, steps=1000):
    xs = np.linspace(0, thr, steps + 1)
    ced = np.array([(nmes <= x).mean() for x in xs])
    return float(np.trapezoid(ced, xs) / thr) if hasattr(np, "trapezoid") else float(np.trapz(ced, xs) / thr)


def summarize(nmes, attrs=None, thr=0.10):
    """Dict with full-set and per-subset NME (%), FR (%), AUC."""
    res = {"full": dict(NME=100 * nmes.mean(), FR=100 * failure_rate(nmes, thr), AUC=auc_ced(nmes, thr), n=len(nmes))}
    if attrs is not None:
        for j, name in enumerate(ATTR_NAMES):
            m = attrs[:, j] == 1
            if m.sum() > 0:
                res[name] = dict(NME=100 * nmes[m].mean(), FR=100 * failure_rate(nmes[m], thr), AUC=auc_ced(nmes[m], thr), n=int(m.sum()))
    return res


def format_table(res, title=""):
    lines = [title, f"{'subset':14s}{'n':>6s}{'NME%':>9s}{'FR%':>8s}{'AUC':>8s}"]
    for k, v in res.items():
        lines.append(f"{k:14s}{v['n']:6d}{v['NME']:9.3f}{v['FR']:8.2f}{v['AUC']:8.4f}")
    return "\n".join(lines)
