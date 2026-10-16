"""Training loops (stages A/B/C), evaluation, the WFLW-C benchmark and the PG-TTA benchmark."""
import os
import time
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

from .vq import HeatmapVQ
from .orformer import ARMORPrior
from .fld import ARMORLandmarkNet
from .losses import stage_a_loss, stage_b_loss, stage_c_loss
from .metrics import nme_per_image, summarize, format_table
from .data import FaceDataset, make_loader
from .degrade import DEG_TYPES
from .tta import ShiftDetector, PGTTA
from .common import N_EDGES


# ---------------------------------------------------------------------------------- utils
def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def amp_ctx(cfg, device):
    use = cfg["amp"] and device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use), use


def cosine_lr(opt, base_lr, step, total, warmup=100):
    warmup = max(1, min(warmup, total // 10))
    lr = base_lr * step / warmup if step < warmup else base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def save_ckpt(path, **objs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in objs.items()}, path)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def x64(x):
    return F.interpolate(x, size=64, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------------- stage A
def train_stage_a(cfg, train_cache, device, log=print):
    seed_all(cfg["seed"])
    ds = FaceDataset(train_cache, cfg, train=True)
    dl = make_loader(ds, cfg["a_bs"], True, cfg)
    model = HeatmapVQ(cfg, N_EDGES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["a_lr"], weight_decay=1e-4)
    ctx, use_amp = amp_ctx(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = cfg["a_epochs"] * len(dl); step = 0; t0 = time.time()
    log(f"[A] HeatmapVQ {count_params(model):.2f}M params, {len(ds)} faces, {len(dl)} it/epoch")
    for ep in range(cfg["a_epochs"]):
        model.train()
        for b in dl:
            b = to_device(b, device)
            lr = cosine_lr(opt, cfg["a_lr"], step, total)
            with ctx:
                out = model(x64(b["img_clean"]))                  # stage A sees *clean* faces only
                loss, logs = stage_a_loss(out, b["edge_hm"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); step += 1
            if step % cfg["log_every"] == 0:
                log(f"[A] ep {ep} it {step}/{total} loss {loss.item():.4f} hm {logs['hm']:.4f} vq {logs['vq']:.4f} ppl {logs['ppl']:.1f} lr {lr:.2e} {time.time()-t0:.0f}s")
    save_ckpt(os.path.join(cfg["out_dir"], "stage_a.pt"), model=model, cfg=cfg)
    return model


# ---------------------------------------------------------------------------------- stage B
def train_stage_b(cfg, train_cache, stage_a, device, log=print):
    seed_all(cfg["seed"] + 1)
    ds = FaceDataset(train_cache, cfg, train=True)
    dl = make_loader(ds, cfg["b_bs"], True, cfg)
    prior = ARMORPrior(cfg, stage_a).to(device)
    params = [p for p in prior.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["b_lr"], weight_decay=1e-4)
    ctx, use_amp = amp_ctx(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = cfg["b_epochs"] * len(dl); step = 0; t0 = time.time()
    log(f"[B] R-ORFormer prior {count_params(prior.former):.2f}M (+enc {count_params(prior.enc):.2f}M) trainable")
    for ep in range(cfg["b_epochs"]):
        prior.train()
        for b in dl:
            b = to_device(b, device)
            lr = cosine_lr(opt, cfg["b_lr"], step, total)
            with ctx:
                idx_tgt = prior.target_codes(x64(b["img_clean"]))
                out = prior(x64(b["img"]))                        # degraded input
                loss, logs = stage_b_loss(out, idx_tgt, b["edge_hm"], b["unrel"], b["dtype"], cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); step += 1
            if step % cfg["log_every"] == 0:
                log(f"[B] ep {ep} it {step}/{total} loss {loss.item():.3f} code {logs['code']:.3f} hm {logs['hm']:.4f} "
                    f"rel {logs['rel']:.3f} type {logs['type']:.3f} accI {logs['accI']:.3f} relMAE {logs['relMAE']:.3f} {time.time()-t0:.0f}s")
    save_ckpt(os.path.join(cfg["out_dir"], "stage_b.pt"), prior=prior, cfg=cfg)
    return prior


# ---------------------------------------------------------------------------------- stage C
def train_stage_c(cfg, train_cache, test_cache, prior, device, log=print, eval_every=1):
    seed_all(cfg["seed"] + 2)
    ds = FaceDataset(train_cache, cfg, train=True)
    dl = make_loader(ds, cfg["c_bs"], True, cfg)
    model = ARMORLandmarkNet(cfg, prior, 98, N_EDGES).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["c_lr"], weight_decay=cfg["c_wd"])
    ctx, use_amp = amp_ctx(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = cfg["c_epochs"] * len(dl); step = 0; t0 = time.time(); best = 1e9
    log(f"[C] landmark net {count_params(model):.2f}M total, trainable {sum(p.numel() for p in params)/1e6:.2f}M")
    for ep in range(cfg["c_epochs"]):
        model.train()
        if not cfg["finetune_prior"]:
            model.prior.eval()
        for b in dl:
            b = to_device(b, device)
            lr = cosine_lr(opt, cfg["c_lr"], step, total, warmup=200)
            star_scale = min(1.0, step / max(1.0, cfg["star_warmup"] * total))
            with ctx:
                out = model(b["img"])
                loss, logs = stage_c_loss(out, b, cfg, star_scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); step += 1
            if step % cfg["log_every"] == 0:
                log(f"[C] ep {ep} it {step}/{total} loss {loss.item():.3f} star {logs['star']:.3f} awing {logs['awing']:.3f} "
                    f"edge {logs['edge']:.4f} vis {logs['vis']:.3f} lr {lr:.2e} {time.time()-t0:.0f}s")
        if (ep + 1) % eval_every == 0 or ep == cfg["c_epochs"] - 1:
            res = evaluate(cfg, model, test_cache, device)
            log(f"[C] epoch {ep}: test NME {res['full']['NME']:.3f}%  FR {res['full']['FR']:.2f}%  AUC {res['full']['AUC']:.4f}")
            if res["full"]["NME"] < best:
                best = res["full"]["NME"]
                save_ckpt(os.path.join(cfg["out_dir"], "stage_c_best.pt"), model=model, cfg=cfg)
    save_ckpt(os.path.join(cfg["out_dir"], "stage_c_last.pt"), model=model, cfg=cfg)
    return model


# ---------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(cfg, model, loader, device, tta=None):
    model.eval()
    preds, gts, attrs, vis_p, vis_t, steps, zs = [], [], [], [], [], [], []
    ctx, _ = amp_ctx(cfg, device)
    for b in loader:
        b = to_device(b, device)
        if tta is not None:
            out = tta(b["img"]); steps.append(out["tta_steps"]); zs.append(float(out["shift_z"].median()))
        else:
            with ctx:
                out = model(b["img"])
        preds.append(out["coords"].float().cpu().numpy()); gts.append(b["pts"].cpu().numpy())
        attrs.append(b["attrs"].cpu().numpy())
        vis_p.append(torch.sigmoid(out["vis_logit"].float()).cpu().numpy()); vis_t.append(b["vis"].cpu().numpy())
    return dict(pred=np.concatenate(preds), gt=np.concatenate(gts), attrs=np.concatenate(attrs),
                vis_p=np.concatenate(vis_p), vis_t=np.concatenate(vis_t), steps=steps, zs=zs)


def evaluate(cfg, model, test_cache, device, fixed_deg=None, tta=None, verbose=False):
    ds = FaceDataset(test_cache, cfg, train=False, fixed_deg=fixed_deg, return_clean_targets=False)
    dl = make_loader(ds, 64, False, cfg)
    P = predict(cfg, model, dl, device, tta)
    nmes = nme_per_image(P["pred"], P["gt"])
    res = summarize(nmes, P["attrs"])
    vis_auc = _vis_accuracy(P["vis_p"], P["vis_t"])
    res["full"]["vis_acc"] = vis_auc
    if P["steps"]:
        res["full"]["tta_steps"] = float(np.mean(P["steps"]))
        res["full"]["tta_z"] = float(np.median(P["zs"]))
    if verbose:
        print(format_table(res, title=f"WFLW test {fixed_deg or ''}"))
    return res


def _vis_accuracy(p, t, thr=0.5):
    """Balanced accuracy of visibility prediction (only meaningful on degraded test cells)."""
    tb = t >= thr; pb = p >= thr
    if tb.all() or (~tb).all():
        return float((pb == tb).mean())
    return 0.5 * (float((pb & tb).sum() / tb.sum()) + float((~pb & ~tb).sum() / (~tb).sum()))


# ---------------------------------------------------------------------------------- WFLW-C benchmark
def wflw_c_benchmark(cfg, model, test_cache, device, severities=(1, 3, 5), types=None, tta=None, log=print):
    """NME on every (degradation type, severity) cell of the test set.  Returns {type: {sev: NME%}}."""
    types = types or DEG_TYPES[1:]
    table = {"clean": {0: evaluate(cfg, model, test_cache, device, None, tta)["full"]["NME"]}}
    log(f"clean: NME {table['clean'][0]:.3f}%")
    for t in types:
        table[t] = {}
        for s in severities:
            r = evaluate(cfg, model, test_cache, device, (t, s), tta)
            table[t][s] = r["full"]["NME"]
            log(f"{t:12s} sev {s}: NME {r['full']['NME']:.3f}%  FR {r['full']['FR']:.2f}%  vis-acc {r['full']['vis_acc']:.3f}"
                + (f"  TTA steps {r['full']['tta_steps']:.2f}" if 'tta_steps' in r['full'] else ""))
    mean_c = float(np.mean([v for t in types for v in table[t].values()]))
    table["mean_corrupted"] = {0: mean_c}
    log(f"mean corrupted NME {mean_c:.3f}%")
    return table


def build_tta(cfg, model, train_cache, device, shift_mode="alpha"):
    det = ShiftDetector(model.prior, mode=shift_mode)
    calib = FaceDataset(train_cache, cfg, train=False)                 # clean centre crops for calibration
    det.fit(make_loader(calib, 64, False, cfg), device)
    return PGTTA(model, cfg, det), det


# ---------------------------------------------------------------------------------- loading
def load_stage_a(cfg, device):
    ck = torch.load(os.path.join(cfg["out_dir"], "stage_a.pt"), map_location=device, weights_only=False)
    m = HeatmapVQ(ck["cfg"], N_EDGES).to(device); m.load_state_dict(ck["model"]); return m.eval()


def load_prior(cfg, stage_a, device):
    ck = torch.load(os.path.join(cfg["out_dir"], "stage_b.pt"), map_location=device, weights_only=False)
    p = ARMORPrior(ck["cfg"], stage_a).to(device); p.load_state_dict(ck["prior"]); return p.eval()


def load_landmark_net(cfg, prior, device, which="best"):
    ck = torch.load(os.path.join(cfg["out_dir"], f"stage_c_{which}.pt"), map_location=device, weights_only=False)
    m = ARMORLandmarkNet(ck["cfg"], prior, 98, N_EDGES).to(device); m.load_state_dict(ck["model"]); return m.eval()
