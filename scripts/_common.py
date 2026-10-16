"""Shared helpers for the command-line scripts (argument parsing, data preparation, logging)."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from armor.config import get_config  # noqa: E402
from armor.data import parse_wflw_list, wflw_lists, build_cache, make_synthetic_cache  # noqa: E402


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--mode", default="QUICK", choices=["QUICK", "FULL", "BASE"], help="configuration preset")
    p.add_argument("--data-root", default="data/WFLW", help="folder holding WFLW_images/ and WFLW_annotations/")
    p.add_argument("--cache-dir", default="cache", help="where the pre-cropped face cache is written")
    p.add_argument("--out-dir", default="runs/armor", help="checkpoints and result files")
    p.add_argument("--synthetic", action="store_true", help="use synthetic faces (code check only, no WFLW needed)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any config value, e.g. --set c_epochs=5 crop_size=256 hg_stacks=4")
    return p


def parse_overrides(items):
    out = {}
    for kv in items:
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build_cfg(args):
    import torch
    overrides = {}
    if args.synthetic and not torch.cuda.is_available():
        overrides.update(CPU_CHECK)              # --set KEY=VALUE still wins
    overrides.update(parse_overrides(args.set))
    cfg = get_config(args.mode, data_root=args.data_root, cache_dir=args.cache_dir, out_dir=args.out_dir, **overrides)
    if not torch.cuda.is_available():           # CPU: no mixed precision, no worker processes (keeps memory low)
        cfg["amp"] = False
        cfg["num_workers"] = 0
    return cfg


# light settings for a CPU-only code check (used by --synthetic when no GPU is present)
CPU_CHECK = dict(a_epochs=1, b_epochs=1, c_epochs=1, a_bs=16, b_bs=8, c_bs=8, log_every=5,
                 code_dim=64, n_codes=64, enc_ch=(32, 64), d_model=96, n_layers=2, n_heads=4,
                 hg_ch=64, hg_depth=2, train_subset=160, test_subset=40, tta_max_steps=2)


def prepare_data(cfg, synthetic=False):
    """Returns (train_cache, test_cache, test_items). Builds the face-crop cache on first use."""
    if synthetic:
        return (make_synthetic_cache(cfg["train_subset"] or 2000, cfg["cache_size"], seed=0),
                make_synthetic_cache(cfg["test_subset"] or 500, cfg["cache_size"], seed=1), None)
    tr_list, te_list = wflw_lists(cfg["data_root"])
    for f in (tr_list, te_list):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found. See README 'Data' for how to lay out WFLW.")
    tr_items, te_items = parse_wflw_list(tr_list), parse_wflw_list(te_list)
    img_root = os.path.join(cfg["data_root"], "WFLW_images")
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    tag = f"s{cfg['cache_size']}_{cfg['train_subset'] or 'all'}"
    train_cache = build_cache(tr_items, img_root, os.path.join(cfg["cache_dir"], f"train_{tag}"),
                              cfg["cache_size"], cfg["cache_scale"], cfg["train_subset"])
    test_cache = build_cache(te_items, img_root, os.path.join(cfg["cache_dir"], f"test_{tag}"),
                             cfg["cache_size"], cfg["cache_scale"], cfg["test_subset"])
    return train_cache, test_cache, te_items


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("saved", path)
