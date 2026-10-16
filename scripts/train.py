#!/usr/bin/env python
"""Train ARMOR-FLD end to end (stage A: clean-face VQ prior, B: R-ORFormer, C: landmark network).

Examples
  python scripts/train.py --mode QUICK --data-root data/WFLW --out-dir runs/quick
  python scripts/train.py --mode FULL  --data-root data/WFLW --out-dir runs/full
  python scripts/train.py --synthetic --set c_epochs=1 a_epochs=1 b_epochs=1     # 2-minute code check
Finished stages are loaded from --out-dir instead of retrained; delete a checkpoint to redo its stage.
"""
import argparse
import os

from _common import add_common_args, build_cfg, prepare_data, device, save_json
from armor import train as T


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--stages", default="ABC", help="which stages to run, e.g. 'C' to retrain only the landmark network")
    p.add_argument("--eval-every", type=int, default=0, help="evaluate stage C every N epochs (0 = ~6 times per run)")
    args = p.parse_args()
    cfg = build_cfg(args)
    dev = device()
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print("device:", dev, "| out_dir:", cfg["out_dir"], "| synthetic:", args.synthetic)
    train_cache, test_cache, _ = prepare_data(cfg, args.synthetic)
    print("train faces:", len(train_cache["attrs"]), "| test faces:", len(test_cache["attrs"]))

    ck = lambda name: os.path.join(cfg["out_dir"], name)  # noqa: E731
    if "A" in args.stages and not os.path.exists(ck("stage_a.pt")):
        stage_a = T.train_stage_a(cfg, train_cache, dev)
    else:
        stage_a = T.load_stage_a(cfg, dev); print("loaded", ck("stage_a.pt"))
    if "B" in args.stages and not os.path.exists(ck("stage_b.pt")):
        prior = T.train_stage_b(cfg, train_cache, stage_a, dev)
    else:
        prior = T.load_prior(cfg, stage_a, dev); print("loaded", ck("stage_b.pt"))
    if "C" in args.stages and not os.path.exists(ck("stage_c_best.pt")):
        every = args.eval_every or max(1, cfg["c_epochs"] // 6)
        model = T.train_stage_c(cfg, train_cache, test_cache, prior, dev, eval_every=every)
    else:
        model = T.load_landmark_net(cfg, prior, dev, "best"); print("loaded", ck("stage_c_best.pt"))

    res = T.evaluate(cfg, model, test_cache, dev, verbose=True)
    save_json({"mode": args.mode, "synthetic": args.synthetic, "wflw_test": res}, ck("results_train.json"))


if __name__ == "__main__":
    main()
