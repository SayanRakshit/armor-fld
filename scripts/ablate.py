#!/usr/bin/env python
"""Ablations: retrain stages B and C with one component removed and run the WFLW-C grid.

  no_degradation_bank          train on clean + geometric augmentation only (p_degrade = 0)
  no_reliability_supervision   degradation bank on, alpha / type / visibility supervision off
  no_visibility_weighting      plain STAR (no visibility weighting, no visibility loss)
  no_shift_gating              not a retrain: evaluate TTA with a fixed number of steps (see evaluate.py --tta)

Example
  python scripts/ablate.py --mode FULL --data-root data/WFLW --out-dir runs/full --name no_degradation_bank
The stage-A prior is shared with the main run in --out-dir; the ablation writes under <out-dir>/ablation_<name>/.
"""
import argparse
import os

from _common import add_common_args, build_cfg, prepare_data, device, save_json
from armor import train as T

ABLATIONS = {
    "no_degradation_bank": dict(p_degrade=0.0),
    "no_reliability_supervision": dict(w_rel=0.0, w_type=0.0, w_vis=0.0, vis_floor=1.0),
    "no_visibility_weighting": dict(w_vis=0.0, vis_floor=1.0),
}


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--name", required=True, choices=sorted(ABLATIONS))
    p.add_argument("--severities", type=int, nargs="*", default=[1, 3, 5])
    args = p.parse_args()
    cfg = build_cfg(args)
    dev = device()
    train_cache, test_cache, _ = prepare_data(cfg, args.synthetic)

    stage_a = T.load_stage_a(cfg, dev)                       # shared clean prior from the main run
    acfg = dict(cfg); acfg.update(ABLATIONS[args.name])
    acfg["out_dir"] = os.path.join(cfg["out_dir"], f"ablation_{args.name}")
    os.makedirs(acfg["out_dir"], exist_ok=True)
    prior = T.train_stage_b(acfg, train_cache, stage_a, dev)
    model = T.train_stage_c(acfg, train_cache, test_cache, prior, dev, eval_every=max(1, acfg["c_epochs"] // 6))
    table = T.wflw_c_benchmark(acfg, model, test_cache, dev, severities=tuple(args.severities))
    print(f"\n[{args.name}] clean NME {table['clean'][0]:.3f} | mean corrupted NME {table['mean_corrupted'][0]:.3f}")
    save_json({"ablation": args.name, "overrides": ABLATIONS[args.name], "wflw_c": table},
              os.path.join(acfg["out_dir"], "results_ablation.json"))


if __name__ == "__main__":
    main()
