#!/usr/bin/env python
"""Evaluate a trained ARMOR-FLD model: WFLW subsets, the WFLW-C corruption grid and (optionally) PG-TTA.

Examples
  python scripts/evaluate.py --mode FULL --data-root data/WFLW --out-dir runs/full
  python scripts/evaluate.py --mode FULL --out-dir runs/full --severities 1 2 3 4 5
  python scripts/evaluate.py --mode FULL --out-dir runs/full --tta            # adds gated / forced / online columns
"""
import argparse
import os

from _common import add_common_args, build_cfg, prepare_data, device, save_json
from armor import train as T
from armor.degrade import DEG_TYPES
from armor.metrics import format_table
from armor.tta import PGTTA


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--severities", type=int, nargs="*", default=[1, 3, 5])
    p.add_argument("--tta", action="store_true", help="also run PG-TTA (gated), forced-2-steps and online variants")
    p.add_argument("--tta-steps", type=int, default=2, help="steps for the forced (no gating) variant")
    args = p.parse_args()
    cfg = build_cfg(args)
    dev = device()
    train_cache, test_cache, _ = prepare_data(cfg, args.synthetic)

    stage_a = T.load_stage_a(cfg, dev)
    prior = T.load_prior(cfg, stage_a, dev)
    model = T.load_landmark_net(cfg, prior, dev, "best")
    results = {"mode": args.mode, "severities": args.severities}

    res = T.evaluate(cfg, model, test_cache, dev, verbose=True)
    results["wflw_test"] = res
    results["wflw_c"] = T.wflw_c_benchmark(cfg, model, test_cache, dev, severities=tuple(args.severities))

    if args.tta:
        tta, det = T.build_tta(cfg, model, train_cache, dev, shift_mode=cfg["tta_shift_mode"])
        print("shift detector:", det.state_dict())
        forced = PGTTA(model, {**cfg, "tta_force_steps": args.tta_steps}, det)
        online = PGTTA(model, {**cfg, "tta_online": True}, det)
        rows = []
        print(f"{'cell':20s} {'NME':>7s} {'gated':>7s} {'forced':>7s} {'online':>7s} | {'z':>6s} {'steps':>5s}")
        for deg in [None] + [(t, s) for t in DEG_TYPES[1:] for s in args.severities]:
            r0 = T.evaluate(cfg, model, test_cache, dev, fixed_deg=deg)
            r1 = T.evaluate(cfg, model, test_cache, dev, fixed_deg=deg, tta=tta)
            r2 = T.evaluate(cfg, model, test_cache, dev, fixed_deg=deg, tta=forced)
            online.reset(); r3 = T.evaluate(cfg, model, test_cache, dev, fixed_deg=deg, tta=online); online.reset()
            row = dict(cell=str(deg or "clean"), NME=r0["full"]["NME"], gated=r1["full"]["NME"], forced=r2["full"]["NME"],
                       online=r3["full"]["NME"], z=r1["full"].get("tta_z", 0.0), steps=r1["full"].get("tta_steps", 0.0))
            rows.append(row)
            print(f"{row['cell']:20s} {row['NME']:7.3f} {row['gated']:7.3f} {row['forced']:7.3f} {row['online']:7.3f} | {row['z']:6.1f} {row['steps']:5.2f}")
        results["tta"] = rows
        results["shift_detector"] = det.state_dict()

    save_json(results, os.path.join(cfg["out_dir"], "results_eval.json"))


if __name__ == "__main__":
    main()
