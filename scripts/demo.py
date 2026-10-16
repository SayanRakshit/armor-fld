#!/usr/bin/env python
"""Run the full pipeline on one image: SCRFD-500M -> face crops -> ARMOR-FLD landmarks -> back-projection.

Examples
  python scripts/demo.py --mode FULL --out-dir runs/full --image photo.jpg
  python scripts/demo.py --mode FULL --out-dir runs/full --image photo.jpg --tta --save out.jpg
Landmarks predicted non-visible (occluded) are drawn in red, visible ones in yellow.
"""
import argparse
import os
import time

import cv2
import numpy as np

from _common import add_common_args, build_cfg, prepare_data, device
from armor import train as T
from armor.data import make_synthetic_cache
from armor.detector import download_scrfd, SCRFD, FacePipeline


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--image", default=None, help="input image (default: a synthetic face, for a code check)")
    p.add_argument("--save", default=None, help="where to write the annotated image (default: <out-dir>/demo_<image>)")
    p.add_argument("--detector", default="det_500m", choices=["det_500m", "det_10g"], help="SCRFD variant")
    p.add_argument("--models-dir", default="models", help="where the SCRFD ONNX file is cached")
    p.add_argument("--score", type=float, default=0.5, help="face detection score threshold")
    p.add_argument("--tta", action="store_true", help="enable PG-TTA (needs the training cache to fit the shift detector)")
    args = p.parse_args()
    cfg = build_cfg(args)
    dev = device()

    stage_a = T.load_stage_a(cfg, dev)
    prior = T.load_prior(cfg, stage_a, dev)
    model = T.load_landmark_net(cfg, prior, dev, "best")
    tta = None
    if args.tta:
        train_cache, _, _ = prepare_data(cfg, args.synthetic)
        tta, det = T.build_tta(cfg, model, train_cache, dev, shift_mode=cfg["tta_shift_mode"])
        print("shift detector:", det.state_dict())

    onnx_path = download_scrfd(args.detector, root=args.models_dir)
    detector = SCRFD(onnx_path, input_size=640)
    print(f"SCRFD: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB) providers {detector.session.get_providers()}")

    if args.image is None:
        syn = make_synthetic_cache(1, 480, seed=5)
        args.image = os.path.join(cfg["out_dir"], "synthetic_face.jpg")
        os.makedirs(cfg["out_dir"], exist_ok=True)
        cv2.imwrite(args.image, cv2.cvtColor(np.asarray(syn["img"][0]), cv2.COLOR_RGB2BGR))
        args.score = min(args.score, 0.15)       # the cartoon face is a weak detection
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    pipe = FacePipeline(detector, model, cfg, dev, tta=tta)
    pipe.run(img_bgr, score_thresh=args.score)                       # warm-up
    t0 = time.time(); results = pipe.run(img_bgr, score_thresh=args.score); dt = 1000 * (time.time() - t0)
    print(f"{len(results)} face(s) in {dt:.1f} ms ({dev.type}{', PG-TTA on' if tta else ''})")
    for r in results:
        print(f"  box {np.round(r['box']).astype(int)} score {r['score']:.2f} | mean visibility {r['visibility'].mean():.2f} "
              f"| mean patch reliability {r['reliability'].mean():.2f}")

    save = args.save or os.path.join(cfg["out_dir"], "demo_" + os.path.basename(args.image))
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    cv2.imwrite(save, FacePipeline.draw(img_bgr, results))
    print("saved", save)


if __name__ == "__main__":
    main()
