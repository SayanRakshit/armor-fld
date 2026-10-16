#!/usr/bin/env python
"""Export the trained landmark network to ONNX and measure size / parameter count / latency.

Example
  python scripts/export_onnx.py --mode FULL --out-dir runs/full
Outputs <out-dir>/armor_fld_<mode>.onnx with inputs  crop (b,3,S,S) normalised RGB
                                            outputs coords (b,98,2) in crop pixels, visibility (b,98), unreliability (b,256)
"""
import argparse
import os
import time

import numpy as np
import torch

from _common import add_common_args, build_cfg, device, save_json
from armor import train as T


class ExportWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        o = self.m(x)
        return o["coords"], torch.sigmoid(o["vis_logit"]), o["alpha"]


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--threads", type=int, default=2, help="CPU threads for the latency measurement")
    args = p.parse_args()
    cfg = build_cfg(args)
    dev = device()

    stage_a = T.load_stage_a(cfg, dev)
    prior = T.load_prior(cfg, stage_a, dev)
    model = T.load_landmark_net(cfg, prior, dev, "best").eval()
    n_prior = sum(p.numel() for p in model.prior.parameters()) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_frozen = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad) / 1e6
    print(f"params: prior {n_prior:.2f}M | landmark branch {n_total - n_prior:.2f}M | total {n_total:.2f}M (frozen {n_frozen:.2f}M)")

    S = cfg["crop_size"]
    export_path = os.path.join(cfg["out_dir"], f"armor_fld_{args.mode.lower()}.onnx")
    kw = dict(input_names=["crop"], output_names=["coords", "visibility", "unreliability"],
              dynamic_axes={"crop": {0: "b"}, "coords": {0: "b"}, "visibility": {0: "b"}, "unreliability": {0: "b"}},
              opset_version=17)
    dummy = torch.randn(1, 3, S, S)
    wrapped = ExportWrapper(model).cpu().eval()
    try:
        torch.onnx.export(wrapped, dummy, export_path, dynamo=False, **kw)
    except TypeError:                       # older torch without the `dynamo` switch
        torch.onnx.export(wrapped, dummy, export_path, **kw)
    model.to(dev)
    size_mb = os.path.getsize(export_path) / 1e6
    print(f"ONNX: {export_path} ({size_mb:.1f} MB fp32)")

    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(export_path, so, providers=["CPUExecutionProvider"])
    x1 = np.random.randn(1, 3, S, S).astype(np.float32); x8 = np.repeat(x1, 8, 0)
    for _ in range(5):
        sess.run(None, {"crop": x1})
    def bench(fn, n=20):
        t0 = time.time()
        for _ in range(n):
            fn()
        return 1000 * (time.time() - t0) / n
    ms1, ms8 = bench(lambda: sess.run(None, {"crop": x1})), bench(lambda: sess.run(None, {"crop": x8})) / 8
    print(f"CPU ({args.threads} threads): {ms1:.1f} ms/face (b=1), {ms8:.1f} ms/face (b=8)")
    # numerical check against PyTorch
    with torch.no_grad():
        ref = wrapped(dummy)[0].numpy()
    onnx_out = sess.run(None, {"crop": dummy.numpy()})[0]
    print(f"max |onnx - torch| on coords: {np.abs(onnx_out - ref).max():.4f} px")
    save_json(dict(params_M=n_total, deployable_params_M=n_total - n_frozen, onnx_MB=size_mb, crop_size=S,
                   cpu_ms_b1=ms1, cpu_ms_b8=ms8, threads=args.threads), os.path.join(cfg["out_dir"], "results_export.json"))


if __name__ == "__main__":
    main()
