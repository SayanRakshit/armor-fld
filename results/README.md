# Result files

Copy the JSON files written by the Colab notebook (Drive: `MyDrive/ARMOR_FLD/full/`) or by the scripts
(`runs/full/`) into this folder so the numbers in the top-level README can be checked:

| File | Produced by | Contents |
|---|---|---|
| `results_full.json` | notebook (FULL mode) / `scripts/train.py` + `scripts/evaluate.py` | WFLW subsets, WFLW-C grid (severities 1, 3, 5), model size |
| `results_full_tta_alpha.json` | notebook TTA cell / `scripts/evaluate.py --tta` | shift-detector calibration (μ, σ), gated / forced / online NME, z and steps per cell |
| `results_full_ablation.json` | notebook ablation helper / `scripts/ablate.py --name no_degradation_bank` | WFLW-C grid of the model trained without the degradation bank |
| `results_export.json` | `scripts/export_onnx.py` | parameters, ONNX size, CPU latency |

Checkpoints (`stage_a.pt`, `stage_b.pt`, `stage_c_best.pt`, ~100 MB together) are not tracked by git; attach them to a
GitHub release if you want to share them.
