# ARMOR-FLD — Adverse-condition Reliability-Modulated ORFormer

Research code for *Face Landmark Detection Under Adverse Conditions* (Sayan Rakshit).
The Colab notebook `ARMOR_FLD_colab.ipynb` writes this package to disk and drives it; the modules
also work as plain Python (`python -c "from armor import train"`).

| module | what it does |
|---|---|
| `config.py`   | QUICK / FULL presets (plain dicts) |
| `common.py`   | WFLW constants (flip map, edge groups), exact affine crop/back-projection, heatmap renderers |
| `degrade.py`  | Degradation Bank: occlusion / blur / low-light / weather / compression with per-pixel unreliability maps and type maps (also defines WFLW-C) |
| `data.py`     | WFLW parsing, face-crop cache, `FaceDataset` (train aug + degradation targets; deterministic WFLW-C cells), synthetic fallback |
| `vq.py`       | Stage A: clean-face VQ edge-heatmap prior (encoder, EMA codebook with dead-code restarts, decoder) |
| `orformer.py` | Stage B: R-ORFormer — messenger tokens, reliability-supervised α, degradation-type head, reliability-biased attention |
| `fld.py`      | Stage C: lite stacked hourglass, reliability-gated prior fusion, windowed soft-argmax + covariance, visibility head |
| `losses.py`   | STAR-V (visibility-weighted, gated STAR), Adaptive Wing, stage objectives |
| `metrics.py`  | NME (inter-ocular), FR@0.1, AUC@0.1, per-attribute subsets |
| `tta.py`      | PG-TTA: VQ shift detector + prior-consistency / flip / entropy adaptation of norm layers and the gate |
| `detector.py` | SCRFD (InsightFace ONNX) decoder, full `FacePipeline`, affine round-trip test |
| `train.py`    | training loops A/B/C, evaluation, WFLW-C benchmark, TTA benchmark, checkpoint I/O |

Pipeline: `SCRFD → 1.3× square crop (128²) → [VQ prior → R-ORFormer → H_rec, α] + hourglass with gated fusion
→ STAR-V landmarks + visibility → exact affine back-projection`.
