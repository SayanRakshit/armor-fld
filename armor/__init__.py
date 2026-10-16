"""ARMOR-FLD: Adverse-condition Reliability-Modulated ORFormer for facial landmark detection.

Novel pipeline built on top of SCRFD (detection) + ORFormer (clean-face prior) + STAR loss:

  1. Degradation Bank with *reliability supervision*  (armor.degrade)
  2. R-ORFormer: reliability- and degradation-type-supervised messenger recovery (armor.orformer)
  3. Reliability-gated heatmap fusion + per-landmark visibility head + STAR-V loss (armor.fld, armor.losses)
  4. Shift detector (mean predicted unreliability) gating Prior-Guided Test-Time Adaptation (armor.tta)
  5. WFLW-C adverse-condition benchmark protocol (armor.train)
"""
__version__ = "0.1.0"
