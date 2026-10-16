"""Experiment configuration presets.

QUICK  : sanity run that finishes on a free Colab T4 in well under an hour (subset of WFLW, few epochs).
FULL   : the real schedule (all 7,500 training faces) - several hours on a T4, ~1.5-2 h on an A100.

Everything is a plain dict so it can be edited inline in the notebook.
"""
import copy

N_LANDMARKS = 98          # WFLW
N_EDGES = 13              # edge/boundary heatmaps rendered from the 98 points (see armor.common.EDGE_GROUPS)
N_DEG_TYPES = 6           # clean, occlusion, blur, lowlight, weather, compression

BASE = dict(
    # ---------------- data ----------------
    data_root="/content/data/WFLW",   # where WFLW_images/ and WFLW_annotations/ live
    cache_dir="/content/cache",       # pre-cropped face cache (fast training on Colab)
    cache_size=160,                   # cached crop resolution (contains the 128 training crop + margin)
    cache_scale=1.6,                  # cached square side = cache_scale * max(box_w, box_h)
    crop_scale=1.3,                   # training / inference crop side = crop_scale * max(box_w, box_h)
    crop_size=128,                    # network input resolution (ORFormer branch sees crop_size // 2 = 64)
    heatmap_size=64,                  # landmark / edge heatmap resolution
    lmk_sigma=2.0,                    # gaussian sigma (heatmap pixels) for landmark heatmaps
    edge_sigma=1.0,                   # gaussian sigma for edge heatmaps
    train_subset=None,                # int -> use only the first N training faces (QUICK mode)
    test_subset=None,
    num_workers=2,
    # ---------------- augmentation ----------------
    aug_rot=30.0, aug_scale=0.10, aug_shift=0.05, aug_flip=0.5,
    aug_color=0.3,                    # brightness/contrast jitter strength
    p_degrade=0.85,                   # probability that the degradation bank is applied to a training crop
    # ---------------- VQ prior (stage A) ----------------
    code_dim=128, n_codes=512, enc_ch=(64, 128), vq_beta=0.25, vq_decay=0.99,
    a_epochs=12, a_lr=7e-4, a_bs=128,
    # ---------------- R-ORFormer (stage B) ----------------
    d_model=192, n_layers=3, n_heads=6, ffn_mult=4, drop=0.1,
    b_epochs=10, b_lr=2e-4, b_bs=64,
    w_code=1.0, w_hm=50.0, w_rel=1.0, w_type=0.5,
    # ---------------- Landmark net (stage C) ----------------
    hg_stacks=2, hg_depth=3, hg_ch=128,
    c_epochs=30, c_lr=1e-3, c_bs=32, c_wd=1e-5,
    finetune_prior=False,             # keep the clean-face prior frozen in stage C (ORFormer-style)
    softargmax_beta=10.0, softargmax_window=9,   # windowed soft-argmax (no centre bias, local covariance for STAR)
    w_star=1.0, w_awing=1.0, w_vis=0.5, w_edge=1.0,  # AWing forms the peaks fast; STAR-V refines them (sub-pixel + anisotropy)
    star_detach=True, star_vr=0.0, vis_floor=0.3,
    star_warmup=0.1,                  # fraction of stage-C steps over which the STAR-V weight ramps 0 -> w_star
    star_gate_px=4,                   # STAR-V only refines peaks already within 4 heat-map px of the target (0 = off)
    # ---------------- test-time adaptation ----------------
    tta_lr=1e-3, tta_max_steps=4, tta_online=False,
    tta_w_prior=1.0, tta_w_flip=1.0, tta_w_ent=0.1,
    tta_shift_mode="alpha",           # "alpha": R-ORFormer mean unreliability (default) | "vq": codebook quantisation error
    tta_z_per_step=0.5,               # one extra TTA step per +0.5 sigma of the shift score (clean faces stay at 0)
    tta_force_steps=None,             # int -> ignore the shift detector and always take this many steps (ablation)
    # ---------------- misc ----------------
    seed=0, amp=True, log_every=50,
    out_dir="/content/drive/MyDrive/ARMOR_FLD",  # checkpoints + results (Drive so they survive a disconnect)
)

QUICK = copy.deepcopy(BASE)
QUICK.update(dict(train_subset=1500, test_subset=600,
                  a_epochs=4, b_epochs=4, c_epochs=15,
                  n_codes=256, c_bs=32, log_every=20))

FULL = copy.deepcopy(BASE)
FULL.update(dict(a_epochs=30, b_epochs=20, c_epochs=60, n_codes=1024, code_dim=192))


def get_config(name="QUICK", **overrides):
    cfg = copy.deepcopy({"QUICK": QUICK, "FULL": FULL, "BASE": BASE}[name.upper()])
    cfg.update(overrides)
    return cfg
