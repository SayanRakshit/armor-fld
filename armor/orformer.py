"""Stage B - R-ORFormer: reliability- and degradation-type-supervised messenger-token recovery.

Differences from the original ORFormer (Chiang et al., WACV 2025):
  * the per-patch score alpha is supervised with the degradation bank's unreliability map
    (BCE), so it becomes a calibrated *reliability* estimate for any degradation, not only occlusion;
  * a degradation-type head predicts what corrupted each patch and its embedding conditions the
    messenger token before the next layer (type-conditioned recovery: a blurred patch still carries
    evidence, a fully occluded one does not);
  * reliability of the previous layer biases attention logits so unreliable patches stop contaminating
    the context that is used to recover them.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import HeatmapVQ


class Attention(nn.Module):
    """Multi-head attention with an additive float mask broadcast as (B, 1|heads, L, S)."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.drop = drop

    def forward(self, xq, xkv, mask=None):
        B, L, d = xq.shape
        S = xkv.shape[1]
        q = self.q(xq).view(B, L, self.heads, self.dh).transpose(1, 2)
        kv = self.kv(xkv).view(B, S, 2, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        if mask is not None:
            mask = mask.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, L, d))


class ReliabilityLayer(nn.Module):
    def __init__(self, d, heads, ffn_mult, drop, n_types):
        super().__init__()
        self.ln_x1, self.ln_x2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ln_m1, self.ln_m2, self.ln_mx = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, heads, drop)
        self.cross_attn = Attention(d, heads, drop)
        self.ffn_x = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Dropout(drop), nn.Linear(d * ffn_mult, d))
        self.ffn_m = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Dropout(drop), nn.Linear(d * ffn_mult, d))
        # reliability score from the disagreement between a patch and its messenger
        self.rel_head = nn.Sequential(nn.Linear(3 * d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.type_head = nn.Sequential(nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, n_types))
        self.type_emb = nn.Linear(n_types, d, bias=False)

    def forward(self, X, M, alpha_prev, diag_mask):
        # alpha_prev (B,S): previous-layer unreliability -> additive key bias log(1 - alpha)
        bias = None if alpha_prev is None else torch.log1p(-alpha_prev.clamp(max=0.95))[:, None, None, :]
        h = self.ln_x1(X)
        X = X + self.self_attn(h, h, bias)                               # patch tokens, unreliable keys down-weighted
        X = X + self.ffn_x(self.ln_x2(X))
        mask = diag_mask if bias is None else (bias + diag_mask)         # messenger i never attends patch i
        M = M + self.cross_attn(self.ln_m1(M), self.ln_mx(X), mask)
        M = M + self.ffn_m(self.ln_m2(M))
        feat = torch.cat([X, M, (X - M).abs()], -1)
        alpha = torch.sigmoid(self.rel_head(feat).float()).squeeze(-1)   # (B,S) unreliability in [0,1]
        type_logits = self.type_head(torch.cat([X, M], -1))              # (B,S,K)
        M = M + self.type_emb(F.softmax(type_logits.float(), -1).to(M.dtype))  # type-conditioned recovery
        return X, M, alpha, type_logits


class RORFormer(nn.Module):
    def __init__(self, code_dim, n_codes, d=192, n_layers=3, heads=6, ffn_mult=4, drop=0.1, n_tokens=256, n_types=6):
        super().__init__()
        self.in_proj = nn.Linear(code_dim, d)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, d) * 0.02)
        self.msg = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList([ReliabilityLayer(d, heads, ffn_mult, drop, n_types) for _ in range(n_layers)])
        self.ln_x, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d)
        self.code_I = nn.Linear(d, n_codes)
        self.code_M = nn.Linear(d, n_codes)
        diag = torch.zeros(1, 1, n_tokens, n_tokens)
        diag[0, 0].fill_diagonal_(-1e4)
        self.register_buffer("diag_mask", diag)

    def forward(self, z_e):
        """z_e: (B,D,16,16) encoder features of the (possibly degraded) face."""
        B, D, H, W = z_e.shape
        X = self.in_proj(z_e.flatten(2).transpose(1, 2)) + self.pos
        M = self.msg.expand(B, H * W, -1) + self.pos
        alpha, alphas, types = None, [], []
        for layer in self.layers:
            X, M, alpha, tl = layer(X, M, alpha, self.diag_mask)
            alphas.append(alpha); types.append(tl)
        return dict(logits_I=self.code_I(self.ln_x(X)), logits_M=self.code_M(self.ln_m(M)),
                    alpha=alpha, alphas=alphas, type_logits=types[-1])


class ARMORPrior(nn.Module):
    """Encoder (fine-tuned) + R-ORFormer + frozen codebook & decoder -> degradation-free edge heatmaps."""

    def __init__(self, cfg, stage_a: HeatmapVQ, n_types=6):
        super().__init__()
        self.enc = copy.deepcopy(stage_a.enc)                              # fine-tuned in stage B
        self.enc_tgt = copy.deepcopy(stage_a.enc).requires_grad_(False)    # frozen: code targets + shift detector
        self.vq = copy.deepcopy(stage_a.vq).requires_grad_(False)
        self.dec = copy.deepcopy(stage_a.dec).requires_grad_(False)
        self.former = RORFormer(cfg["code_dim"], cfg["n_codes"], cfg["d_model"], cfg["n_layers"],
                                cfg["n_heads"], cfg["ffn_mult"], cfg["drop"], 256, n_types)
        self.n_codes = cfg["n_codes"]

    def train(self, mode=True):
        super().train(mode)
        self.enc_tgt.eval(); self.vq.eval(); self.dec.eval()                # frozen parts always in eval mode
        return self

    def _retrieve(self, logits):
        """Straight-through code retrieval: hard code features forward, gradient via the soft distribution."""
        soft = F.softmax(logits.float(), -1)
        soft_feat = soft @ self.vq.embed
        hard = self.vq.lookup(logits.argmax(-1))
        return hard + (soft_feat - soft_feat.detach())

    @torch.no_grad()
    def target_codes(self, x64_clean):
        z = self.enc_tgt(x64_clean)
        B, D, H, W = z.shape
        return self.vq.nearest(z.permute(0, 2, 3, 1).reshape(-1, D).float()).view(B, H * W)

    @torch.no_grad()
    def shift_score(self, x64):
        """VQ quantization error of the frozen clean-face encoder (large = far from the clean manifold)."""
        z = self.enc_tgt(x64).float()
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        idx = self.vq.nearest(flat)
        return (flat - self.vq.lookup(idx)).pow(2).mean(1).view(B, -1).mean(1)

    def forward(self, x64):
        z_e = self.enc(x64)
        B, D, H, W = z_e.shape
        o = self.former(z_e)
        Z_I = self._retrieve(o["logits_I"])                                # (B,S,D) from the patch itself
        Z_M = self._retrieve(o["logits_M"])                                # (B,S,D) from the context
        a = o["alpha"].unsqueeze(-1)
        Z_rec = (1 - a) * Z_I + a * Z_M
        hm = self.dec(Z_rec.transpose(1, 2).reshape(B, D, H, W))
        o.update(hm=hm, z_rec=Z_rec)
        return o
