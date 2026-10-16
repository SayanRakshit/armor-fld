"""Stage A - quantized clean-face edge-heatmap generator (VQ prior)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                  nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    """3x64x64 -> code_dim x16x16 (two stride-2 stages)."""

    def __init__(self, code_dim=128, ch=(64, 128), in_ch=3):
        super().__init__()
        c1, c2 = ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1), ResBlock(c1),
            nn.Conv2d(c1, c1, 4, stride=2, padding=1), ResBlock(c1),            # 32
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), ResBlock(c2), ResBlock(c2),  # 16
            nn.GroupNorm(8, c2), nn.SiLU(), nn.Conv2d(c2, code_dim, 1))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """code_dim x16x16 -> out_ch x64x64 edge heatmaps in [0,1]."""

    def __init__(self, code_dim=128, ch=(128, 64), out_ch=13):
        super().__init__()
        c1, c2 = ch
        self.net = nn.Sequential(
            nn.Conv2d(code_dim, c1, 3, padding=1), ResBlock(c1), ResBlock(c1),
            nn.ConvTranspose2d(c1, c2, 4, stride=2, padding=1), ResBlock(c2),      # 32
            nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1), ResBlock(c2),      # 64
            nn.GroupNorm(8, c2), nn.SiLU(), nn.Conv2d(c2, out_ch, 3, padding=1))

    def forward(self, z):
        return torch.sigmoid(self.net(z))


class VectorQuantizerEMA(nn.Module):
    """EMA codebook with dead-code restarts (stable even on short Colab schedules)."""

    def __init__(self, n_codes, dim, beta=0.25, decay=0.99, eps=1e-5, restart_after=200):
        super().__init__()
        self.n_codes, self.dim, self.beta, self.decay, self.eps = n_codes, dim, beta, decay, eps
        embed = torch.randn(n_codes, dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_sum", embed.clone())
        self.register_buffer("unused", torch.zeros(n_codes))
        self.restart_after = restart_after

    @torch.no_grad()
    def nearest(self, flat):  # flat: (M, dim)
        d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)[None])
        return d.argmin(1)

    def lookup(self, idx):
        return F.embedding(idx, self.embed)

    def forward(self, z_e):
        """z_e: (B,D,H,W). Returns z_q (straight-through), idx (B,H,W), commitment loss, perplexity."""
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D).float()
        idx = self.nearest(flat)
        z_q = self.lookup(idx).view(B, H, W, D).permute(0, 3, 1, 2)
        loss = self.beta * F.mse_loss(z_e.float(), z_q.detach())
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
                count = onehot.sum(0)
                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                n = self.ema_count.sum()
                cnt = (self.ema_count + self.eps) / (n + self.n_codes * self.eps) * n
                self.embed.copy_(self.ema_sum / cnt.unsqueeze(1))
                # dead-code restart: codes unused for `restart_after` steps are re-seeded from the batch
                self.unused.add_(1.0); self.unused[count > 0] = 0
                dead = self.unused > self.restart_after
                if dead.any():
                    k = int(dead.sum())
                    pick = flat[torch.randint(0, flat.shape[0], (k,), device=flat.device)]
                    self.embed[dead] = pick
                    self.ema_sum[dead] = pick
                    self.ema_count[dead] = 1.0
                    self.unused[dead] = 0
        z_q_st = z_e + (z_q.to(z_e.dtype) - z_e).detach()
        probs = torch.bincount(idx, minlength=self.n_codes).float() / idx.numel()
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return z_q_st, idx.view(B, H, W), loss, perplexity


class HeatmapVQ(nn.Module):
    """Encoder E + codebook C + decoder D (ORFormer stage A, trained on *clean* faces)."""

    def __init__(self, cfg, n_edges=13):
        super().__init__()
        self.enc = Encoder(cfg["code_dim"], cfg["enc_ch"])
        self.vq = VectorQuantizerEMA(cfg["n_codes"], cfg["code_dim"], cfg["vq_beta"], cfg["vq_decay"])
        self.dec = Decoder(cfg["code_dim"], cfg["enc_ch"][::-1], n_edges)

    def forward(self, x64):
        z_e = self.enc(x64)
        z_q, idx, vq_loss, ppl = self.vq(z_e)
        hm = self.dec(z_q)
        return dict(hm=hm, idx=idx, vq_loss=vq_loss, perplexity=ppl, z_e=z_e, z_q=z_q)

    @torch.no_grad()
    def encode_indices(self, x64):
        z_e = self.enc(x64)
        B, D, H, W = z_e.shape
        return self.vq.nearest(z_e.permute(0, 2, 3, 1).reshape(-1, D).float()).view(B, H * W)

    @torch.no_grad()
    def quant_error(self, x64):
        """Per-image mean squared distance to the nearest code = VQ shift score (used by PG-TTA)."""
        z_e = self.enc(x64).float()
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        idx = self.vq.nearest(flat)
        err = (flat - self.vq.lookup(idx)).pow(2).mean(1).view(B, -1).mean(1)
        return err
