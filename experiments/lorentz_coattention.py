"""
v1 hyperbolic co-attention block (ViLBERT-style co-attention)

One block per stream (image, text), pre-norm:
    x = x (+) SelfAttn(LN(x))
    img, txt = img (+) CrossAttn(LN(img) -> LN(txt)),  txt (+) CrossAttn(LN(txt) -> LN(img))   [parallel]
    x = x (+) FFN(LN(x))
(+) = add space parts, recompute time.
"""
import math

import torch
from torch import nn
import hypercore.nn as hnn


def lorentz_add(x, y, c):
    """Residual: add space parts, recompute time like Hypformer"""
    space = x[..., 1:] + y[..., 1:]
    time = (space.pow(2).sum(dim=-1, keepdim=True) + c).sqrt()
    return torch.cat([time, space], dim=-1)


class LorentzAttention(nn.Module):
    """Single-head attention on the hyperboloid. Self-attention: q_in is kv_in. Cross: different streams."""

    def __init__(self, manifold, dim, scale_mult=4.0):
        super().__init__()
        self.manifold = manifold
        # LorentzLinear: Euclidean linear on the full point (incl. time) -> new space part -> recompute time.
        self.Wq = hnn.LorentzLinear(manifold, dim, dim - 1)
        self.Wk = hnn.LorentzLinear(manifold, dim, dim - 1)
        self.Wv = hnn.LorentzLinear(manifold, dim, dim - 1)
        # Learned divisor. HyperCore inits it at sqrt(dim), which makes attention near one-hot at init
        # (entropy ratio ~0.15 at dim 513). 4x matches flat dot-product attention's sharpness at init (~0.7),
        # so the flat twin and this block start comparable. Measured on random inputs only.
        # No additive bias (HyperCore has one): softmax ignores a constant shift, so it could never learn.
        self.scale = nn.Parameter(torch.tensor(scale_mult * math.sqrt(dim)))

    def forward(self, q_in, kv_in, key_pad=None):
        """q_in [B, Nq, dim], kv_in [B, Nk, dim], key_pad [B, Nk] bool, True = padding (ignored)."""
        q, k, v = self.Wq(q_in), self.Wk(kv_in), self.Wv(kv_in)
        # 2c + 2<q,k>_L = -(squared Lorentzian distance): closer key -> higher score.  [B, Nq, Nk]
        scores = (2 * self.manifold.c + 2 * self.manifold.cinner(q, k)) / self.scale
        if key_pad is not None:
            scores = scores.masked_fill(key_pad[:, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        # Lorentzian centroid for weighted sum of values  [B, Nq, dim]
        return self.manifold.lorentzian_centroid(v, attn), attn


class LorentzFFN(nn.Module):
    """HyperCore's transformer MLP: LorentzLinear -> GELU on space part -> LorentzLinear. Hidden = 4x space dim."""

    def __init__(self, manifold, dim):
        super().__init__()
        hidden = 4 * (dim - 1)
        self.fc = hnn.LorentzLinear(manifold, dim, hidden)
        self.act = hnn.LorentzActivation(manifold, activation=nn.GELU())
        self.proj = hnn.LorentzLinear(manifold, hidden + 1, dim - 1)

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


def _stream(manifold, dim, scale_mult):
    """All sublayers for one stream. Each stream has its own weights (as in ViLBERT)."""
    return nn.ModuleDict(dict(
        ln_self=hnn.LorentzLayerNorm(manifold, dim - 1),
        self_attn=LorentzAttention(manifold, dim, scale_mult),
        ln_cross=hnn.LorentzLayerNorm(manifold, dim - 1),
        cross_attn=LorentzAttention(manifold, dim, scale_mult),
        ln_ffn=hnn.LorentzLayerNorm(manifold, dim - 1),
        ffn=LorentzFFN(manifold, dim),
    ))


class LorentzCoAttentionBlock(nn.Module):
    def __init__(self, manifold, dim, scale_mult=4.0):
        super().__init__()
        self.manifold = manifold
        self.img = _stream(manifold, dim, scale_mult)
        self.txt = _stream(manifold, dim, scale_mult)

    def forward(self, img, txt, txt_pad=None, return_attn=False):
        """img [B, Ni, dim], txt [B, Nt, dim], txt_pad [B, Nt] bool (True = padding). Images have no padding."""
        c = self.manifold.c
        I, T = self.img, self.txt
        # Force float32 for all Lorentz math, as MERU does (hyperboloid constraint is fragile in bf16).
        with torch.autocast(device_type=img.device.type, enabled=False):
            img, txt = img.float(), txt.float()

            # 1. Self-attention within each stream.
            a_ii, w_ii = I.self_attn(I.ln_self(img), I.ln_self(img))
            a_tt, w_tt = T.self_attn(T.ln_self(txt), T.ln_self(txt), key_pad=txt_pad)
            img, txt = lorentz_add(img, a_ii, c), lorentz_add(txt, a_tt, c)

            # 2. Cross-attention, both directions, both computed from the same (pre-update) inputs.
            li, lt = I.ln_cross(img), T.ln_cross(txt)
            a_it, w_it = I.cross_attn(li, lt, key_pad=txt_pad)  # image queries -> text keys/values
            a_ti, w_ti = T.cross_attn(lt, li)                   # text queries -> image keys/values
            img, txt = lorentz_add(img, a_it, c), lorentz_add(txt, a_ti, c)

            # 3. Feed-forward within each stream.
            img = lorentz_add(img, I.ffn(I.ln_ffn(img)), c)
            txt = lorentz_add(txt, T.ffn(T.ln_ffn(txt)), c)

        if return_attn:
            return img, txt, dict(img_self=w_ii, txt_self=w_tt, img_to_txt=w_it, txt_to_img=w_ti)
        return img, txt