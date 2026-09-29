"""
LorentzCrossAttention implements cross-modal multi-head attention in Lorentzian geometry.

Based on:
    - Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space (https://arxiv.org/abs/2407.01290)
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/hridaymeka/hyperbolic_vlm/HyperCore')
from hypercore.manifolds import Lorentz


class LorentzCrossAttention(nn.Module):
    """
    Cross-modal attention in Lorentzian geometry.

    Queries are drawn from one modality (e.g. text tokens) and keys/values
    from another (e.g. image patches). Query and source sequences may have
    different lengths.

    Args:
        manifold (Lorentz): Lorentzian manifold.
        d_model (int): Total model dimension. Must be divisible by n_head.
        n_head (int): Number of attention heads.
    """
    def __init__(self, manifold: Lorentz, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.manifold = manifold
        self.n_head = n_head
        self.d_model = d_model
        dim_per_head = d_model // n_head

        self.Wq = nn.Linear(d_model, n_head * (dim_per_head - 1))
        self.Wk = nn.Linear(d_model, n_head * (dim_per_head - 1))
        self.Wv = nn.Linear(d_model, n_head * (dim_per_head - 1))
        self.out_proj = nn.Linear(n_head * (dim_per_head - 1), d_model - 1)
        self.scale = nn.Parameter(torch.tensor(float(d_model) ** 0.5))

    def _lift(self, x_space: torch.Tensor) -> torch.Tensor:
        x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + self.manifold.c) ** 0.5
        return torch.cat([x_time, x_space], dim=-1)

    def forward(
        self,
        query_input: torch.Tensor,
        source_input: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_input:  (B, N_q, D) points on the Lorentz manifold.
            source_input: (B, N_kv, D) points on the Lorentz manifold.
            mask:         optional boolean mask of shape (B, N_q, N_kv).

        Returns:
            output:  (B, N_q, D) attended output on the Lorentz manifold.
            weights: (B, H, N_q, N_kv) attention weights.
        """
        B, N_q, D = query_input.shape
        _, N_kv, _ = source_input.shape
        H, dh = self.n_head, D // self.n_head

        q = self._lift(self.Wq(query_input).view(B, N_q, H, dh - 1).transpose(1, 2))
        k = self._lift(self.Wk(source_input).view(B, N_kv, H, dh - 1).transpose(1, 2))
        v = self._lift(self.Wv(source_input).view(B, N_kv, H, dh - 1).transpose(1, 2))

        attn = (2 * self.manifold.c + 2 * self.manifold.cinner(q, k)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        weights = torch.softmax(attn, dim=-1)

        out = self.manifold.lorentzian_centroid(v, weights)
        out_space = self.out_proj(out[..., 1:].transpose(1, 2).reshape(B, N_q, H * (dh - 1)))
        return self._lift(out_space), weights


if __name__ == "__main__":
    manifold = Lorentz(c=1.0)
    model = LorentzCrossAttention(manifold, d_model=512, n_head=8)

    text = manifold.random_normal(2, 16, 512)
    image = manifold.random_normal(2, 196, 512)
    out, weights = model(text, image)

    assert out.shape == text.shape
    assert weights.shape == (2, 8, 16, 196)
    assert manifold._check_point_on_manifold(out)[0]
    print("ok")
    print(f"output: {out.shape} | weights: {weights.shape} | on manifold: {manifold._check_point_on_manifold(out)[0]} | output norm: {out[..., 1:].norm(dim=-1).mean():.4f}")