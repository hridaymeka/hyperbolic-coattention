"""
Gradient sanity check for LorentzCrossAttention.

Goal: prove the module can actually learn -- nonzero gradients flow to its
weights, and an optimizer step actually changes them. This does NOT prove
the module learns anything *useful* -- that's the real ablation later. This
just rules out the failure mode from the Alma track (a "learned" component
that silently never trains).

Run locally on CPU. No cluster needed for this.

Matched to the actual LorentzCrossAttention signature: forward(query_input,
source_input, mask=None) -> (output, weights). Run from
~/hyperbolic_vlm/experiments/ so the local import resolves.
"""

import torch
import torch.nn.functional as F

from hyperbolic_cross_attention import LorentzCrossAttention
from hypercore.manifolds import Lorentz

torch.manual_seed(0)

# ---- Config ----
BATCH = 6                 # need >2 "scenes" for a non-degenerate contrastive signal
NUM_TEXT_TOKENS = 4       # small on purpose -- this is a sanity check, not a real batch
NUM_IMG_PATCHES = 16
DIM = 64                  # small dim -> fast CPU run
NUM_HEADS = 4
CURVATURE = 1.0
LR = 1e-2
STEPS = 150

# ---- Manifold + module ----
manifold = Lorentz(c=CURVATURE)
attn = LorentzCrossAttention(manifold, d_model=DIM, n_head=NUM_HEADS)

optimizer = torch.optim.Adam(attn.parameters(), lr=LR)

# ---- Synthetic on-manifold inputs ----
# Use the manifold's own random_normal (same as the __main__ block in
# hyperbolic_cross_attention.py) so inputs are guaranteed valid points on
# the hyperboloid, not just random noise.
text_tokens = manifold.random_normal(BATCH, NUM_TEXT_TOKENS, DIM)
img_patches = manifold.random_normal(BATCH, NUM_IMG_PATCHES, DIM)

# Fake "positive pair" target: scene i's text should end up close to scene i's
# image after fusion, and far from every other scene's image (real InfoNCE
# structure, not self-distance -- avoids the degenerate "compare a tensor to
# itself" bug). We use img_patches' own mean-pooled space component as a
# stand-in "image embedding" -- crude, but it's a genuinely different tensor
# from the cross-attention output, so the loss has real signal to chase.
image_embed = img_patches[..., 1:].mean(dim=1)  # [BATCH, DIM-1], space component only
labels = torch.arange(BATCH)  # scene i pairs with image i

# ---- Snapshot weights before any training ----
def flatten_params(module):
    return torch.cat([p.detach().flatten().clone() for p in module.parameters()])

weights_before = flatten_params(attn)

# ---- Step 1: single forward/backward, check gradients ----
optimizer.zero_grad()
out, _weights = attn(query_input=text_tokens, source_input=img_patches)

# Pool text-token outputs to a single vector per batch element (mean over
# tokens) so we have something to run a similarity/contrastive loss on.
# Compare against image_embed (a DIFFERENT tensor, not itself) -- real
# cross-tensor contrastive structure.
pooled = out[..., 1:].mean(dim=1)  # [BATCH, DIM-1], space component only, to match image_embed
sims = -torch.cdist(pooled, image_embed, p=2)  # [BATCH, BATCH]
loss = F.cross_entropy(sims, labels)

loss.backward()

print("=== Step 1 gradient check ===")
any_nonzero = False
for name, p in attn.named_parameters():
    if p.grad is None:
        print(f"  [FAIL] {name}: grad is None")
        continue
    grad_norm = p.grad.norm().item()
    nonzero = grad_norm > 0
    any_nonzero = any_nonzero or nonzero
    status = "OK" if nonzero else "ZERO"
    print(f"  [{status}] {name}: grad_norm={grad_norm:.6f}")

if not any_nonzero:
    raise RuntimeError("All gradients are zero or None -- module is NOT learning. Stop here and debug.")

optimizer.step()
weights_after_step1 = flatten_params(attn)
changed = not torch.allclose(weights_before, weights_after_step1)
print(f"\nWeights changed after step 1: {changed}")
if not changed:
    raise RuntimeError("Weights did not change after optimizer.step() -- check optimizer/param registration.")

# ---- Extended check: does loss go down over more steps on this tiny batch? ----
print(f"\n=== Overfitting {STEPS} steps on the tiny synthetic batch ===")
losses = []
for step in range(STEPS):
    optimizer.zero_grad()
    out, _weights = attn(query_input=text_tokens, source_input=img_patches)
    pooled = out[..., 1:].mean(dim=1)
    sims = -torch.cdist(pooled, image_embed, p=2)
    loss = F.cross_entropy(sims, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 25 == 0 or step == STEPS - 1:
        print(f"  step {step:4d}  loss={loss.item():.4f}")

print(f"\nLoss went from {losses[0]:.4f} to {losses[-1]:.4f}")
print("Downward trend expected if the module can fit even this toy signal.")

# ---- Final on-manifold check (sanity that training didn't break the geometry) ----
# Using the same check as the file's own __main__ block. _check_point_on_manifold
# returns a tuple whose [0] is a bool; not assuming what else it returns.
with torch.no_grad():
    final_out, _weights = attn(query_input=text_tokens, source_input=img_patches)
    check_result = manifold._check_point_on_manifold(final_out)
    print(f"\nOn-manifold after training: {check_result[0]}")

    # Manual constraint check too, as a cross-check independent of the
    # library's own function: <x,x>_Lorentz should equal -1/c.
    time_comp = final_out[..., 0]
    space_comp = final_out[..., 1:]
    constraint_violation = (time_comp ** 2 - (space_comp ** 2).sum(dim=-1) - 1.0 / CURVATURE).abs().max().item()
    print(f"Manual constraint check, max violation: {constraint_violation:.2e}  (expect ~1e-5 or smaller)")
