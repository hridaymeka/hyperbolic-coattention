import sys
sys.path.insert(0, '/Users/hridaymeka/hyperbolic_vlm/HyperCore')
import torch
from hypercore.manifolds import Lorentz
from hypercore.nn.attention.lorentz_former_conv import LorentzMultiheadAttention

manifold = Lorentz(c=1.0)
attn = LorentzMultiheadAttention(manifold, 64, 64, 8, attention_type='full', trans_heads_concat=True)

text = manifold.random_normal(2, 16, 512)
image = manifold.random_normal(2, 196, 512)

out = attn(text, image)
print(out.shape)