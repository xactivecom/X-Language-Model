import torch
import torch.nn as nn

class SwiGLU(nn.Module):
  def __init__(self, d_model, d_ff_glu=None):
    super().__init__()
    # Use 8/3 ratio for hidden layer for performance
    d_ff_glu = d_ff_glu or int((8/3) * d_model)
    self.w_in = nn.Linear(d_model, 2*d_ff_glu)
    self.w_out = nn.Linear(d_ff_glu, d_model)

  def forward(self, x):
    # Chunk the result into two halfs
    a, b = self.w_in(x).chunk(2, dim=-1)

    # sigmoid linear unit
    gated = nn.functional.silu(a) * b
    return self.w_out(gated)
