import torch
import torch.nn as nn

# Multi-head causal attention mechanism using parallelism from matrix multiplication.
class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

    self.d_out = d_out
    self.num_heads = num_heads

    # Reduce the projection dimension to match the output dimension 
    self.head_dim = d_out // num_heads

    # Initialize the random trainable weight matrices
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    # Create projection layer to combine outputs
    self.out_proj = nn.Linear(d_out, d_out)

    # Create dropout layer to reduce overfitting - the model becomes less reliant on
    # any specific hidden set of layer units. Used only in training and then disabled.
    # Dropout is longer used by OpenAI GPT.
    self.dropout = nn.Dropout(dropout)

    # Create a causal mask using the upper triangular matrix and register it so it gets.
    # automatically moved to the appropriate device (CPU or GPU) along with the model.
    # This avoids having to redefine the mask layer during iterations of the forward() method.
    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

  # Forward pass computation
  def forward(self, x):
    # Extract dimensions from input
    batch_size, num_tokens, d_in = x.shape

    # Compute weight vectors
    # Shape: batch_size, num_tokens, d_out
    queries = self.W_query(x)
    keys = self.W_key(x)
    values = self.W_value(x)

    # Reshape the weight vectors so they can be applied to multiple heads by matrix multiplication.
    # The view() call splits the matrix by adding a num_heads dimension, then unroll the last dimension
    # to result in shape: batch_size, num_tokens, num_heads, head_dim. 
    # View() returns a contigous memory object, and is more efficient than reshape(), which makes a copy.
    queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
    keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
    values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

    # Transpose the middle two dimensionss from (num_tokens, num_heads) to (num_heads, num_tokens)
    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    # Compute attention scores using dot product of queries times keys
    # Transpose dimensions 2 and 3
    attn_scores = queries @ keys.transpose(2, 3) # omega

    # Apply causal mask to remove future tokens.
    # It is performed before normalization so that row sums still sum to 1.
    # The negative infinity values are treated by softmax as zero, since e**(-infinity) = 0.
    # Note everything in PyTorch that ends with "_" runs as an inplace operation (ie. not copied).
    attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

    # Compute attention weights by normalizing with softmax, then apply dropout
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # Compute context vectors of all inputs
    # Shape: batch_size, num_tokens, num_heads, head_dim
    context_vec = (attn_weights @ values).transpose(1, 2)

    # Combine heads to output dimension (num_heads, head_dim).
    # The contiguous() call is needed since transpose() may have fragmented the tensor in memory.
    context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)
    return context_vec
  

# Normalization layer
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()

    # Small threshold constant
    self.eps = 1e-5

    # Scale and shift trainable layer
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  # Forward pass computation
  def forward(self, x):
    # Compute mean and variance of input tensor
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # For normalization, add a small denominator constant to prevent division by zero
    norm_x = (x - mean) / torch.sqrt(var + self.eps)

    # Scale and shift
    return self.scale * norm_x + self.shift


# GELU network layer
class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  # Compute GELU function
  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi))
      * (x + 0.044715 * torch.pow(x, 3))))


# Feed forward network
class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.layers = nn.Sequential(
      # Expand the network (e.g. 768 -> 3072)
      nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),

      # Learn from the network using GELU
      GELU(),

      # Compress the network (e.g. 3072 -> 768)
      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
    )
  
  def forward(self, x):
    return self.layers(x)

