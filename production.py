import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tiktoken


# Class to load dataset end GPT-2 encode
class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    # Tokenize the entire text
    token_ids = tokenizer.encode(txt, allowed_special={ "<|endoftext|>" })
    assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

    # Use a sliding window to chunk the book into overlapping sequences of max_length
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i + max_length]
      target_chunk = token_ids[i + 1: i + max_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]


# Function to create encoding using PyTorch dataloader.
# Dataloader provides utilities for input data batching and shuffling.
def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

  # Initialize the tokenizer
  tokenizer = tiktoken.get_encoding("gpt2")

  # Create dataset
  dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

  # Create iterable dataloader to load batches of data:
  # - batch_size loads batch of data of specified size during each iteration
  # - shuffle=True changes the order of data indices to achieve better generalization
  # - drop_last=True drops the last batch if it is shorter than the batch_size to prevent loss spikes during training
  # - num_workers does not work in notebooks
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
  return dataloader


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


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    # Declare network layers
    self.att = MultiHeadAttention(
      d_in=cfg["emb_dim"],
      d_out=cfg["emb_dim"],
      context_length=cfg["context_length"],
      num_heads=cfg["n_heads"], 
      dropout=cfg["drop_rate"],
      qkv_bias=cfg["qkv_bias"])
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    # Shortcut connection for attention block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
    x = self.drop_shortcut(x)
    x = x + shortcut  # Add the original input back

    # Shortcut connection for feed forward block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut  # Add the original input back

    return x


class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    # Declare input and output embedding layers
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

    # Declare dropout layer
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    # Declare transformer block layers
    self.xform_blocks = nn.Sequential(
      *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

    # Declare final normalization layer
    self.final_norm = LayerNorm(cfg["emb_dim"])

    # Declare linear layer
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape

    # Map vocabulary space to embedding space
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

    # Combine token and positional embeddings
    # Shape [batch_size, num_tokens, emb_size]
    x = tok_embeds + pos_embeds

    # Process dropout
    x = self.drop_emb(x)

    # Process transformer blocks
    x = self.xform_blocks(x)

    # Process final normalization
    x = self.final_norm(x)

    # Map embedding space to vocabulary space
    logits = self.out_head(x)
    return logits


# Generate text one token at a time
def generate_text_simple(model, idx, max_new_tokens, context_size):
  # Truncate current context if it exceeds the supported context size
  for _ in range(max_new_tokens):
    # Get token context
    idx_cond = idx[:, -context_size:]

    # Get the predictions from the tokens.
    # Disable gradient computation to suppress the model training.
    with torch.no_grad():
      logits = model(idx_cond)

    # Focus only on the last time step
    logits = logits[:, -1, :]

    # Apply softmax to get probabilities
    probas = torch.softmax(logits, dim=-1)

    # Get the idx of the vocab entry with the highest probability value
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)

    # Append sampled index to the running sequence
    idx = torch.cat((idx, idx_next), dim=1)

  return idx
