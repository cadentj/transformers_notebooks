import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PreTrainedModel, PretrainedConfig

class SingleHeadAttention(nn.Module):
    def __init__(self, d):
        """
        Here we will assume that the input dimensions are same as the
        output dims.
        """
        super().__init__()

        self.q_layer = torch.nn.Linear(d, d)
        self.k_layer = torch.nn.Linear(d, d)
        self.v_layer = torch.nn.Linear(d, d)

    def forward(self, x, mask=None, return_weights=False):
        """
        Assume x is <t x d> -- t being the sequence length, d
        the embed dims.

        W_q, W_k, and W_v are weights for projecting into queries,
        keys, and values, respectively. Here these will have shape
        <d x t>, yielding d dimensional vectors for each input.

        This function should return a t dimensional attention vector
        for each input -- i.e., an attention matrix with shape <t x t>,
        and the values derived from this <t x d>.

        Derive Q, K, V matrices, then self attention weights. These should
        be used to compute the final representations (t x d); optionally
        return the weights matrix if `return_weights=True`.
        """
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)

        A = Q @ K.transpose(-2, -1)
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        weights = F.softmax(A, dim=-1)

        if return_weights:
          return weights, weights @ V

        return weights @ V

class CogModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super(CogModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([SelfAttentionLayer(embed_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, src):
        mask = torch.triu(torch.ones(
            src.size(1), src.size(1), dtype=torch.bool, device=src.device)).T
        src = self.embed(src)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, mask)
        src = self.norm(src)
        output = self.decoder(src)
        return output

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = SingleHeadAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

    def forward(self, src, mask=None):
        src = src + self.norm1(self.self_attn(src, mask=mask))
        return src

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Create a custom configuration
class CogConfig(PretrainedConfig):
    model_type = "custom_transformer"

    def __init__(self, vocab_size=30000, embed_dim=512, num_layers=6, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

# Create a model that can be loaded with from_pretrained
class CogModelForHF(PreTrainedModel):
    config_class = CogConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = CogModel(config.vocab_size, config.embed_dim, config.num_layers)

    def forward(self, src):
        return self.model(src)