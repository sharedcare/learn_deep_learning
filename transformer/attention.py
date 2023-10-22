import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbedding(nn.Module):
    """Compute the position embedding for word vectors (1d absolute Sinusoid embedding, not trainable)
    $PE_{pos, i} = \sin(\frac{pos}{10000^{2i / d_model}})$ if $i$ is even,
    $PE_{pos, i} = \cos(\frac{pos}{10000^{2i / d_model}})$ if $i$ is odd.

    """
    def __init__(self, d_model: int, max_len: int, device: str ="mps") -> None:
        super().__init__(d_model, max_len, device)
        pos_emb = np.array([
            [pos / np.power(10000, (2 * i) / d_model) for i in range(d_model)] 
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])    # (max_len, d_model)
        pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])  
        pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])
        self.pos_table = torch.FloatTensor(pos_emb, device=device).unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, word_vec: torch.Tensor):
        word_vec += self.pos_table[:, :word_vec.size(1)]
        return word_vec


class Attention(nn.Module):
    """ Multi-head Self Attention mechnism
    attention(Q, K, V) = \softmax(\frac{QK^T}{\sqrt{d_k}})V

    """
    def __init__(self, d_model: int, n_dim: int, n_heads: int, max_len: int) -> None:
        super().__init__(d_model, n_dim, n_heads, max_len)
        self.d_model = d_model
        self.n_dim = n_dim
        self.w_q = nn.Linear(d_model, n_heads * n_dim)
        self.w_k = nn.Linear(d_model, n_heads * n_dim)
        self.w_v = nn.Linear(d_model, n_heads * n_dim)
        self.out_layer = nn.Linear(n_heads * n_dim, d_model)

    def forward(self, embed_word_vec, attn_mask) -> torch.Tensor:
        query = self.w_q(embed_word_vec)
        key = self.w_k(embed_word_vec)
        value = self.w_v(embed_word_vec)

        attention_scores = torch.matmul(query, key.transpose(1, 0))

        # Scale
        attention_scores /= torch.sqrt(self.n_dim)

        # Mask (Opt.)
        attention_scores.masked_fill_(attn_mask)

        attention = nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attention, value)
        
        # Add & Norm
        context += embed_word_vec
        output = nn.LayerNorm(self.out_layer(context))
        return output, attention

