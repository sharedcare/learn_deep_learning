import numpy as np
import torch
import torch.nn as nn


def get_attn_pad_mask(seq):
    """
    Args:
        seq: [batch_size, seq_len]

    Returns:
        pad_attn_mask: [batch_size, 1, seq_len]

    """
    batch_size = seq.size(0)
    seq_len = seq.size(1)
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


class PositionEmbedding(nn.Module):
    """Compute the position embedding for word vectors (1d absolute Sinusoid embedding, not trainable)
    $PE_{pos, i} = \sin(\frac{pos}{10000^{2i / d_model}})$ if $i$ is even,
    $PE_{pos, i} = \cos(\frac{pos}{10000^{2i / d_model}})$ if $i$ is odd.

    """
    def __init__(self, d_model: int, max_len: int, device: str ="cpu") -> None:
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


class ScaleDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = \softmax(\frac{QK^T}{\sqrt{d_k}})V

    """
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        # Scale
        attention_scores /= torch.sqrt(self.d_k)
        # Mask
        attention_scores.masked_fill_(attn_mask, 1e-9)

        attn = nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    """ Multi-head Self Attention mechanism
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
    """
    def __init__(self, d_model: int, d_k: int, n_heads: int, device: str) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, n_heads * d_k)
        self.w_k = nn.Linear(d_model, n_heads * d_k)
        self.w_v = nn.Linear(d_model, n_heads * d_k)
        self.attention = ScaleDotProductAttention()
        self.out_layer = nn.Linear(n_heads * d_k, d_model)

    def forward(self, x, attn_mask):
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        context, attn = self.attention(query, key, value, attn_mask)
        
        # Add & Norm
        context += x
        output = nn.LayerNorm(self.out_layer(context))
        return output, attn


class FeedForwardNet(nn.Module):
    """Feed Forward Layer
    /max(0, x w_1 + b_1) w_2 + b_2

    """
    def __init__(self, d_model, hidden_dim, device):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.device = device

    def forward(self, x):
        residual = x
        output = self.ffn(x)
        return nn.LayerNorm(output + residual)


class Encoder(nn.Module):
    def __init__(self, d_model, vocab_size, d_k, n_heads, max_len, n_layers, d_ffn, device='cpu'):
        super(Encoder, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)

        class EncoderLayer(nn.Module):
            def __init__(self):
                super(EncoderLayer, self).__init__()
                self.attention = MultiHeadAttention(d_model, d_k, n_heads, device)
                self.ffn = FeedForwardNet(d_model, d_ffn, device)

            def forward(self, x, attn_mask):
                output, attn = self.attention(x, attn_mask)
                output = self.ffn(output)
                return output, attn

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer() for layer in range(n_layers)]
        )

    def forward(self, x):
        x = self.word_emb(x)
        x = self.pos_emb(x)
        attn_mask = get_attn_pad_mask(x)
        attns = []
        for layer in self.encoder_layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)

        return x, attns


if __name__ == "__main__":
    D_MODEL = 512   # dimension of word embedding
    D_FFN = 2048    # dimension of feed forward net
    D_K = 64        # dimension of K, Q, V
    N_LAYERS = 6    # num of encoder/decoder layers
    N_HEADS = 8     # num of multi-head attention
