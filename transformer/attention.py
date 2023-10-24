import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List
from torch import Tensor


def get_padding_mask(seq: np.ndarray) -> Tensor:
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


def get_attn_mask(seq: np.ndarray) -> Tensor:
    """

    Args:
        seq: [batch_size, seq_len]

    Returns:

    """
    batch_size = seq.size(0)
    seq_len = seq.size(1)
    subseq_mask = torch.triu(torch.ones([batch_size, seq_len, seq_len]), diagonal=1)
    return subseq_mask


class PositionEmbedding(nn.Module):
    """Compute the position embedding for word vectors (1d absolute Sinusoid embedding, not trainable)
    $PE_{pos, i} = \sin(\frac{pos}{10000^{2i / d_model}})$ if $i$ is even,
    $PE_{pos, i} = \cos(\frac{pos}{10000^{2i / d_model}})$ if $i$ is odd.

    """

    def __init__(self, d_model: int, max_len: int, device: str = "cpu") -> None:
        super(PositionEmbedding, self).__init__()
        pos_emb = np.array([
            [pos / np.power(10000, (2 * i) / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])  # (max_len, d_model)
        pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])
        pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])
        self.pos_table = torch.FloatTensor(pos_emb, device=device).unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, word_vec: Tensor) -> Tensor:
        word_vec += self.pos_table[:, :word_vec.size(1)]
        return word_vec


class ScaleDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = \softmax(\frac{QK^T}{\sqrt{d_k}})V

    """

    def __init__(self) -> None:
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
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
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, n_heads * d_k)
        self.w_k = nn.Linear(d_model, n_heads * d_k)
        self.w_v = nn.Linear(d_model, n_heads * d_k)
        self.attention = ScaleDotProductAttention()
        self.out_layer = nn.Linear(n_heads * d_k, d_model)
        self.to(device)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        context, attn = self.attention(query, key, value, attn_mask)

        # Add & Norm
        context += q
        output = nn.LayerNorm(self.d_mode)(self.out_layer(context))
        return output, attn


class FeedForwardNet(nn.Module):
    """Feed Forward Layer
    /max(0, x w_1 + b_1) w_2 + b_2

    """

    def __init__(self, d_model: int, hidden_dim: int, device: Optional[str] = "cpu") -> None:
        super(FeedForwardNet, self).__init__()
        self.d_model = d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        output = self.ffn(x)
        return nn.LayerNorm(self.d_model)(output + residual)


class Encoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, d_k: int, n_heads: int, max_len: int, n_layers: int, d_ffn: int, device: Optional[str] = 'cpu') -> None:
        super(Encoder, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)

        class EncoderLayer(nn.Module):
            def __init__(self) -> None:
                super(EncoderLayer, self).__init__()
                self.attention = MultiHeadAttention(d_model, d_k, n_heads, device)
                self.ffn = FeedForwardNet(d_model, d_ffn, device)

            def forward(self, x: Tensor, padding_mask: Tensor) -> Tuple[Tensor, Tensor]:
                output, attn = self.attention(x, x, x, padding_mask)
                output = self.ffn(output)
                return output, attn

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer() for layer in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x = self.word_emb(x)
        x = self.pos_emb(x)
        padding_mask = get_padding_mask(x)
        attns = []
        for layer in self.encoder_layers:
            x, attn = layer(x, padding_mask)
            attns.append(attn)

        return x, attns


class Decoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, d_k: int, n_heads: int, max_len: int, n_layers: int, d_ffn: int, device: Optional[str] = 'cpu') -> None:
        super(Decoder, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)

        class DecoderLayer(nn.Module):
            def __init__(self) -> None:
                super(DecoderLayer, self).__init__()
                self.masked_attention = MultiHeadAttention(d_model, d_k, n_heads, device)
                self.attention = MultiHeadAttention(d_model, d_k, n_heads, device)
                self.ffn = FeedForwardNet(d_model, d_ffn, device)

            def forward(self, x: Tensor, enc_out: Tensor, padding_mask: Tensor, attn_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                output, masked_attn = self.attention(x, x, x, attn_mask)
                output, attn = self.attention(output, enc_out, enc_out, padding_mask)
                output = self.ffn(output)
                return output, masked_attn, attn

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer() for layer in range(n_layers)]
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor, enc_outs: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        x = self.word_emb(x)
        x = self.pos_emb(x)
        padding_mask = get_padding_mask(x)
        attn_mask = get_attn_mask(x)
        self_attns = []
        enc_attns = []
        for layer in self.decoder_layers:
            x, masked_attn, attn = layer(x, enc_outs, padding_mask, attn_mask)
            self_attns.append(masked_attn)
            enc_attns.append(attn)

        output = self.output_layer(x)
        output = nn.Softmax(dim=-1)(output)
        return output, self_attns, enc_attns


class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, d_k: int, n_heads: int, max_len: int, n_layers: int, d_ffn: int, device: Optional[str] = 'cpu') -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, vocab_size, d_k, n_heads, max_len, n_layers, d_ffn, device)
        self.decoder = Decoder(d_model, vocab_size, d_k, n_heads, max_len, n_layers, d_ffn, device)

    def forward(self, enc_inputs: Tensor, dec_inputs: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]:
        enc_outs, enc_attns = self.encoder(enc_inputs)
        dec_preds, dec_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outs)
        return dec_preds, enc_attns, dec_attns, dec_enc_attns


if __name__ == "__main__":
    D_MODEL = 512  # dimension of word embedding
    D_FFN = 2048  # dimension of feed forward net
    D_K = 64  # dimension of K, Q, V
    N_LAYERS = 6  # num of encoder/decoder layers
    N_HEADS = 8  # num of multi-head attention

    transformer = Transformer(D_MODEL, 1024, D_K, N_HEADS, 512, N_LAYERS, D_FFN)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=1e-3)
