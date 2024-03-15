from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array
from transformers.activations import ACT2FN


def repeat_kv(x, repeats):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if repeats == 1:
        return x
    else:
        # (m, seq_len, n_kv_heads, 1, head_dim)
        # --> (m, seq_len, n_kv_heads, n_rep, head_dim)
        # --> (m, seq_len, n_kv_heads * n_rep, head_dim)
        x = mx.expand_dims(x, 3)
        x = mx.repeat(x, repeats, axis=3)
        x = x.reshape([batch_size, seq_len, n_kv_heads * repeats, head_dim])
        return x


@dataclass
class ModelArgs:
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    vocab_size: int = -1  # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    max_position_embeddings: int = 2048

    # RoPE params
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0

    device: mx.Device = None


class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim):
        self.cache_k = mx.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim))
        self.cache_v = mx.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim))

    def update(self, batch_size, start_pos, xk, xv):
        self.cache_k[:batch_size, start_pos :start_pos + xk.size(1)] = xk
        self.cache_v[:batch_size, start_pos :start_pos + xv.size(1)] = xv

    def get(self, batch_size, start_pos, seq_len):
        keys = self.cache_k[:batch_size,  :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        return keys, values


class RMSNorm(nn.Module):
    """Implementation of RMSNorm,
    working exactly as built-in nn.RMSNorm layer.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = mx.ones(dim)

    def _norm(self, x: array):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * mx.rsqrt(x.square().mean(axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: array):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.astype(mx.float32)).astype(x.dtype)


class ScaleDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = \softmax(\frac{QK^T}{\sqrt{d_k}})V

    """

    def __init__(self) -> None:
        super(ScaleDotProductAttention, self).__init__()

    def __call__(
        self, Q: array, K: array, V: array, attn_mask: Optional[array] = None
    ) -> Tuple[array, array]:
        attention_scores = mx.matmul(Q, K.transpose(-2, -1))
        # Scale
        attention_scores /= mx.sqrt(self.d_k)
        # Mask
        if attn_mask is not None:
            attention_scores += attn_mask

        attn = nn.Softmax(dim=-1)(attention_scores)
        context = mx.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    """Multi-head Self Attention mechanism
    self.fc1 = nn.Linear()
    self.fc2 = nn.Linear()
    """

    def __init__(self, args: ModelArgs) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = args.hidden_size
        self.d_k = args.head_dim
        self.local_n_heads = args.num_attention_heads
        self.w_q = nn.Linear(args.hidden_size, args.num_attention_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(args.hidden_size, args.num_attention_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(args.hidden_size, args.num_attention_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(args.num_attention_heads * self.d_k, args.hidden_size, bias=False)
        self.attention = ScaleDotProductAttention()
        self.cache = KVCache(
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            n_kv_heads=args.num_key_value_heads,
            head_dim=self.d_k,
        )
        self.rope = nn.RoPE(
            args.head_dim, base=args.rope_theta
        )

    def __call__(
        self, x: array, attn_mask: Optional[array] = None
    ) -> Tuple[array, array]:
        queries = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        queries = self.rope(queries)
        keys = self.rope(keys)

        keys = repeat_kv(keys, self.repeats)
        values = repeat_kv(values, self.repeats)

        context, attn = self.attention(queries, keys, values, attn_mask)
        output = self.w_o(context)
        return output, attn


class Feedforward(nn.Module):
    """ SwiGLU """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=False,
        )
        self.w3 = nn.Linear(
            args.intermediate_size,
            args.hidden_size,
            bias=False,
        )

        self.act = ACT2FN[args.hidden_act]

    def __call__(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.ffn = Feedforward(args)

        self.attn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, attn_mask):
        # Add & Norm
        x = self.attn_norm(x)
        hidden_states = x + self.attention(x, attn_mask)
        hidden_states = self.ffn_norm(hidden_states)
        out = hidden_states + self.ffn(hidden_states)
        return out

