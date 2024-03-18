from dataclasses import dataclass
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array


ACT2FN = {
    "gelu": nn.GELU(),
    "mish": nn.Mish(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
}

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

def sample(logits: array, temperature: int = 1.0) -> array:
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


@dataclass
class ModelArgs:
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    vocab_size: int = -1  # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # Needed for KV cache
    num_key_value_heads: Optional[int] = None
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
        self.cache_k[:batch_size, start_pos :start_pos + xk.shape[1]] = xk
        self.cache_v[:batch_size, start_pos :start_pos + xv.shape[1]] = xv

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

    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def __call__(
        self, Q: array, K: array, V: array, attn_mask: Optional[array] = None
    ) -> Tuple[array, array]:
        batch_size, _, seq_len, _ = Q.shape
        attention_scores = mx.matmul(Q, K.transpose(0, 1, 3, 2))
        # Scale
        attention_scores /= mx.sqrt(self.head_dim)
        # Mask
        if attn_mask is not None:
            attention_scores += attn_mask

        attn = mx.softmax(attention_scores.astype(mx.float32), axis=-1).astype(Q)
        context = mx.matmul(attn, V)
        output = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head Self Attention mechanism
    self.fc1 = nn.Linear()
    self.fc2 = nn.Linear()

    working exactly as built-in nn.MultiHeadAttention layer.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        self.head_dim = self.hidden_dim // self.n_heads
        self.w_q = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.w_k = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.w_v = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.w_o = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.attention = ScaleDotProductAttention(head_dim=self.head_dim)
        self.cache = KVCache(
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            n_kv_heads=args.num_key_value_heads,
            head_dim=self.head_dim,
        )
        self.rope = nn.RoPE(
            args.head_dim, base=args.rope_theta
        )

    def __call__(
        self, x: array, attn_mask: Optional[array] = None, start_pos: Optional[int] = None,
    ) -> array:
        batch_size, seq_len, _ = x.shape
        # x, queries: (bs, seq_len, hidden_size)
        queries = self.w_q(x)
        # keys, values: (bs, seq_len, n_kv_heads * head_dim)
        keys = self.w_k(x)
        values = self.w_v(x)

        # queries: (bs, seq_len, hidden_size) -> (bs, seq_len, n_heads, head_dim)
        queries = queries.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        # keys, values: (bs, seq_len, n_kv_heads * head_dim) -> (bs, seq_len, n_kv_heads, head_dim)
        keys = keys.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if start_pos is None:
            # train and fine-tune
            # queries: (bs, seq_len, hidden_size) -> (bs, n_heads, head_dim)
            queries = self.rope(queries)
            # keys: (bs, seq_len, n_kv_heads * head_dim) -> (bs, n_kv_heads, head_dim)
            keys = self.rope(keys)
        else:
            # inference
            # queries: (bs, seq_len, hidden_size) -> (bs, n_heads, head_dim)
            queries = self.rope(queries)
            # keys: (bs, seq_len, n_kv_heads * head_dim) -> (bs, n_kv_heads, head_dim)
            keys = self.rope(keys)
            # replace the entry in the cache
            self.cache.update(batch_size, start_pos, keys, values)
            keys, values = self.cache.get(batch_size, start_pos, seq_len)
            # keys, values: (bs, seq_len, n_kv_heads, head_dim) --> (bs, seq_len, n_kv_heads, head_dim)
            keys = repeat_kv(keys, self.repeats)
            values = repeat_kv(values, self.repeats)

        # queries: (bs, n_heads, *seq_len*, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        # keys, values: (bs, n_heads, seq_len, head_dim)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        output, attn = self.attention(queries, keys, values, attn_mask)
        output = self.w_o(output)
        return output


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

    def __call__(self, x: array) -> array:
        # x(bs, seq_len, hidden_size) -> mm1(bs, seq_len, intermediate_size)
        mm1 = self.w1(x)
        # x(bs, seq_len, hidden_size) -> mm3(bs, seq_len, intermediate_size)
        mm3 = self.w3(x)
        # mm2(bs, seq_len, hidden_size)
        mm2 = self.w2(self.act(mm1) * mm3)
        return mm2


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.ffn = Feedforward(args)

        self.attn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self,
                 x: array,
                 attn_mask: Optional[array] = None,
                 start_pos: Optional[int] = None) -> array:
        # Norm & Add
        # x is input token embedding with shape (bs, seq_len, hidden_size)
        x = self.attn_norm(x)
        # hidden_states: (bs, seq_len, hidden_size)
        hidden_states = x + self.attention(x, attn_mask, start_pos)
        hidden_states = self.ffn_norm(hidden_states)
        # out: (bs, seq_len, hidden_size)
        out = hidden_states + self.ffn(hidden_states)
        return out


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, x: array) -> array:
        # training
        # x is input token with shape (bs, seq_len)
        batch_size, seq_len = x.shape
        # attn_mask: (seq_len, seq_len)
        attn_mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        attn_mask = attn_mask.astype(self.tok_embeddings.weight.dtype)
        # hidden_states: (bs, seq_len, hidden_size)
        hidden_states = self.tok_embeddings(x)
        # Apply all decoder layers
        for layer in self.layers:
            # hidden_states: (bs, seq_len, hidden_size)
            hidden_states = layer(hidden_states, attn_mask)
        hidden_states = self.norm(hidden_states)
        # out: (bs, seq_len, vocab_size)
        out = self.output(hidden_states).astype(mx.float16)
        return out

    def generate(self,
                 x: array,
                 max_gen_len: int,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 start_pos: int = 0):
        # inference
        # x is input prompt with shape (bs, seq_len)
        batch_size, seq_len = x.shape
        # hidden_states: (bs, seq_len, hidden_size)
        hidden_states = self.tok_embeddings(x)

        attn_mask = None
        if seq_len > 1:
            attn_mask = mx.full(
                (1, 1, seq_len, seq_len), float("-inf")
            )
            attn_mask = mx.triu(attn_mask, k=start_pos + 1).astype(self.tok_embeddings.weight.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask, start_pos)
        hidden_states = self.norm(hidden_states)
        # out: (bs, 1, vocab_size)
        out = self.output(hidden_states[:, -1, :]).astype(mx.float32)
        out = sample(out)
        yield out
