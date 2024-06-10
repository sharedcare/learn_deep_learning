from dataclasses import dataclass
from glob import glob
import json
from typing import Any, Optional, Tuple
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array
from mlx.utils import tree_unflatten
import sentencepiece as spm


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
    batch_size, n_kv_heads, seq_len, head_dim = x.shape
    if repeats == 1:
        return x
    else:
        # (m, n_kv_heads, seq_len, 1, head_dim)
        # --> (m, n_kv_heads, n_rep, seq_len, head_dim)
        # --> (m, n_kv_heads * n_rep, seq_len, head_dim)
        x = mx.expand_dims(x, 2)
        x = mx.repeat(x, repeats, axis=2)
        x = x.reshape([batch_size, n_kv_heads * repeats, seq_len, head_dim])
        return x

def sample(logits: array, temperature: int = 1.0) -> array:
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


@dataclass
class ModelArgs:
    dim: int = 4096
    hidden_dim: int = 11008
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 64
    vocab_size: int = -1  # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-05
    hidden_act: str = "silu"

    # Needed for KV cache
    n_kv_heads: Optional[int] = None
    max_batch_size: int = 32
    max_seq_len: int = 2048

    max_position_embeddings: int = 2048

    # RoPE params
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    rope_traditional: bool = True

    device: mx.Device = None


class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, dtype=mx.float16):
        self.cache_k = mx.zeros([max_batch_size, n_kv_heads, max_seq_len, head_dim], dtype=dtype)
        self.cache_v = mx.zeros([max_batch_size, n_kv_heads, max_seq_len, head_dim], dtype=dtype)

    def update(self, batch_size, start_pos, xk, xv):
        self.cache_k[:batch_size, :, start_pos :start_pos + xk.shape[2]] = xk
        self.cache_v[:batch_size, :, start_pos :start_pos + xv.shape[2]] = xv

    def get(self, batch_size, start_pos, seq_len):
        keys = self.cache_k[:batch_size, :, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :, :start_pos + seq_len]
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
        attention_scores /= self.head_dim ** 0.5
        # Mask
        if attn_mask is not None:
            attention_scores += attn_mask

        attn = mx.softmax(attention_scores.astype(mx.float32), axis=-1).astype(attention_scores.dtype)
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
        self.hidden_dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        # self.scale = args.head_dim**-0.5
        self.head_dim = self.hidden_dim // self.n_heads
        self.wq = nn.Linear(
            args.dim, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim, args.dim, bias=False
        )
        self.sdpa = ScaleDotProductAttention(head_dim=self.head_dim)
        self.cache = KVCache(
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            # dtype=self.wk.weight.dtype,
        )
        self.rope = nn.RoPE(
            self.head_dim, base=args.rope_theta
        )

    def __call__(
        self, x: array, attn_mask: Optional[array] = None, start_pos: Optional[int] = None,
    ) -> array:
        batch_size, seq_len, _ = x.shape
        # x, queries: (bs, seq_len, dim)
        queries = self.wq(x)
        # keys, values: (bs, seq_len, n_kv_heads * head_dim)
        keys = self.wk(x)
        values = self.wv(x)

        # queries: (bs, seq_len, dim) -> (bs, n_heads, seq_len, head_dim)
        queries = queries.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        # keys, values: (bs, seq_len, n_kv_heads * head_dim) -> (bs, n_kv_heads, seq_len, head_dim)
        keys = keys.reshape(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if start_pos is None:
            # train and fine-tune
            # queries: (bs, seq_len, dim) -> (bs, n_heads, head_dim)
            queries = self.rope(queries)
            # keys: (bs, seq_len, n_kv_heads * head_dim) -> (bs, n_kv_heads, head_dim)
            keys = self.rope(keys)
        else:
            # inference
            # queries: (bs, n_heads, seq_len, head_dim)
            queries = self.rope(queries, offset=start_pos)
            # keys: (bs, n_kv_heads, seq_len, head_dim)
            keys = self.rope(keys, offset=start_pos)
            # replace the entry in the cache
            self.cache.update(batch_size, start_pos, keys, values)
            keys, values = self.cache.get(batch_size, start_pos, seq_len)
            # keys, values: (bs, n_kv_heads, seq_len, head_dim) --> (bs, n_heads, seq_len, head_dim)
            keys = repeat_kv(keys, self.repeats)
            values = repeat_kv(values, self.repeats)

        # queries: (bs, n_heads, *seq_len*, head_dim)
        # queries = queries.transpose(0, 2, 1, 3)
        # keys, values: (bs, n_heads, seq_len, head_dim)
        # keys = keys.transpose(0, 2, 1, 3)
        # values = values.transpose(0, 2, 1, 3)

        output, attn = self.sdpa(queries, keys, values, attn_mask)
        output = self.wo(output)
        return output


class Feedforward(nn.Module):
    """ SwiGLU """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(
            args.hidden_dim,
            args.dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False,
        )
        self.act = ACT2FN[args.hidden_act]

    def __call__(self, x: array) -> array:
        # x(bs, seq_len, dim) -> mm1(bs, seq_len, hidden_dim)
        mm1 = self.w1(x)
        # x(bs, seq_len, dim) -> mm3(bs, seq_len, hidden_dim)
        mm3 = self.w3(x)
        # mm2(bs, seq_len, dim)
        act1 = self.act(mm1)
        mm2 = self.w2(act1 * mm3)
        return mm2


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.feed_forward = Feedforward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self,
                 x: array,
                 attn_mask: Optional[array] = None,
                 start_pos: Optional[int] = None) -> array:
        # Norm & Add
        # x is input token embedding with shape (bs, seq_len, dim)
        residual = x
        x = self.attention_norm(x)
        # hidden_states: (bs, seq_len, dim)
        hidden_states = residual + self.attention(x, attn_mask, start_pos)
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        # out: (bs, seq_len, dim)
        out = residual + self.feed_forward(hidden_states)
        return out


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [
            DecoderBlock(args=args) for _ in range(args.n_layers)
        ]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, x: array) -> array:
        # training
        # x is input token with shape (bs, seq_len)
        batch_size, seq_len = x.shape
        # attn_mask: (seq_len, seq_len)
        attn_mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        attn_mask = attn_mask.astype(self.tok_embeddings.weight.dtype)
        # hidden_states: (bs, seq_len, dim)
        hidden_states = self.tok_embeddings(x)
        # Apply all decoder layers
        for layer in self.layers:
            # hidden_states: (bs, seq_len, dim)
            hidden_states = layer(hidden_states, attn_mask)
        hidden_states = self.norm(hidden_states)
        # out: (bs, seq_len, vocab_size)
        out = self.output(hidden_states).astype(mx.float16)
        return out

    def generate(
        self,
        x: array,
        temperature: float = 0.6,
        top_p: float = 0.9,
        start_pos: int = 0,
    ):
        # inference
        # x is input prompt with shape (bs, seq_len)
        batch_size, seq_len = x.shape
        # hidden_states: (bs, seq_len, dim)
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
        out = sample(out, temperature)
        yield out

        for cur_pos in range(8, self.max_seq_len):
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            hidden_states = out[:, None]

            hidden_states = self.tok_embeddings(hidden_states)
            for layer in self.layers:
                hidden_states = layer(hidden_states, None, cur_pos)
            hidden_states = self.norm(hidden_states)
            out = sample(
                self.output(hidden_states[:, -1, :]).astype(mx.float32), temperature
            )

            yield out

    def sanitize_config(self, config, weights):
        config.pop("model_type", None)
        n_heads = config["n_heads"]
        if "n_kv_heads" not in config:
            config["n_kv_heads"] = n_heads
        if "head_dim" not in config:
            config["head_dim"] = config["dim"] // n_heads
        if "hidden_dim" not in config:
            config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = weights["output.weight"].shape[-1]
        if "rope_theta" not in config:
            config["rope_theta"] = 10000
        unused = ["multiple_of", "ffn_dim_multiplier"]
        for k in unused:
            config.pop(k, None)
        return config

    def load_model(self, path):
        model_path = Path(path)
        unsharded_weights_path = Path(model_path / "weights.npz")
        if unsharded_weights_path.is_file():
            print("[INFO] Loading model from {}.".format(unsharded_weights_path))
            weights = mx.load(str(unsharded_weights_path))
        else:
            sharded_weights_glob = str(model_path / "weights.*.npz")
            weight_files = glob(sharded_weights_glob)
            print("[INFO] Loading model from {}.".format(sharded_weights_glob))

            if len(weight_files) == 0:
                raise FileNotFoundError("No weights found in {}".format(model_path))

            weights = {}
            for wf in weight_files:
                weights.update(mx.load(wf).items())

        with open(model_path / "config.json", "r") as f:
            config = self.sanitize_config(json.loads(f.read()), weights)
            quantization = config.pop("quantization", None)
        model = Llama(ModelArgs(**config))
        if quantization is not None:
            nn.QuantizedLinear.quantize_module(model, **quantization)
        model.update(tree_unflatten(list(weights.items())))
        tokenizer = spm.SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
        return model, tokenizer


def sanitize_config(config, weights):
    config.pop("model_type", None)
    n_heads = config["n_heads"]
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "hidden_dim" not in config:
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    if config.get("vocab_size", -1) < 0:
        config["vocab_size"] = weights["output.weight"].shape[-1]
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config


if __name__ == "__main__":
    mx.random.seed(0)
    llama_config = ModelArgs()
    llama = Llama(llama_config)
    llama_pretrained_model_path = "TinyLlama-1.1B-Chat-v1.0"
    model, tokenizer = llama.load_model(llama_pretrained_model_path)
    prompt = "The capital city of Canada: "
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])
    for token in model.generate(x, 0.0):
        s = tokenizer.decode([token.item()])
        print(s, end="", flush=True)
