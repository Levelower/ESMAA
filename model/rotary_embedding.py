import paddle
import paddle.nn as nn


def rotate_half(x):
    x1, x2 = paddle.split(x, num_or_sections=2, axis=-1)
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :x.shape[-2], :]
    sin = sin[:, :x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_with_mask(x, cos, sin, aa_mask=None):
    if aa_mask is not None:
        bsz_heads = x.shape[0]
        num_heads = bsz_heads // aa_mask.shape[0]
        aa_mask = aa_mask.unsqueeze(1).expand([-1, num_heads, -1])
        aa_mask = aa_mask.reshape([-1, aa_mask.shape[-1]])
        head_dim = cos.shape[-1]
        x_pos = paddle.cumsum(aa_mask.astype("int64"), axis=-1) - 1
        x_pos = x_pos.unsqueeze(-1).expand([-1, -1, head_dim])
        cos = paddle.gather(cos.expand([bsz_heads, -1, -1]), index=x_pos, axis=1)
        sin = paddle.gather(sin.expand([bsz_heads, -1, -1]), index=x_pos, axis=1)
    else:
        cos = cos[:, :x.shape[-2], :]
        sin = sin[:, :x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, *_, **__):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2).astype('float32') / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        if (seq_len != self._seq_len_cached) or (self._cos_cached is None or self._cos_cached.place != x.place):
            self._seq_len_cached = seq_len
            t = paddle.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            emb = paddle.concat([freqs, freqs], axis=-1)

            self._cos_cached = emb.cos().unsqueeze(0)
            self._sin_cached = emb.sin().unsqueeze(0)

        return self._cos_cached, self._sin_cached

    def forward(self, q, k):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class RotaryEmbeddingWithMask(nn.Layer):
    def __init__(self, dim: int, *_, **__):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2).astype('float32') / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        if (seq_len != self._seq_len_cached) or (self._cos_cached is None or self._cos_cached.place != x.place):
            self._seq_len_cached = seq_len
            t = paddle.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            emb = paddle.concat([freqs, freqs], axis=-1)

            self._cos_cached = emb.cos().unsqueeze(0)
            self._sin_cached = emb.sin().unsqueeze(0)

        return self._cos_cached, self._sin_cached

    def forward(self, q, k, aa_mask):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        return (
            apply_rotary_pos_emb_with_mask(q, self._cos_cached, self._sin_cached, aa_mask),
            apply_rotary_pos_emb_with_mask(k, self._cos_cached, self._sin_cached, aa_mask),
        )
