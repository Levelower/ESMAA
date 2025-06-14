import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .multihead_attention import MultiheadAttention, MultiheadAttentionWithBias
from .axial_attention import ColumnSelfAttention, RowSelfAttention


def gelu(x):
    return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))

def symmetrize(x):
    return x + paddle.transpose(x, perm=list(range(len(x.shape) - 2)) + [-1, -2])

def apc(x):
    a1 = x.sum(-1, keepdim=True)
    a2 = x.sum(-2, keepdim=True)
    a12 = x.sum((-1, -2), keepdim=True)
    avg = a1 * a2 / a12
    return x - avg

class ESM1LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = self.create_parameter([hidden_size], default_initializer=nn.initializer.Constant(1.0))
            self.bias = self.create_parameter([hidden_size], default_initializer=nn.initializer.Constant(0.0))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        x = (x - mean) / paddle.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x

ESM1bLayerNorm = nn.LayerNorm

class TransformerLayer(nn.Layer):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, add_bias_kv=True, use_esm1b_layer_norm=False, use_rotary_embeddings=False):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, attention_heads, add_bias_kv=add_bias_kv, use_rotary_embeddings=use_rotary_embeddings)
        norm_cls = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm
        self.self_attn_layer_norm = norm_cls(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = norm_cls(embed_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(x, x, x, key_padding_mask=self_attn_padding_mask, need_weights=True, need_head_weights=need_head_weights, attn_mask=self_attn_mask)
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x, attn

class TransformerLayerWithPairRep(nn.Layer):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, add_bias_kv=True, use_esm1b_layer_norm=False, use_rotary_embeddings=False):
        super().__init__()
        self.self_attn = MultiheadAttentionWithBias(embed_dim, attention_heads, add_bias_kv=add_bias_kv, use_rotary_embeddings=use_rotary_embeddings)
        norm_cls = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm
        self.self_attn_layer_norm = norm_cls(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = norm_cls(embed_dim)

    def forward(self, x, attn_mask=None, self_attn_padding_mask=None, aa_mask=None, need_head_weights=False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn, pair_rep = self.self_attn(x, x, x, key_padding_mask=self_attn_padding_mask, aa_mask=aa_mask, need_weights=True, need_head_weights=need_head_weights, attn_mask=attn_mask, return_pair_rep=True)
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x, attn, pair_rep

class NormalizedResidualBlock(nn.Layer):
    def __init__(self, layer, embedding_dim, dropout=0.1):
        super().__init__()
        self.layer = layer
        self.dropout_module = nn.Dropout(dropout)
        self.layer_norm = ESM1bLayerNorm(embedding_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *rest = outputs
        else:
            rest = None
        x = self.dropout_module(x)
        x = residual + x
        return (x, *rest) if rest else x

class FeedForwardNetwork(nn.Layer):
    def __init__(self, embedding_dim, ffn_embedding_dim, activation_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(activation_dropout)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AxialTransformerLayer(nn.Layer):
    def __init__(self, embedding_dim=768, ffn_embedding_dim=3072, num_attention_heads=8, dropout=0.1, attention_dropout=0.1, activation_dropout=0.1, max_tokens_per_msa=2**14):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        row_self_attention = RowSelfAttention(embedding_dim, num_attention_heads, dropout=dropout, max_tokens_per_msa=max_tokens_per_msa)
        column_self_attention = ColumnSelfAttention(embedding_dim, num_attention_heads, dropout=dropout, max_tokens_per_msa=max_tokens_per_msa)
        feed_forward_layer = FeedForwardNetwork(embedding_dim, ffn_embedding_dim, activation_dropout=activation_dropout)

        self.row_self_attention = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer):
        return NormalizedResidualBlock(layer, self.embedding_dim, self.dropout_prob)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
        x, row_attn = self.row_self_attention(x, self_attn_mask=self_attn_mask, self_attn_padding_mask=self_attn_padding_mask)
        x, column_attn = self.column_self_attention(x, self_attn_mask=self_attn_mask, self_attn_padding_mask=self_attn_padding_mask)
        x = self.feed_forward_layer(x)
        return (x, column_attn, row_attn) if need_head_weights else x

class RobertaLMHead(nn.Layer):
    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = self.create_parameter([output_dim], default_initializer=nn.initializer.Constant(0.0))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        B, L, D = x.shape
        x = F.linear(x, self.weight.T, self.bias)
        x = x.reshape([B, L, -1])
        return x

class ContactPredictionHead(nn.Layer):
    def __init__(self, in_features, prepend_bos, append_eos, bias=True, eos_idx=None):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias_attr=bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        if self.append_eos:
            eos_mask = (tokens != self.eos_idx).astype(attentions.dtype)  # [B, T]
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)      # [B, T, T]
            attentions = attentions * eos_mask.unsqueeze(1).unsqueeze(1)  # [B, L, H, T, T]
            attentions = attentions[..., :-1, :-1]
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]

        b, l, h, t, _ = attentions.shape
        attentions = attentions.reshape([b, l * h, t, t])  # [B, C, T, T]

        attentions = attentions.astype(self.regression.weight.dtype)
        attentions = apc(symmetrize(attentions))           # [B, C, T, T]
        attentions = attentions.transpose([0, 2, 3, 1])     # [B, T, T, C]

        return self.activation(self.regression(attentions).squeeze(-1))  # [B, T, T]


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx=padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input):
        if input.shape[1] > self.max_positions:
            raise ValueError(f"Sequence length {input.shape[1]} exceeds max position {self.max_positions}")
        mask = (input != self.padding_idx).astype("int64")
        positions = (paddle.cumsum(mask, axis=1) * mask).astype("int64") + self.padding_idx
        return F.embedding(positions, self.weight, padding_idx=self.padding_idx)

class SinusoidalPositionalEmbedding(nn.Layer):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", paddle.to_tensor([1.0]))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.shape[0]:
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.astype(self._float_tensor.dtype)
        positions = self.make_positions(x)
        return paddle.index_select(self.weights, positions.flatten(), axis=0).reshape([bsz, seq_len, -1]).detach()

    def make_positions(self, x):
        mask = (x != self.padding_idx).astype("int64")
        range_buf = paddle.arange(x.shape[1], dtype="int64") + self.padding_idx + 1
        range_buf = range_buf.unsqueeze(0).expand_as(x)
        return range_buf * mask + self.padding_idx * (1 - mask)

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim, dtype="float32") * -emb)
        emb = paddle.arange(num_embeddings, dtype="float32").unsqueeze(1) * emb.unsqueeze(0)
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=1)
        if self.embed_dim % 2 == 1:
            emb = paddle.concat([emb, paddle.zeros([num_embeddings, 1])], axis=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb
