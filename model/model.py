import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union

from .tokenizer import Alphabet
from .modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayerWithPairRep

def symmetrize(x):
    return x + paddle.transpose(x, [0, 1, 3, 2])

def apc(x):
    a1 = x.sum(-1, keepdim=True)
    a2 = x.sum(-2, keepdim=True)
    a12 = x.sum([-1, -2], keepdim=True)
    avg = a1 * a2 / a12
    return x - avg

def gelu(x):
    return F.gelu(x.astype('float32')).astype(x.dtype)

class NonLinearHead(nn.Layer):
    def __init__(self, input_dim, out_dim, hidden=None):
        super().__init__()
        hidden = input_dim if hidden is None else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = gelu

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class DistanceHead(nn.Layer):
    def __init__(self, heads):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = gelu

    def forward(self, x):
        bsz, seq_len, _, _ = x.shape
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).reshape([bsz, seq_len, seq_len])
        x = (x + paddle.transpose(x, [0, 2, 1])) * 0.5
        return x

def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return paddle.exp(-0.5 * ((x - mean) / std) ** 2) / (a * std)

class GaussianLayer(nn.Layer):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.means.weight.set_value(paddle.uniform([1, K], min=0, max=3))
        self.stds.weight.set_value(paddle.uniform([1, K], min=0, max=3))
        self.bias.weight.set_value(paddle.zeros([edge_types, 1]))
        self.mul.weight.set_value(paddle.ones([edge_types, 1]))

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).astype(x.dtype)
        bias = self.bias(edge_type).astype(x.dtype)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand([-1, -1, -1, self.K])
        mean = self.means.weight.reshape([-1])
        std = paddle.abs(self.stds.weight.reshape([-1])) + 1e-5
        return gaussian(x.astype('float32'), mean, std).astype(self.means.weight.dtype)

class ESM2_AA(nn.Layer):
    def __init__(
        self,
        num_layers=33,
        embed_dim=1280,
        attention_heads=20,
        alphabet: Union[Alphabet, str] = "ESM-AA",
        token_dropout=True,
        build_dist_head=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.build_dist_head = build_dist_head

        self.embed_scale = 1.0
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx
        )

        self.layers = nn.LayerList([
            TransformerLayerWithPairRep(
                self.embed_dim,
                4 * self.embed_dim,
                self.attention_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
            ) for _ in range(self.num_layers)
        ])

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

        K = 128
        n_edge_type = self.alphabet_size * self.alphabet_size
        self.gbf_proj = NonLinearHead(K, self.attention_heads)
        self.gbf = GaussianLayer(K, n_edge_type)

        if self.build_dist_head:
            self.dist_head = DistanceHead(self.attention_heads)

    def forward(self, tokens, src_distance=None, src_edge_type=None, aa_mask=None, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        bsz, seq_len = tokens.shape

        if src_distance is None:
            src_distance = paddle.zeros([bsz, seq_len, seq_len], dtype='float32')
        if src_edge_type is None:
            src_edge_type = paddle.full([bsz, seq_len, seq_len], -1, dtype='int64')

        not_valid_pair_mask = src_edge_type == -1

        def get_dist_features(dist, et):
            dist = dist.astype('float32')
            et = et.astype('int64')

            not_valid_mask = et == -1
            et_valid   = paddle.where(not_valid_mask, paddle.zeros_like(et),   et)
            dist_valid = paddle.where(not_valid_mask, paddle.zeros_like(dist), dist)

            gbf_feature     = self.gbf(dist_valid, et_valid)              # [B,L,L,K]
            gbf_result      = self.gbf_proj(gbf_feature)                  # [B,L,L,H]
            graph_attn_bias = gbf_result.transpose([0, 3, 1, 2])          # [B,H,L,L]

            nv_mask_4d      = not_valid_mask.unsqueeze(1).astype('bool')  # [B,1,L,L]
            graph_attn_bias = paddle.where(
                nv_mask_4d.expand_as(graph_attn_bias), 
                paddle.zeros_like(graph_attn_bias),
                graph_attn_bias
            )

            graph_attn_bias = graph_attn_bias.reshape([-1, seq_len, seq_len])
            return graph_attn_bias

        attn_bias = get_dist_features(src_distance, src_edge_type)
        padding_mask = tokens == self.padding_idx

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            mask = (tokens == self.mask_idx).unsqueeze(-1)
            x = paddle.where(mask, paddle.zeros_like(x), x)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).astype('int64').sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).astype('float32').sum(-1) / src_lengths.astype('float32')
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed).unsqueeze(-1).unsqueeze(-1)

        if padding_mask.any():
            x = x * (~padding_mask).astype(x.dtype).unsqueeze(-1)

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        attn_weights = [] if need_head_weights else None
        x = x.transpose([1, 0, 2])

        if not_valid_pair_mask.shape[0] < attn_bias.shape[0]:
            num_heads = attn_bias.shape[0] // not_valid_pair_mask.shape[0]
            not_valid_pair_mask = not_valid_pair_mask.unsqueeze(1).expand([-1, num_heads, -1, -1])
            not_valid_pair_mask = not_valid_pair_mask.reshape([-1, seq_len, seq_len])
        attn_bias = paddle.where(not_valid_pair_mask, paddle.zeros_like(attn_bias), attn_bias)

        for layer_idx, layer in enumerate(self.layers):
            x, attn, pair_rep = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
                attn_mask=attn_bias,
                aa_mask=aa_mask,
            )
            if not_valid_pair_mask is not None:
                pair_rep = paddle.where(not_valid_pair_mask, paddle.zeros_like(pair_rep), pair_rep)
                
            attn_bias = pair_rep

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose([1, 0, 2])
            if need_head_weights:
                attn_weights.append(attn.transpose([1, 0, 2, 3]))

        x = self.emb_layer_norm_after(x)
        pair_rep = pair_rep.reshape([bsz, -1, seq_len, seq_len]).transpose([0, 2, 3, 1])
        x = x.transpose([1, 0, 2])

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations, "pair_rep": pair_rep}

        if need_head_weights:
            attentions = paddle.stack(attn_weights, axis=1)
            if padding_mask.any():
                attention_mask = (~padding_mask).astype(attentions.dtype)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask.unsqueeze(1).unsqueeze(1)
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]