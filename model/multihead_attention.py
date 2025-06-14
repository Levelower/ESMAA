import math
import uuid
from typing import Dict, Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle import nn

from .rotary_embedding import RotaryEmbedding, RotaryEmbeddingWithMask


def _utils_softmax(x, dim, onnx_trace=False):
    return F.softmax(x.astype("float32"), axis=dim)


def _masked_fill(t, mask, value):
    value_tensor = paddle.full_like(t, value)
    return paddle.where(mask, value_tensor, t)


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key):
        return f"{self._incremental_state_id}.{key}"

    def get_incremental_state(
        self,
        incremental_state,
        key,
    ):
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state,
        key,
        value,
    ):
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b is not FairseqIncrementalState
    )
    return cls


@with_incremental_state
class MultiheadAttention(nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        use_rotary_embeddings=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim 必须能被 num_heads 整除"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, (
            "自注意力要求 query/key/value 维度一致"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias_attr=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias_attr=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

        if add_bias_kv:
            self.bias_k = self.create_parameter(
                shape=[1, 1, embed_dim],
                dtype="float32",
                default_initializer=nn.initializer.XavierNormal()
            )
            self.bias_v = self.create_parameter(
                shape=[1, 1, embed_dim],
                dtype="float32",
                default_initializer=nn.initializer.XavierNormal()
            )
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.rot_emb = RotaryEmbedding(dim=self.head_dim) if use_rotary_embeddings else None

        self.enable_torch_version = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            gain = 1 / math.sqrt(2)
            nn.initializer.XavierUniform(gain=gain)(self.k_proj.weight)
            nn.initializer.XavierUniform(gain=gain)(self.v_proj.weight)
            nn.initializer.XavierUniform(gain=gain)(self.q_proj.weight)
        else:
            nn.initializer.XavierUniform()(self.k_proj.weight)
            nn.initializer.XavierUniform()(self.v_proj.weight)
            nn.initializer.XavierUniform()(self.q_proj.weight)

        nn.initializer.XavierUniform()(self.out_proj.weight)
        if self.out_proj.bias is not None:
            self.out_proj.bias.set_value(paddle.zeros_like(self.out_proj.bias))

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state and static_kv:
                key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * self.scaling

        if self.bias_k is not None:
            k = paddle.concat([k, self.bias_k.tile([1, bsz, 1])], axis=0)
            v = paddle.concat([v, self.bias_v.tile([1, bsz, 1])], axis=0)
            if attn_mask is not None:
                attn_mask = paddle.concat(
                    [attn_mask, paddle.zeros([attn_mask.shape[0], 1], dtype=attn_mask.dtype)],
                    axis=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [key_padding_mask, paddle.zeros([key_padding_mask.shape[0], 1], dtype=key_padding_mask.dtype)],
                    axis=1,
                )

        q = q.reshape([tgt_len, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        if k is not None:
            k = k.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        if v is not None:
            v = v.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])

        if saved_state is not None:
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"].reshape([bsz * self.num_heads, -1, self.head_dim])
                k = prev_key if static_kv else paddle.concat([prev_key, k], axis=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"].reshape([bsz * self.num_heads, -1, self.head_dim])
                v = prev_value if static_kv else paddle.concat([prev_value, v], axis=1)
            prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)
            key_padding_mask = self._append_prev_key_padding_mask(
                key_padding_mask,
                prev_key_padding_mask,
                bsz,
                k.shape[1],
                static_kv,
            )

            saved_state["prev_key"] = k.reshape([bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_value"] = v.reshape([bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_key_padding_mask"] = key_padding_mask
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        src_len = k.shape[1]

        if self.add_zero_attn:
            zero_k = paddle.zeros([k.shape[0], 1, k.shape[2]], dtype=k.dtype)
            zero_v = paddle.zeros([v.shape[0], 1, v.shape[2]], dtype=v.dtype)
            k = paddle.concat([k, zero_k], axis=1)
            v = paddle.concat([v, zero_v], axis=1)
            src_len += 1
            if attn_mask is not None:
                attn_mask = paddle.concat(
                    [attn_mask, paddle.zeros([attn_mask.shape[0], 1], dtype=attn_mask.dtype)],
                    axis=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [key_padding_mask, paddle.zeros([key_padding_mask.shape[0], 1], dtype=key_padding_mask.dtype)],
                    axis=1,
                )

        if self.rot_emb is not None:
            q, k = self.rot_emb(q, k)

        attn_weights = paddle.bmm(q, k.transpose([0, 2, 1]))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.tile([attn_weights.shape[0], 1, 1])
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            attn_weights = _masked_fill(
                attn_weights,
                key_padding_mask.unsqueeze(1).unsqueeze(2).astype("bool"),
                float("-inf"),
            )
            attn_weights = attn_weights.reshape([bsz * self.num_heads, tgt_len, src_len])

        if before_softmax:
            return attn_weights, v 

        attn_weights_float = _utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_probs = F.dropout(attn_weights_float, p=self.dropout, training=self.training)
        attn = paddle.bmm(attn_probs.astype(v.dtype), v)
        attn = attn.transpose([1, 0, 2]).reshape([tgt_len, bsz, embed_dim])
        attn = self.out_proj(attn)

        attn_weights_out = None
        if need_weights:
            attn_weights_out = attn_weights_float.reshape([
                bsz, self.num_heads, tgt_len, src_len
            ]).transpose([1, 0, 2, 3])
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(axis=0)

        return attn, attn_weights_out

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask,
        prev_key_padding_mask,
        batch_size,
        src_len,
        static_kv,
    ) :
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = paddle.concat(
                [prev_key_padding_mask.astype("float32"), key_padding_mask.astype("float32")],
                axis=1,
            )
        elif prev_key_padding_mask is not None:
            filler = paddle.zeros(
                [batch_size, src_len - prev_key_padding_mask.shape[1]],
                dtype=prev_key_padding_mask.dtype,
            )
            new_key_padding_mask = paddle.concat([prev_key_padding_mask.astype("float32"), filler], axis=1)
        elif key_padding_mask is not None:
            filler = paddle.zeros(
                [batch_size, src_len - key_padding_mask.shape[1]],
                dtype=key_padding_mask.dtype,
            )
            new_key_padding_mask = paddle.concat([filler, key_padding_mask.astype("float32")], axis=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(
        self,
        incremental_state,
        new_order,
    ):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer:
            for k in list(input_buffer.keys()):
                buf = input_buffer[k]
                if buf is not None:
                    if self.encoder_decoder_attention and buf.shape[0] == new_order.shape[0]:
                        break
                    input_buffer[k] = paddle.index_select(buf, index=new_order, axis=0)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(self, incremental_state):
        return self.get_incremental_state(incremental_state, "attn_state") or {}

    def _set_input_buffer(self, incremental_state, buffer):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = f"{name}." if name else ""
        items_to_add, keys_to_remove = {}, []
        for k in list(state_dict.keys()):
            if k.endswith(prefix + "in_proj_weight"):
                dim = state_dict[k].shape[0] // 3
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]
                keys_to_remove.append(k)
                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict:
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
                    keys_to_remove.append(k_bias)
        for k in keys_to_remove:
            del state_dict[k]
        state_dict.update(items_to_add)



@with_incremental_state
class MultiheadAttentionWithBias(nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        use_rotary_embeddings=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim 必须能被 num_heads 整除"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, (
            "自注意力要求 query/key/value 维度一致"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias_attr=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias_attr=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

        if add_bias_kv:
            self.bias_k = self.create_parameter(
                shape=[1, 1, embed_dim],
                dtype="float32",
                default_initializer=nn.initializer.XavierNormal(),
            )
            self.bias_v = self.create_parameter(
                shape=[1, 1, embed_dim],
                dtype="float32",
                default_initializer=nn.initializer.XavierNormal(),
            )
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.onnx_trace = False
        self.rot_emb = RotaryEmbeddingWithMask(dim=self.head_dim) if use_rotary_embeddings else None
        self.enable_torch_version = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            gain = 1 / math.sqrt(2)
            nn.initializer.XavierUniform(gain=gain)(self.k_proj.weight)
            nn.initializer.XavierUniform(gain=gain)(self.v_proj.weight)
            nn.initializer.XavierUniform(gain=gain)(self.q_proj.weight)
        else:
            nn.initializer.XavierUniform()(self.k_proj.weight)
            nn.initializer.XavierUniform()(self.v_proj.weight)
            nn.initializer.XavierUniform()(self.q_proj.weight)
        nn.initializer.XavierUniform()(self.out_proj.weight)
        if self.out_proj.bias is not None:
            self.out_proj.bias.set_value(paddle.zeros_like(self.out_proj.bias))

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        aa_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
        return_pair_rep=False,
    ):
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state and static_kv:
                key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * self.scaling

        if self.bias_k is not None:
            k = paddle.concat([k, self.bias_k.tile([1, bsz, 1])], axis=0)
            v = paddle.concat([v, self.bias_v.tile([1, bsz, 1])], axis=0)
            if attn_mask is not None:
                attn_mask = paddle.concat(
                    [attn_mask, paddle.zeros([attn_mask.shape[0], 1], dtype=attn_mask.dtype)],
                    axis=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [key_padding_mask, paddle.zeros([key_padding_mask.shape[0], 1], dtype=key_padding_mask.dtype)],
                    axis=1,
                )

        q = q.reshape([tgt_len, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        if k is not None:
            k = k.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        if v is not None:
            v = v.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])

        if saved_state is not None:
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"].reshape([bsz * self.num_heads, -1, self.head_dim])
                k = prev_key if static_kv else paddle.concat([prev_key, k], axis=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"].reshape([bsz * self.num_heads, -1, self.head_dim])
                v = prev_value if static_kv else paddle.concat([prev_value, v], axis=1)
            prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)
            key_padding_mask = self._append_prev_key_padding_mask(
                key_padding_mask,
                prev_key_padding_mask,
                bsz,
                k.shape[1],
                static_kv,
            )
            saved_state["prev_key"] = k.reshape([bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_value"] = v.reshape([bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_key_padding_mask"] = key_padding_mask
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        src_len = k.shape[1]

        if self.add_zero_attn:
            zero_k = paddle.zeros([k.shape[0], 1, k.shape[2]], dtype=k.dtype)
            zero_v = paddle.zeros([v.shape[0], 1, v.shape[2]], dtype=v.dtype)
            k = paddle.concat([k, zero_k], axis=1)
            v = paddle.concat([v, zero_v], axis=1)
            src_len += 1
            if attn_mask is not None:
                attn_mask = paddle.concat(
                    [attn_mask, paddle.zeros([attn_mask.shape[0], 1], dtype=attn_mask.dtype)],
                    axis=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [key_padding_mask, paddle.zeros([key_padding_mask.shape[0], 1], dtype=key_padding_mask.dtype)],
                    axis=1,
                )

        if self.rot_emb is not None:
            q, k = self.rot_emb(q, k, aa_mask)

        attn_weights = paddle.bmm(q, k.transpose([0, 2, 1]))
        attn_weights = MultiheadAttentionWithBias.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask == -np.inf, -1e8 if attn_mask.dtype == paddle.float32 else -1e4)
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            attn_weights = _masked_fill(
                attn_weights,
                key_padding_mask.unsqueeze(1).unsqueeze(2).astype("bool"),
                float("-inf"),
            )
            attn_weights = attn_weights.reshape([bsz * self.num_heads, tgt_len, src_len])

        if before_softmax:
            return attn_weights, v, attn_weights

        pair_rep = None
        if return_pair_rep:
            pair_rep = paddle.clip(attn_weights, min=-1e30, max=1e30)
            pair_rep = pair_rep.masked_fill(pair_rep == -np.inf, 0)

        attn_weights_float = _utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_probs = F.dropout(attn_weights_float, p=self.dropout, training=self.training)
        attn = paddle.bmm(attn_probs.astype(v.dtype), v)
        attn = attn.transpose([1, 0, 2]).reshape([tgt_len, bsz, embed_dim])
        attn = self.out_proj(attn)

        attn_weights_out=None
        if need_weights:
            attn_weights_out = attn_weights_float.reshape([
                bsz, self.num_heads, tgt_len, src_len
            ]).transpose([1, 0, 2, 3])
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(axis=0)

        return attn, attn_weights_out, pair_rep

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask,
        prev_key_padding_mask,
        batch_size,
        src_len,
        static_kv,
    ):
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = paddle.concat(
                [prev_key_padding_mask.astype("float32"), key_padding_mask.astype("float32")],
                axis=1,
            )
        elif prev_key_padding_mask is not None:
            filler = paddle.zeros(
                [batch_size, src_len - prev_key_padding_mask.shape[1]],
                dtype=prev_key_padding_mask.dtype,
            )
            new_key_padding_mask = paddle.concat(
                [prev_key_padding_mask.astype("float32"), filler], axis=1
            )
        elif key_padding_mask is not None:
            filler = paddle.zeros(
                [batch_size, src_len - key_padding_mask.shape[1]],
                dtype=key_padding_mask.dtype,
            )
            new_key_padding_mask = paddle.concat([filler, key_padding_mask.astype("float32")], axis=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(
        self,
        incremental_state,
        new_order,
    ):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer:
            for k in list(input_buffer.keys()):
                buf = input_buffer[k]
                if buf is not None:
                    if self.encoder_decoder_attention and buf.shape[0] == new_order.shape[0]:
                        break
                    input_buffer[k] = paddle.index_select(buf, index=new_order, axis=0)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(self, incremental_state):
        return self.get_incremental_state(incremental_state, "attn_state") or {}

    def _set_input_buffer(self, incremental_state, buffer):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = f"{name}." if name else ""
        items_to_add, keys_to_remove = {}, []
        for k in list(state_dict.keys()):
            if k.endswith(prefix + "in_proj_weight"):
                dim = state_dict[k].shape[0] // 3
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]
                keys_to_remove.append(k)
                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict:
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
                    keys_to_remove.append(k_bias)
        for k in keys_to_remove:
            del state_dict[k]
        state_dict.update(items_to_add)
