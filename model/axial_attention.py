import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class RowSelfAttention(nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa=2 ** 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.shape[0]
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, start : start + max_rows]
                if self_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = F.softmax(attns, axis=-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(x[start : start + max_rows], attn_probs)
            outputs.append(output)

        output = paddle.concat(outputs, axis=0)
        return output, attn_probs

    def compute_attention_weights(self, x, scaling, self_attn_mask=None, self_attn_padding_mask=None):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        q = self.q_proj(x).reshape([num_rows, num_cols, batch_size, self.num_heads, self.head_dim])
        k = self.k_proj(x).reshape([num_rows, num_cols, batch_size, self.num_heads, self.head_dim])
        q = q * scaling

        if self_attn_padding_mask is not None:
            mask = 1 - self_attn_padding_mask.transpose([1, 2, 0]).unsqueeze(3).unsqueeze(4).astype(q.dtype)
            q *= mask

        attn_weights = paddle.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if self_attn_mask is not None:
            raise NotImplementedError

        if self_attn_padding_mask is not None:
            attn_weights = attn_weights.astype("float32")  # ensure numerical stability
            mask = self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2)
            attn_weights = paddle.where(
                mask,
                paddle.full_like(attn_weights, -10000),
                attn_weights
            )

        return attn_weights

    def compute_attention_update(self, x, attn_probs):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        v = self.v_proj(x).reshape([num_rows, num_cols, batch_size, self.num_heads, self.head_dim])
        context = paddle.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = paddle.reshape(context, [num_rows, num_cols, batch_size, embed_dim])
        output = self.out_proj(context)
        return output

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        if (num_rows * num_cols > self.max_tokens_per_msa) and not paddle.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(
                x, scaling, self_attn_mask, self_attn_padding_mask
            )
            attn_probs = F.softmax(attn_weights, axis=-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs


class ColumnSelfAttention(nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa=2 ** 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def _batched_forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                x[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, :, start : start + max_cols]
                if self_attn_padding_mask is not None
                else None,
            )
            outputs.append(output)
            attns.append(attn)
        output = paddle.concat(outputs, axis=1)
        attns = paddle.concat(attns, axis=1)
        return output, attns

    def compute_attention_update(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        if num_rows == 1:
            attn_probs = paddle.ones([
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows
            ], dtype=x.dtype)
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).reshape([num_rows, num_cols, batch_size, self.num_heads, self.head_dim])
            k = self.k_proj(x).reshape([num_rows, num_cols, batch_size, self.num_heads, self.head_dim])
            v = self.v_proj(x).reshape([num_rows, num_cols, batch_size, self.num_heads, self.head_dim])
            q = q * self.scaling

            attn_weights = paddle.einsum("icnhd,jcnhd->hcnij", q, k)

            if self_attn_mask is not None:
                raise NotImplementedError
            if self_attn_padding_mask is not None:
                mask = self_attn_padding_mask.transpose([2, 0, 1]).unsqueeze(0).unsqueeze(3)
                attn_weights = paddle.where(
                    mask,
                    paddle.full_like(attn_weights, -10000),
                    attn_weights
                )

            attn_probs = F.softmax(attn_weights, axis=-1)
            attn_probs = self.dropout_module(attn_probs)
            context = paddle.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.reshape([num_rows, num_cols, batch_size, embed_dim])
            output = self.out_proj(context)
        return output, attn_probs

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        if (num_rows * num_cols) > self.max_tokens_per_msa and not paddle.is_grad_enabled():
            return self._batched_forward(
                x,
                self_attn_mask,
                self_attn_padding_mask,
            )
        else:
            return self.compute_attention_update(x, self_attn_mask, self_attn_padding_mask)
