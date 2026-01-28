import torch
from torch import nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import repeat
import abc
import torch
from functools import cached_property
from einops import einsum, rearrange, repeat
from torch import nn
from einops import rearrange

class TimerXL(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.output_len = model_args["output_len"]

        args = argparse.Namespace()
        args.input_token_len = 96
        args.output_token_len = 96
        args.d_model = 1024
        args.n_heads = 8
        args.e_layers = 8
        args.d_ff = 2048
        args.dropout = 0.1
        args.activation = 'relu'
        args.use_norm = True
        args.flash_attention = False
        args.covariate = False
        args.output_attention = False

        self.model = Model(args)
        self.model.load_state_dict(torch.load('baselines/TimerXL/checkpoint.pth')) 
        # download from https://github.com/thuml/OpenLTM/blob/main/load_pth_ckpt.ipynb
        self.model.requires_grad_(False)

        # 这个模型的输入输出长度都必须是96的倍数

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, train: bool = True, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        input_data = history_data[..., 0] # [B, L, N, C]

        prediction = self.model(input_data, None, None)[:, :self.output_len, :].unsqueeze(-1)

        # print(f"Prediction shape: {prediction.shape}, Output length: {self.output_len}")

        return prediction


class Model(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting 

    Paper: https://arxiv.org/abs/2410.04803
    
    GitHub: https://github.com/thuml/Timer-XL
    
    Citation: @article{liu2024timer,
        title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
        author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        journal={arXiv preprint arXiv:2410.04803},
        year={2024}
    }
    """
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.output_attention = configs.output_attention
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(True, attention_dropout=configs.dropout,
                                    output_attention=self.output_attention, 
                                    d_model=configs.d_model, num_heads=configs.n_heads,
                                    covariate=configs.covariate, flash_attention=configs.flash_attention),
                                    configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head = nn.Linear(configs.d_model, configs.output_token_len)
        self.use_norm = configs.use_norm

    def forecast(self, x, x_mark, y_mark):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B, C, N, D]
        embed_out = self.embedding(x)
        # [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N)
        # [B, C * N, P]
        dec_out = self.head(embed_out)
        # [B, C, N * P]
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def forward(self, x, x_mark, y_mark):
        return self.forecast(x, x_mark, y_mark)



class TimerBlock(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TimerBlock, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, n_vars,
                                           n_tokens, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, n_vars, n_tokens,
                                     attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TimerLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(TimerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            n_vars=n_vars,
            n_tokens=n_tokens,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn



class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None


class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int, num_heads: int, **kwargs):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads

    @abc.abstractmethod
    def forward(self, x, seq_id): ...


class RotaryProjection(Projection):
    def __init__(self, *, proj_width: int, num_heads: int, max_len: int = 512, base: int = 10000):
        super().__init__(proj_width, num_heads)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta,
                             "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x):
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(self, x, seq_id):
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)


class QueryKeyProjection(nn.Module):
    def __init__(self, dim: int, num_heads: int, proj_layer, kwargs=None, partial_factor=None):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            **(kwargs or {}),
        )
        self.key_proj = self.query_proj

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self):
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(self, query, key, query_id, kv_id):
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key




class AttentionBias(nn.Module, abc.ABC):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(self, query_id, kv_id): ...


class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int):
        super().__init__(dim, num_heads)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(self, query_id, kv_id):
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(
            self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = ~ind * weight[:1] + ind * weight[1:]
        return bias

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class TimerMultivariateMask():
    def __init__(self, B, n_vars, n_tokens, device="cpu"):
        mask_shape = [B, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.ones((n_vars, n_vars), dtype=torch.bool).to(device)
            self._mask2 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask = torch.kron(self._mask1, self._mask2)
    @property
    def mask(self):
        return self._mask

class TimerCovariateMask():
    def __init__(self, B, n_vars, n_tokens, device="cpu"):
        mask_shape = [B, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.eye(n_vars, dtype=torch.bool).to(device)
            self._mask2 = torch.tril(torch.ones(mask_shape, dtype=torch.bool)).to(device)
            self._mask = ~torch.kron(self._mask1, self._mask2)
            self._mask[:, :, -n_tokens:, :-n_tokens] = False
            
    @property
    def mask(self):
        return self._mask