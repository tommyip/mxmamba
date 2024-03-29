import math
from typing import Tuple

import mlx.core as mx
from mlx import nn

from mxmamba import weights_util
from mxmamba.scan import scan


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layer: int,
        d_conv: int = 4,
        d_state: int = 16,
        expand: int = 2,
        pad_vocab_size_multiple: int = 8,
    ):
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += (pad_vocab_size_multiple -
                           vocab_size % pad_vocab_size_multiple)
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.d_state = d_state
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = [ResidualBlock(d_model, d_conv, d_state, expand)
                       for _ in range(n_layer)]
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def __call__(self, input_ids) -> mx.array:
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        return self.lm_head(x)

    def step(self, input_ids, caches):
        x = self.embedding(input_ids)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        x = self.norm_f(x)

        return self.lm_head(x), caches

    def generate(
        self,
        input_ids: mx.array,
        max_length: int,
        top_k: int = 40,
        eos_token_id: int = None
    ) -> mx.array:
        seq_ids = input_ids
        # (h, inputs) per layer
        caches = [(
            mx.zeros((1, self.d_inner, self.d_state)),
            mx.zeros((1, self.d_conv - 1, self.d_inner))
        ) for _ in range(len(self.layers))]
        for i in range(max_length):
            logits, caches = self.step(seq_ids[:, i], caches)
            if i + 1 >= input_ids.shape[1]:
                next_token = mx.random.categorical(logits)
                if eos_token_id is not None and next_token[0] == eos_token_id:
                    break
                seq_ids = mx.concatenate(
                    [seq_ids, mx.expand_dims(next_token, 1)], axis=1)
        return seq_ids

    @staticmethod
    def from_pretrained(hf_repo_id: str):
        weights, config = weights_util.load_cached_weights(hf_repo_id)
        model = Mamba(
            d_model=config['d_model'],
            vocab_size=config['vocab_size'],
            n_layer=config['n_layer'],
        )
        weight_list = []
        for name, weight in weights.items():
            if name.endswith('conv1d.weight'):
                weight = depthwise_conv1d_weights(weight)
            weight_list.append((name, weight))
        model.load_weights(weight_list)
        return model


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        d_state: int = 16,
        expand: int = 2,
    ):
        self.mixer = MambaBlock(d_model, d_conv, d_state, expand)
        self.norm = nn.RMSNorm(d_model)

    def __call__(self, x) -> mx.array:
        """
        Args:
            x: shape (b, l, c)

        Returns:
            shape (b, l, c)
        """
        return self.mixer(self.norm(x)) + x

    def step(self, x, cache) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Args:
            x : shape (b, c)
            cache : (h, inputs)
                h : (b, d, n)
                inputs : shape (b, d_conv - 1, d)
        Return:
            (y, (h, inputs))
                y : shape (b, c)
        """
        output, cache = self.mixer.step(self.norm(x), cache)
        return output + x, cache


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        d_state: int = 16,
        expand: int = 2,
    ):
        self.d_model = d_model
        self.d_conv = d_conv
        self.d_state = d_state
        self.expand = expand
        self.dt_rank = dt_rank = math.ceil(d_model / 16)
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.A_log = mx.zeros((d_inner, d_state))
        self.D = mx.zeros(d_inner)

    def __call__(self, x) -> mx.array:
        """
        Args:
            x: shape (b, l, c)

        Returns:
            shape (b, l, c)
        """
        _, l, _ = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2d)
        # shape (b, l, d)
        x, res = x_and_res.split(2, axis=-1)
        x = self.conv1d(x)[:, :l]
        x = nn.silu(x)
        y = self.ssm(x)
        y = y * nn.silu(res)
        return self.out_proj(y)

    def ssm(self, x) -> mx.array:
        """
        Args:
            x: shape (b, l, d)

        Returns:
            shape (b, l, d)
        """
        A = -mx.exp(self.A_log)  # shape (d, n)
        D = self.D  # shape (d,)

        x_dbl = self.x_proj(x)  # shape (b, l, dt_rank + 2n)

        # shape delta (b, l, dt_rank). B, C (b, l, n)
        delta, B, C = x_dbl.split(
            [self.dt_rank, self.dt_rank + self.d_state], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))  # shape (b, l, d)

        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D) -> mx.array:
        """
        Args:
            u: shape (b, l, d)
            delta: shape (b, l, d)
            A: shape (d, n)
            B, C: shape (b, l, n)
            D: shape (d,)

        Return:
            shape (b, l, d)
        """
        b, l, d = u.shape

        # Discretize A and B
        # exp(einsum('bld, dn -> bldn', delta, A))
        deltaA = mx.exp(mx.expand_dims(delta, 3) * A)
        # einsum('bld, bln, bld -> bldn', delta, B, u)
        deltaB_u = (mx.expand_dims(delta, 3) *
                    mx.expand_dims(B, 2) * mx.expand_dims(u, 3))

        y = scan(deltaA, deltaB_u, C)

        return y + u * D

    def step(self, x, cache) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Args:
            x : shape (b, c)
            cache : (h, inputs)
                h : shape (b, d, n)
                inputs : (b, d_conv - 1, d)
        Returns:
            (y, cache)
                y : shape (b, c)
        """
        h, inputs = cache

        x_and_res = self.in_proj(x)  # (b, 2d)
        x, res = x_and_res.split(2, axis=1)  # (b, d), (b, d)
        x_cache = mx.expand_dims(x, 1)
        x = self.conv1d(mx.concatenate([inputs, x_cache], axis=1))[
            :, self.d_conv - 1]

        x = nn.silu(x)
        y, h = self.ssm_step(x, h)

        res = nn.silu(res)

        y = y * res
        y = self.out_proj(y)

        inputs = mx.concatenate([inputs[:, 1:], x_cache], axis=1)

        return y, (h, inputs)

    def ssm_step(self, x, h) -> mx.array:
        """
        Args:
            x : shape (b, d)
            h : shape (b, d, n)
        Returns:
            (y, h)
            y : shape (b, d)
            h : shape (b, d, n)
        """
        A = -mx.exp(self.A_log)  # (d, n)
        D = self.D

        deltaBC = self.x_proj(x)  # (b, dt_rank + 2n)
        # delta : (b, dt_rank)
        # B, C : (b, n)
        delta, B, C = deltaBC.split(
            [self.dt_rank, self.dt_rank + self.d_state], axis=1)
        delta = nn.softplus(self.dt_proj(delta))  # (b, d)

        deltaA = mx.exp(mx.expand_dims(delta, -1) * A)  # (b, d, n)
        deltaB_x = (mx.expand_dims(delta, -1) * mx.expand_dims(B, 1) *
                    mx.expand_dims(x, -1))  # (b, d, n)

        h = deltaA * h + deltaB_x  # (b, d, n)
        y = (h @ mx.expand_dims(C, -1)).squeeze(2)  # (b, d)
        y = y + D * x

        return y, h


def depthwise_conv1d_weights(torch_weights) -> mx.array:
    """
    MLX does not (yet) support depthwise convolution. We can emulate it with
    normal convolution by zeroing out some weights.

    Args:
        weights: shape (d, 1, d_conv)

    Return:
        shape (d, d_conv, d)

    Credit:
        Repo: https://github.com/alxndrTL/mamba.py
        Source: https://github.com/alxndrTL/mamba.py/blob/e1fc13f59c7780bf538568717da667236f5e1d3d/mlx/misc.py#L77-L106
        License: MIT
    """
    d, _, d_conv = torch_weights.shape
    mlx_weights = mx.zeros((d, d_conv, d))
    indices = mx.arange(d)
    mlx_weights[indices, :, indices] = torch_weights.squeeze(axis=1)

    return mlx_weights
