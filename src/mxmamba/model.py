import math

import mlx.core as mx
from mlx import nn


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        d_state: int = 16,
        expand: int = 2,
    ):
        self.d_model = d_model
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
        n = A.shape[1]

        # Discretize A and B
        # exp(einsum('bld, dn -> bldn', delta, A))
        deltaA = mx.exp(mx.expand_dims(delta, 3) * mx.expand_dims(A, (0, 1)))
        # einsum('bld, bln, bld -> bldn', delta, B, u)
        deltaB_u = (mx.expand_dims(delta, 3) *
                    mx.expand_dims(B, 2) * mx.expand_dims(u, 3))

        # Selective scan (sequential implementation)
        x = mx.zeros((b, d, n))
        y = mx.zeros((b, l, d))
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # einsum('bdn, bn -> bd', x, C[:, i])
            y[:, i] = (x * mx.expand_dims(C[:, i], 1)).sum(2)

        return y + u * D


def depthwise_conv1d_weights(torch_weights) -> mx.array:
    """
    MLX does not (yet) support depthwise convolution. We can emulate it with
    normal convolution by zeroing out some weights.

    Args:
        weights: shape (d, 1, d_conv)

    Return:
        shape (d, d_conv, d)
    """
    d, _, d_conv = torch_weights.shape
    mlx_weights = mx.zeros((d, d_conv, d))
    indices = mx.arange(d)
    mlx_weights[indices, :, indices] = torch_weights.squeeze(axis=1)

    return mlx_weights
