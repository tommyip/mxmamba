import mlx
import mlx.core as mx
import torch
import pytest
from einops import einsum

from mxmamba.model import depthwise_conv1d_weights
from .util import to_mlx, to_torch


@pytest.mark.parametrize('seed', range(10))
def test_depthwise_conv1d(seed: int):
    torch.manual_seed(seed)

    L, C, K = 20, 768, 4

    weight = torch.rand((C, 1, K))
    bias = torch.rand(C)

    torch_conv1d = torch.nn.Conv1d(C, C, K, groups=C)
    torch_conv1d.weight = torch.nn.Parameter(weight)
    torch_conv1d.bias = torch.nn.Parameter(bias)
    mlx_conv1d = mlx.nn.Conv1d(C, C, K)
    mlx_conv1d.weight = depthwise_conv1d_weights(to_mlx(weight))
    mlx_conv1d.bias = to_mlx(bias)

    x = torch.rand((1, C, L))

    torch_y = torch_conv1d(x)
    mlx_y = mlx_conv1d(to_mlx(x).swapaxes(1, 2)).swapaxes(1, 2)

    assert torch.allclose(torch_y, to_torch(mlx_y))


@pytest.mark.parametrize('seed', range(10))
def test_deltaA(seed: int):
    torch.manual_seed(seed)

    b, l, d, n = 4, 20, 1536, 16
    delta = torch.randn(b, l, d)
    A = torch.randn(d, n)

    torch_deltaA = torch.exp(
        einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
    mlx_deltaA = mx.exp(mx.expand_dims(to_mlx(delta), 3) * to_mlx(A))

    max_diff = (torch_deltaA - to_torch(mlx_deltaA)).abs().max().item()
    print(max_diff)
    assert torch.allclose(torch_deltaA, to_torch(mlx_deltaA)), max_diff


@pytest.mark.parametrize('seed', range(10))
def test_deltaB_u(seed: int):
    torch.manual_seed(seed)

    b, l, d, n = 4, 20, 1536, 16
    delta = torch.rand(b, l, d)
    B = torch.rand(b, l, n)
    u = torch.rand(b, l, d)

    torch_deltaB_u = einsum(
        delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
    mlx_deltaB_u = (mx.expand_dims(to_mlx(delta), 3) *
                    mx.expand_dims(to_mlx(B), 2) *
                    mx.expand_dims(to_mlx(u), 3))

    assert torch.allclose(torch_deltaB_u, to_torch(mlx_deltaB_u))
