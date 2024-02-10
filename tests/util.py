import numpy as np
import torch
import mlx.core as mx


def to_mlx(torch_tensor: torch.Tensor) -> mx.array:
    return mx.array(torch_tensor.numpy())


def to_torch(mlx_array: mx.array) -> torch.Tensor:
    return torch.tensor(np.array(mlx_array))
