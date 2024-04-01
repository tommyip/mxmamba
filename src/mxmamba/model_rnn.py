import math
from typing import Tuple, NamedTuple, List, Optional

import mlx.core as mx
from mlx import nn

from mxmamba import weights_util


# Hyperparameters common to all models
D_CONV = 4
D_STATE = 16
EXPAND = 2
VOCAB_SIZE_PAD_MULTIPLE = 8


class Config(NamedTuple):
    c: int
    d: int
    n: int
    d_conv: int
    dt_rank: int
    vocab_size: int
    n_layer: int

    @staticmethod
    def from_config(config):
        vocab_size = config['vocab_size']
        if vocab_size % VOCAB_SIZE_PAD_MULTIPLE != 0:
            vocab_size += (VOCAB_SIZE_PAD_MULTIPLE -
                           vocab_size % VOCAB_SIZE_PAD_MULTIPLE)
        return Config(
            c=config['d_model'],
            d=config['d_model'] * EXPAND,
            n=D_STATE,
            d_conv=D_CONV,
            dt_rank=math.ceil(config['d_model'] / 16),
            vocab_size=vocab_size,
            n_layer=config['n_layer'])


class Mamba(nn.Module):
    def __init__(self, conf: Config):
        self.conf = conf
        self.embedding = nn.Embedding(conf.vocab_size, conf.c)
        self.layers = [ResidualBlock(conf) for _ in range(conf.n_layer)]
        self.norm_f = nn.RMSNorm(conf.c)
        self.lm_head = nn.Linear(conf.c, conf.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    @staticmethod
    def from_pretrained(hf_repo_id):
        weights, config = weights_util.load_cached_weights(hf_repo_id)
        config = Config.from_config(config)
        model = Mamba(config)
        model.load_weights(list(weights.items()))
        return model

    def empty_caches(self):
        return [
            (mx.zeros((self.conf.d, self.conf.n)), mx.zeros(
                (self.conf.d_conv - 1, self.conf.d)))
            for _ in range(self.conf.n_layer)
        ]


class ResidualBlock(nn.Module):
    def __init__(self, conf: Config):
        self.mixer = MambaBlock(conf)
        self.norm = nn.RMSNorm(conf.c)


class Conv1dStep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight = mx.zeros((out_channels, 1, kernel_size))
        self.bias = mx.zeros(out_channels)

    def __call__(self, x):
        return (self.weight.squeeze(1) * x.T).sum(1) + self.bias


class MambaBlock(nn.Module):
    def __init__(self, conf: Config):
        self.in_proj = nn.Linear(conf.c, conf.d * 2, bias=False)
        self.conv1d = Conv1dStep(conf.d, conf.d, conf.d_conv)
        self.x_proj = nn.Linear(conf.d, conf.dt_rank + conf.n * 2, bias=False)
        self.dt_proj = nn.Linear(conf.dt_rank, conf.d)
        self.out_proj = nn.Linear(conf.d, conf.c, bias=False)
        self.A_log = mx.zeros((conf.d, conf.n))
        self.D = mx.zeros(conf.d)


def step(
    self: Mamba,
    token_id: int,
    caches: List[Tuple[mx.array, mx.array]],
) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
    """
    Args:
        self: Mamba model initialized with weights.
        token_id: The current input token id.
        caches: Hidden state `h` and previous inputs `inputs` for each layer.
            h: shape (d, n) f32 tensor. The RNN state.
            inputs: shape (d_conv - 1, d) f32 tensor. The projected inputs
                from previous steps to compute 1D convolution over the time
                dimension.
    Returns:
        (next_token_logits, caches)
    """
    x = self.embedding(mx.array(token_id))  # (c,)
    for i, layer in enumerate(self.layers):
        h, inputs = caches[i]  # (d, n), (d_conv - 1, d)
        x_ = x  # (c,)
        x = layer.norm(x)
        mixer = layer.mixer
        x_and_res = mixer.in_proj(x)  # (2d,)
        x, res = x_and_res.split(2)  # (d,), (d,)
        x_cache = mx.expand_dims(x, 0)
        x = mixer.conv1d(mx.concatenate([inputs, x_cache]))  # (d,)
        x = nn.silu(x)
        A = -mx.exp(mixer.A_log)  # (d, n)
        deltaBC = mixer.x_proj(x)  # (dt_rank + 2n,)
        delta, B, C = deltaBC.split(
            [self.conf.dt_rank, self.conf.dt_rank + self.conf.n])
        delta = mx.expand_dims(nn.softplus(mixer.dt_proj(delta)), -1)  # (d, 1)
        deltaA = mx.exp(delta * A)  # (d, n)
        deltaB_x = (delta * mx.expand_dims(B, 0) *
                    mx.expand_dims(x, -1))  # (d, n)
        h = deltaA * h + deltaB_x  # (d, n)
        x = (h @ mx.expand_dims(C, -1)).squeeze(1) + mixer.D * x  # (d,)
        x = x * nn.silu(res)
        x = mixer.out_proj(x)  # (c,)
        x += x_
        inputs = mx.concatenate([inputs[1:], x_cache])
        caches[i] = h, inputs
    x = self.norm_f(x)
    return self.lm_head(x), caches


def generate(
    model: Mamba,
    input_ids: List[int],
    max_length: int,
    top_k: int = 40,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    token_ids = [x for x in input_ids]
    caches = model.empty_caches()
    for i in range(max_length):
        logits, caches = step(model, token_ids[i], caches)
        if i + 1 >= input_ids.shape[0]:
            next_token = mx.random.categorical(logits)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            token_ids.append(next_token.item())
    return token_ids


if __name__ == '__main__':
    model = Mamba.from_pretrained('state-spaces/mamba-130m')
    caches = model.empty_caches()
    token_ids = mx.array([1])
    next_token_id, _ = step(model, token_ids, caches)
    print(next_token_id)
