import json
from functools import partial
import pytest
import torch
import mlx.core as mx
from transformers.utils.hub import cached_file
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME

from mxmamba.ref_model import (
    MambaBlock as RefMambaBlock,
    ModelArgs as RefModelArgs)
from mxmamba.model import MambaBlock, depthwise_conv1d_weights
from .util import to_mlx, to_torch

TEST_PRETRAINED_MODEL = 'state-spaces/mamba-130m'

allclose = partial(torch.allclose, rtol=1e-5, atol=1e-5)


def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                        _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name):
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                        _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file,
                      weights_only=True,
                      map_location='cpu',
                      mmap=True)


def init_ref_block(config, state_dict, layer: int = 0) -> RefMambaBlock:
    args = RefModelArgs(
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        vocab_size=config['vocab_size']
    )

    new_state_dict = {}
    for key in state_dict:
        if key.startswith(f'backbone.layers.{layer}.mixer'):
            new_key = key.replace(f'backbone.layers.{layer}.mixer.', '')
            new_state_dict[new_key] = state_dict[key]

    block = RefMambaBlock(args)
    block.load_state_dict(new_state_dict)
    block.eval()
    return block


def init_block(config, state_dict, layer: int = 0) -> MambaBlock:
    block = MambaBlock(d_model=config['d_model'])
    weights = []
    for key in state_dict:
        if key.startswith(f'backbone.layers.{layer}.mixer'):
            name = key.replace(f'backbone.layers.{layer}.mixer.', '')
            weight = mx.array(state_dict[key].numpy())
            if name.endswith('conv1d.weight'):
                weight = depthwise_conv1d_weights(weight)
            weights.append((name, weight))
    block.load_weights(weights)
    return block


config = load_config_hf(TEST_PRETRAINED_MODEL)
state_dict = load_state_dict_hf(TEST_PRETRAINED_MODEL)


@pytest.mark.parametrize('seed', range(10))
def test_mamba_block(seed: int):
    torch.manual_seed(seed)
    BATCH_SIZE = 4
    SEQ_LEN = 20

    ref_block = init_ref_block(config, state_dict)
    block = init_block(config, state_dict)

    x = torch.rand((BATCH_SIZE, SEQ_LEN, config['d_model']))

    ref_y = ref_block(x)
    y = to_torch(block(to_mlx(x)))

    max_diff = (y - ref_y).abs().max().item()

    assert allclose(y, ref_y), f'Max diff = {max_diff}'
