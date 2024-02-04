import json

import numpy as np
import mlx.core as mx
import torch
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from mxmamba.model import MambaBlock, depthwise_conv1d_weights
from mxmamba.ref_model import (
    MambaBlock as RefMambaBlock,
    ModelArgs as RefModelArgs)

TEST_PRETRAINED_MODEL = 'state-spaces/mamba-130m'


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


def test_mamba_block():
    BATCH_SIZE = 4
    SEQ_LEN = 20

    ref_block = init_ref_block(config, state_dict)
    block = init_block(config, state_dict)

    x = mx.random.normal((BATCH_SIZE, SEQ_LEN, config['d_model']))
    xt = torch.tensor(np.array(x, copy=False))

    y = block(x)
    with torch.no_grad():
        yt = ref_block(xt)
        yt = mx.array(np.array(yt, copy=False))

    breakpoint()

    assert mx.allclose(y, yt).item()
