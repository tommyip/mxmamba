#!/usr/bin/env python

import sys
from os import path
from pathlib import Path
import json
from typing import Tuple, Dict, Union

import torch
import mlx.core as mx

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

CACHE_DIR = '.cache'
HELP = '''Usage:
    weights_util.py <huggingface repo id>'''


def cache_path(hf_repo_id: str) -> str:
    sanitized_repl_id = hf_repo_id.replace('/', '-')
    return path.join(CACHE_DIR, sanitized_repl_id + '.gguf')


def download_transcode_and_cache(hf_repo_id: str):
    """
    Download pretrained weights from huggingface in PyTorch pickle format
    and transcode to GGUF format.
    """
    config_file = cached_file(hf_repo_id, CONFIG_NAME,
                              _raised_exceptions_for_missing_entries=False)
    state_dict_file = cached_file(hf_repo_id, WEIGHTS_NAME,
                                  _raise_exceptions_for_missing_entries=False)

    with open(config_file) as f:
        config = json.load(f)
        metadata = {k: mx.array(v) for k, v in config.items()
                    if isinstance(v, int) or isinstance(v, bool)}

    state_dict = torch.load(
        state_dict_file, weights_only=True, map_location='cpu')

    arrays = {}
    for key in state_dict:
        name = key.replace('backbone.', '')
        arrays[name] = mx.array(state_dict[key].numpy())

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    mx.save_gguf(cache_path(hf_repo_id), arrays, metadata)


def load_cached_weights(
    hf_repo_id: str,
) -> Tuple[Dict[str, mx.array], Dict[str, Union[int, bool]]]:
    file_path = cache_path(hf_repo_id)
    weights, metadata = mx.load(file_path, return_metadata=True)
    config = {k: v.item() for k, v in metadata.items()}
    return weights, config


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(HELP)
        exit(1)

    download_transcode_and_cache(sys.argv[1])
