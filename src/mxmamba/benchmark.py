import time

import mlx.core as mx
from transformers import AutoTokenizer

from mxmamba.model import Mamba

TOKENIZER_HF_REPO_ID = 'EleutherAI/gpt-neox-20b'
MAMBA_HF_REPO_ID = 'state-spaces/mamba-130m'

MAX_LENGTH = 100

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_HF_REPO_ID)
model = Mamba.from_pretrained(MAMBA_HF_REPO_ID)

prompt = 'MLX is a machine learning framework developed by'
input_ids = mx.array(tokenizer(prompt, return_tensors='np').input_ids)

start = time.perf_counter()
output_ids = model.generate(input_ids, MAX_LENGTH)
elapsed = time.perf_counter() - start

n_tokens = output_ids.shape[1] - input_ids.shape[1]
output = tokenizer.decode(output_ids[0].tolist())

print('Output:', output)
print('Generated tokens:', n_tokens)
print(f'Elapsed: {elapsed:.3f}s ({n_tokens/elapsed:.3f} tok/s)')
