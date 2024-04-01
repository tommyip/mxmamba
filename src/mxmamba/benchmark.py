import time

from transformers import AutoTokenizer

from mxmamba.model_rnn import Mamba, generate

TOKENIZER_HF_REPO_ID = 'EleutherAI/gpt-neox-20b'
MAMBA_HF_REPO_ID = 'state-spaces/mamba-130m'

MAX_LENGTH = 100

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_HF_REPO_ID)
model = Mamba.from_pretrained(MAMBA_HF_REPO_ID)

prompt = 'MLX is a machine learning framework developed by'
input_ids = tokenizer(prompt, return_tensors='np').input_ids[0]

start = time.perf_counter()
output_ids = generate(model, input_ids, MAX_LENGTH,
                      eos_token_id=tokenizer.eos_token_id)
elapsed = time.perf_counter() - start

n_tokens = len(output_ids) - len(input_ids)
output = tokenizer.decode(output_ids)

print('Output:', output)
print('Generated tokens:', n_tokens)
print(f'Elapsed: {elapsed:.3f}s ({n_tokens/elapsed:.3f} tok/s)')
