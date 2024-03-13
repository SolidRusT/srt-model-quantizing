import os
import torch
from transformers import AutoTokenizer, TextStreamer
from awq import AutoAWQForCausalLM

# Make CUDA operations synchronous to get accurate error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

model_path = "/srv/home/shaun/repos/Eris-Remix-7B-DPO-AWQ"
system_message = "You Gecko AI. Gecko is really good at math and likes to write stories."

# Load model
model = AutoAWQForCausalLM.from_quantized(model_path,
                                          fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          trust_remote_code=True)

streamer = TextStreamer(tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""

prompt = "Write a story about 2 gay men that made a million dollar business, using their mouths."

tokens = tokenizer(prompt_template.format(system_message=system_message, prompt=prompt),
                   return_tensors='pt').input_ids.cuda()

# Check if token IDs are within the vocabulary range
max_token_id = tokenizer.vocab_size - 1
assert tokens.max() <= max_token_id, "Token ID exceeds vocabulary size."
assert tokens.ge(0).all() and tokens.lt(tokenizer.vocab_size).all(), "Input IDs out of range."

# Assuming generation and streaming work as expected; include relevant checks or configurations if necessary.
try:
    generation_output = model.generate(tokens,
                                  streamer=streamer,
                                  max_new_tokens=512)
    # generation_output = model.generate(tokens, max_new_tokens=512)
    pass
except Exception as e:
    print(f"Error during model generation: {e}")
