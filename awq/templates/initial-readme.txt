---
base_model: {AUTHOR}/{MODEL}
library_name: transformers
tags:
- 4-bit
- AWQ
- text-generation
- autotrain_compatible
- endpoints_compatible
pipeline_tag: text-generation
inference: false
quantized_by: Suparious
---
# {AUTHOR}/{MODEL} AWQ

- Model creator: [{AUTHOR}](https://huggingface.co/{AUTHOR})
- Original model: [{MODEL}](https://huggingface.co/{AUTHOR}/{MODEL})



## How to use

### Install the necessary packages

```bash
pip install --upgrade autoawq autoawq-kernels
```

### Example Python code

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

model_path = "solidrust/{MODEL}-AWQ"
system_message = "You are {MODEL}, incarnated as a powerful AI. You were created by {AUTHOR}."

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

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(prompt_template.format(system_message=system_message,prompt=prompt),
                  return_tensors='pt').input_ids.cuda()

# Generate output
generation_output = model.generate(tokens,
                                  streamer=streamer,
                                  max_new_tokens=512)
```

### About AWQ

AWQ is an efficient, accurate and blazing-fast low-bit weight quantization method, currently supporting 4-bit quantization. Compared to GPTQ, it offers faster Transformers-based inference with equivalent or better quality compared to the most commonly used GPTQ settings.

AWQ models are currently supported on Linux and Windows, with NVidia GPUs only. macOS users: please use GGUF models instead.

It is supported by:

- [Text Generation Webui](https://github.com/oobabooga/text-generation-webui) - using Loader: AutoAWQ
- [vLLM](https://github.com/vllm-project/vllm) - version 0.2.2 or later for support for all model types.
- [Hugging Face Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
- [Transformers](https://huggingface.co/docs/transformers) version 4.35.0 and later, from any code or client that supports Transformers
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - for use from Python code
