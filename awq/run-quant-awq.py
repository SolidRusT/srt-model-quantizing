import torch
import os
import gc
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import cProfile, pstats

# Ensure TOKENIZERS_PARALLELISM is set before any tokenizers are loaded
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

access_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
model_path = "Weyaxi/EulerMath-Mistral-7B"
quant_path = 'temp/EulerMath-Mistral-7B-AWQ'
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

def quantize_model():
    gc.enable()
    model = AutoAWQForCausalLM.from_pretrained(model_path, token=access_token, trust_remote_code=True, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token, trust_remote_code=True)
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    print(f'Model is quantized and saved at "{quant_path}"')
    gc.collect()

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    quantize_model()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

    gc.collect()
