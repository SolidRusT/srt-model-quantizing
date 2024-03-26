import torch
#torch.cuda.is_available = lambda: False
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import gc
import cProfile, pstats

access_token = os.getenv('HUGGING_FACE_HUB_TOKEN')

model_path = "temp/LewdMistral-7B-0.2"
quant_path = 'temp/LewdMistral-7B-0.2-AWQ'

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}
# "modules_to_not_convert": modules_to_not_convert

def quantize_model():
    # Explicitly enabling garbage collection
    gc.enable()

    # Load model and tokenizer directly to CPU
    model = AutoAWQForCausalLM.from_pretrained(
      model_path, token=access_token, trust_remote_code=True, **{"use_cache": False} #, safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token, trust_remote_code=True)

    # Perform model quantization entirely on CPU
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model and tokenizer
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')

    # Cleanup
    gc.collect()

try:
    profiler = cProfile.Profile()
    profiler.enable()

    # Running the quantization process
    quantize_model()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("CUDA out of memory error occurred, despite optimization attempts.")
    else:
        raise
# "model_max_length": 1000000000000000019884624838656,
