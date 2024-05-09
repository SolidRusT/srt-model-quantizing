import torch
import os
import gc
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import cProfile, pstats
import argparse

def quantize_model(model_path, quant_path, quant_config):
    gc.enable()
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True, use_cache=False, max_memory={0: "11GB",1: "11GB"})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.quantize(tokenizer, quant_config=quant_config)
    print(f'Saving model at {quant_path}')
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    print(f'Model is quantized and saved at "{quant_path}"')
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize a model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--quant_path', type=str, required=True, help='Path to save the quantized model')
    parser.add_argument('--zero_point', type=bool, default=True, help='Use zero point')
    parser.add_argument('--q_group_size', type=int, default=128, help='Quantization group size')
    parser.add_argument('--w_bit', type=int, default=4, help='Weight bit')
    parser.add_argument('--version', type=str, default="GEMM", help='Version')

    args = parser.parse_args()

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }

    profiler = cProfile.Profile()
    profiler.enable()

    quantize_model(args.model_path, args.quant_path, quant_config)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

    gc.collect()
