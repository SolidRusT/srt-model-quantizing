# app/quantization.py

import os
import logging
from typing import Dict, Any
from tqdm import tqdm
from app.config import Config
import shutil
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
from app.converter import load_safetensors_model, save_safetensors_model  # Add this import

logger = logging.getLogger(__name__)

def run_quantization(model_path: str, quant_config: Dict[str, Any], output_dir: str) -> None:
    """
    Run the quantization process on the given model using AutoAWQ.

    Args:
        model_path (str): Path to the model directory.
        quant_config (Dict[str, Any]): Configuration for quantization.
        output_dir (str): Directory to save the quantized model.
    """
    try:
        logger.info(f"Starting quantization for model at {model_path}")
        print(f"Starting quantization for model at {model_path}")
        
        # Load model and tokenizer
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Quantize
        logger.info("Performing AWQ quantization")
        print("Performing AWQ quantization")
        model.quantize(tokenizer, quant_config=quant_config)

        # Save quantized model
        logger.info(f"Saving quantized model to {output_dir}")
        print(f"Saving quantized model to {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Quantization completed successfully. Quantized model saved to {output_dir}")
        print(f"Quantization completed successfully. Quantized model saved to {output_dir}")
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        print(f"Quantization failed: {str(e)}")
        raise

def validate_quant_config(quant_config: dict) -> None:
    """
    Validate the quantization configuration.

    Args:
        quant_config (dict): Configuration parameters for quantization.

    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    if not isinstance(quant_config.get("zero_point"), bool):
        raise ValueError("zero_point must be a boolean value")
    
    if not isinstance(quant_config.get("q_group_size"), int) or quant_config["q_group_size"] <= 0:
        raise ValueError("q_group_size must be a positive integer")
    
    if not isinstance(quant_config.get("w_bit"), int) or quant_config["w_bit"] not in [2, 4, 8]:
        raise ValueError("w_bit must be 2, 4, or 8")
    
    if quant_config.get("version") not in ["GEMM", "GEMV"]:
        raise ValueError("version must be either 'GEMM' or 'GEMV'")

def get_quantized_model_size(model_path: str) -> int:
    """
    Get the total size of all quantized model files in bytes.
    """
    total_size = 0
    for root, _, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    logger.info(f"Total quantized model size: {total_size / (1024 * 1024):.2f} MB")
    return total_size

def validate_quantized_model(output_dir: str) -> bool:
    """
    Validate the quantized model by loading it and performing a simple inference.

    Args:
        output_dir (str): Directory containing the quantized model.

    Returns:
        bool: True if validation is successful, False otherwise.
    """
    try:
        logger.info(f"Validating quantized model in {output_dir}")
        print(f"Validating quantized model in {output_dir}")

        # Load the quantized model and tokenizer
        model = AutoAWQForCausalLM.from_quantized(output_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

        # Prepare a sample input
        sample_text = "Hello, how are you?"
        inputs = tokenizer(sample_text, return_tensors="pt")

        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        # Move outputs back to CPU for decoding
        outputs = outputs.cpu()

        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Generated text: {generated_text}")
        print(f"Generated text: {generated_text}")

        logger.info("Quantized model validation successful")
        print("Quantized model validation successful")
        return True

    except Exception as e:
        logger.error(f"Quantized model validation failed: {str(e)}")
        print(f"Quantized model validation failed: {str(e)}")
        return False
