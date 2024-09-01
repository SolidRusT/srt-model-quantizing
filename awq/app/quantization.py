# app/quantization.py

import subprocess
import os
import logging
from typing import Dict, Any
from tqdm import tqdm
from app.config import Config
from app.converter import load_safetensors_model, save_safetensors_model

logger = logging.getLogger(__name__)

def run_quantization(model_path: str, quant_config: Dict[str, Any], output_dir: str) -> None:
    """
    Run the quantization process on the given model.

    Args:
        model_path (str): Path to the model directory.
        quant_config (Dict[str, Any]): Configuration for quantization.
        output_dir (str): Directory to save the quantized model.
    """
    try:
        logger.info(f"Starting quantization for model at {model_path}")
        print(f"Starting quantization for model at {model_path}")
        
        # Load the model
        model = load_safetensors_model(model_path)
        
        # Perform quantization
        quantized_model = quantize_model(model, quant_config)
        
        # Save the quantized model
        save_quantized_model(quantized_model, output_dir)
        
        logger.info(f"Quantization completed successfully. Quantized model saved to {output_dir}")
        print(f"Quantization completed successfully. Quantized model saved to {output_dir}")
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        print(f"Quantization failed: {str(e)}")
        raise

def quantize_model(model: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quantize the model using the specified configuration.

    Args:
        model (Dict[str, Any]): The model to quantize.
        config (Dict[str, Any]): Quantization configuration.

    Returns:
        Dict[str, Any]: The quantized model.
    """
    logger.info("Performing quantization")
    print("Performing quantization")
    
    # Placeholder for actual quantization logic
    # Replace this with your actual quantization implementation
    quantized_model = {}
    for key, tensor in tqdm(model.items(), desc="Quantizing tensors"):
        # Simulate quantization (replace with actual quantization logic)
        quantized_model[key] = tensor * 0.5
    
    return quantized_model

def save_quantized_model(quantized_model: Dict[str, Any], output_dir: str) -> None:
    """
    Save the quantized model to the specified directory.

    Args:
        quantized_model (Dict[str, Any]): The quantized model to save.
        output_dir (str): Directory to save the quantized model.
    """
    logger.info(f"Saving quantized model to {output_dir}")
    print(f"Saving quantized model to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model in chunks
    chunk_size = 100  # Adjust this value based on your needs
    total_keys = len(quantized_model)
    
    for i in tqdm(range(0, total_keys, chunk_size), desc="Saving quantized model"):
        chunk = {k: quantized_model[k] for k in list(quantized_model.keys())[i:i+chunk_size]}
        chunk_file = os.path.join(output_dir, f"quantized_model_{i//chunk_size:05d}.safetensors")
        save_safetensors_model(chunk, chunk_file)

    logger.info(f"Quantized model saved successfully to {output_dir}")
    print(f"Quantized model saved successfully to {output_dir}")

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
            if file.endswith('.safetensors'):
                total_size += os.path.getsize(os.path.join(root, file))
    
    logger.info(f"Total quantized model size: {total_size / (1024 * 1024):.2f} MB")
    return total_size
