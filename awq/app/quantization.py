# app/quantization.py

import subprocess
import os
import logging
from typing import Dict, Any
from app.config import Config
from app.converter import load_safetensors_model, save_safetensors_model

logger = logging.getLogger(__name__)

def run_quantization(model_path: str, quant_config: dict) -> None:
    """
    Run the quantization process on the given model using the specified configuration.

    Args:
        model_path (str): Path to the directory containing model files.
        quant_config (dict): Configuration parameters required for quantization.

    Raises:
        RuntimeError: If the quantization process fails.
    """
    try:
        validate_quant_config(quant_config)

        # Load the model
        logger.info(f"Loading model from {model_path}")
        model_state_dict = load_safetensors_model(model_path)

        # Prepare quantization command
        quant_script_path = os.path.join(Config.APP_DIR, Config.QUANT_SCRIPT)
        output_dir = os.path.join(Config.DATA_DIR, f"{os.path.basename(model_path)}-AWQ")
        os.makedirs(output_dir, exist_ok=True)

        quantization_command = [
            "python", quant_script_path,
            "--model_path", model_path,
            "--quant_path", output_dir,
            "--zero_point", str(quant_config.get("zero_point", True)),
            "--q_group_size", str(quant_config.get("q_group_size", 128)),
            "--w_bit", str(quant_config.get("w_bit", 4)),
            "--version", quant_config.get("version", "GEMM")
        ]

        logger.info(f"Running quantization command: {' '.join(quantization_command)}")

        # Execute the quantization command
        result = subprocess.run(quantization_command, capture_output=True, text=True, check=True)
        logger.info("Model quantization completed successfully.")
        logger.debug(f"Quantization output: {result.stdout}")

        # Load the quantized model
        quantized_model = load_safetensors_model(output_dir)

        # Save the quantized model, potentially splitting into shards
        save_safetensors_model(quantized_model, os.path.join(output_dir, "model.safetensors"))

        logger.info(f"Quantized model saved to {output_dir}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Quantization failed: {e.stderr}")
        raise RuntimeError(f"Quantization command failed with code {e.returncode}")
    except Exception as e:
        logger.exception(f"Unexpected error during quantization: {str(e)}")
        raise RuntimeError(f"Failed to run quantization: {str(e)}")

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
