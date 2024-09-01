# app/quantization.py

import subprocess
import os
import logging
from app.config import Config

def run_quantization(model_path: str, quant_config: dict, logger: logging.Logger) -> None:
    """
    Run the quantization process on the given model using the specified configuration.

    Args:
        model_path (str): Path to the model file that needs to be quantized.
        quant_config (dict): Configuration parameters required for quantization.
        logger (logging.Logger): Logger instance for logging the quantization process.

    Raises:
        RuntimeError: If the quantization process fails.
    """
    try:
        # Construct the quantization command using quantization parameters
        quantization_command = [
            "python", Config.QUANT_SCRIPT,
            "--model_path", model_path,
            "--quant_path", os.path.join(Config.DATA_DIR, f"{os.path.basename(model_path)}-AWQ"),
            "--zero_point", str(quant_config.get("zero_point", True)),
            "--q_group_size", str(quant_config.get("q_group_size", 128)),
            "--w_bit", str(quant_config.get("w_bit", 4)),
            "--version", quant_config.get("version", "GEMM")
        ]

        # Log the constructed command
        logger.info(f"Running quantization command: {' '.join(quantization_command)}")

        # Execute the command and capture output
        result = subprocess.run(quantization_command, capture_output=True, text=True, check=True)

        logger.info("Model quantization completed successfully.")
        logger.debug(f"Quantization output: {result.stdout}")

        # Check if the quantized model file exists
        expected_output_path = os.path.join(Config.DATA_DIR, f"{os.path.basename(model_path)}-AWQ", "pytorch_model.bin")
        if not os.path.exists(expected_output_path):
            raise FileNotFoundError(f"Expected quantized model file not found at {expected_output_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Quantization failed: {e.stderr}")
        raise RuntimeError(f"Quantization command failed with code {e.returncode}")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
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
