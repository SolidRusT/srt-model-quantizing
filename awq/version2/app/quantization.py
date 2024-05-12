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
            "python", "scripts/quantize_model.py",
            "--input", model_path,
            "--output", quant_config.get("output_path", "quantized_model"),
            "--bit_width", str(quant_config.get("bit_width", 4)),
            "--fuse_layers", str(quant_config.get("fuse_layers", True))
        ]

        # Add optional parameters based on quant_config
        if "other_option" in quant_config:
            quantization_command.extend(["--other_option", str(quant_config["other_option"])])

        # Log the constructed command
        logger.info(f"Running quantization command: {' '.join(quantization_command)}")

        # Execute the command and capture output
        result = subprocess.run(quantization_command, capture_output=True, text=True)

        # Check if the process ran successfully
        if result.returncode != 0:
            logger.error(f"Quantization failed: {result.stderr}")
            raise RuntimeError(f"Quantization command failed with code {result.returncode}")

        logger.info("Model quantization completed successfully.")
    except Exception as e:
        logger.exception(f"Exception during quantization: {str(e)}")
        raise RuntimeError(f"Failed to run quantization: {str(e)}")
