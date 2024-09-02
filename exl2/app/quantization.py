import os
import logging
from typing import Dict, Any
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config

logger = logging.getLogger(__name__)

def run_quantization(model_path: str, quant_config: Dict[str, Any], output_dir: str) -> None:
    """
    Run the quantization process on the given model using Exllama2.

    Args:
        model_path (str): Path to the model directory.
        quant_config (Dict[str, Any]): Configuration for quantization.
        output_dir (str): Directory to save the quantized model.
    """
    # Note: The actual quantization is now handled in main.py using convert_model
    # This function is kept for potential future use or additional processing
    logger.info(f"Quantization for model at {model_path} completed. Output saved to {output_dir}")
    print(f"Quantization for model at {model_path} completed. Output saved to {output_dir}")

def validate_quantized_model(model_path: str) -> bool:
    """
    Validate the quantized Exllama2 model.

    Args:
        model_path (str): Path to the quantized model directory.

    Returns:
        bool: True if the model is valid, False otherwise.
    """
    try:
        logger.info(f"Validating Exllama2 quantized model at {model_path}")
        print(f"Validating Exllama2 quantized model at {model_path}")
        
        # Load the model configuration
        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()

        # Try to load the model
        model = ExLlamaV2(config)
        model.load()

        # Perform a simple inference test
        tokenizer = model.tokenizer
        input_text = "Once upon a time,"
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.7)
        
        output_text = tokenizer.decode(output_ids[0])
        
        logger.info(f"Model validation successful. Sample output: {output_text}")
        print(f"Model validation successful. Sample output: {output_text}")
        
        return True
    except Exception as e:
        logger.error(f"Exllama2 model validation failed: {str(e)}")
        print(f"Exllama2 model validation failed: {str(e)}")
        return False