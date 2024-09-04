# app/quantization.py

import os
import logging
from typing import Dict, Any
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)

logger.info(f"AutoAWQ version: {awq.__version__}")
print(f"AutoAWQ version: {awq.__version__}")

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

        # Print information about the model files
        logger.info(f"Model files in {model_path}:")
        for item in os.listdir(model_path):
            file_path = os.path.join(model_path, item)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logger.info(f"- {item}: {file_size:.2f} MB")
            print(f"- {item}: {file_size:.2f} MB")

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device("cuda")
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            logger.info(f"Using CUDA. Available GPU memory: {available_memory / 1e9:.2f} GB")
            print(f"Using CUDA. Available GPU memory: {available_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA is not available. Using CPU for quantization. This will be significantly slower.")
            print("WARNING: CUDA is not available. Using CPU for quantization. This will be significantly slower.")

        # Load model and tokenizer
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if cuda_available else torch.float32,
                device_map="auto" if cuda_available else None
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory error. The model is too large for your GPU.")
                print("CUDA out of memory error. The model is too large for your GPU.")
                raise
            else:
                raise

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Quantize
        logger.info("Performing AWQ quantization")
        print("Performing AWQ quantization")
        
        # Check if the model has the quantize method
        if hasattr(model, 'quantize'):
            model.quantize(tokenizer, quant_config=quant_config)
        else:
            logger.error("The loaded model does not support the 'quantize' method. It may not be compatible with AWQ quantization.")
            print("The loaded model does not support the 'quantize' method. It may not be compatible with AWQ quantization.")
            raise AttributeError("Model does not support 'quantize' method")

        # Save quantized model
        logger.info(f"Saving quantized model to {output_dir}")
        print(f"Saving quantized model to {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Quantization completed successfully. Quantized model saved to {output_dir}")
        print(f"Quantization completed successfully. Quantized model saved to {output_dir}")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            error_msg = (
                "CUDA out of memory error. The model is too large for your GPU. "
                "Try using a smaller model or a GPU with more memory."
            )
        elif "Cannot copy out of meta tensor" in str(e):
            error_msg = (
                "Meta device error. This may be due to insufficient GPU memory. "
                "Try increasing your GPU memory allocation or using a smaller model."
            )
        elif "Expected all tensors to be on the same device" in str(e):
            error_msg = (
                "Device mismatch error during quantization. "
                "This may be due to insufficient GPU memory. "
                "Try using a smaller model or increasing your GPU memory allocation."
            )
        elif "You can't move a model that has some modules offloaded to cpu or disk" in str(e):
            error_msg = (
                "Model offloading error. The model is too large to fit entirely in GPU memory. "
                "Try using a GPU with more memory or implement a strategy for handling large models."
            )
        else:
            error_msg = str(e)
        logger.error(f"Quantization failed: {error_msg}")
        print(f"Quantization failed: {error_msg}")
        raise RuntimeError(error_msg) from e
    except AttributeError as e:
        if "object has no attribute 'quantize'" in str(e):
            error_msg = (
                "The loaded model does not support AWQ quantization. "
                "Make sure you're using a compatible model and the correct version of AutoAWQ."
            )
            logger.error(f"Quantization failed: {error_msg}")
            print(f"Quantization failed: {error_msg}")
        else:
            logger.error(f"Quantization failed: {str(e)}")
            print(f"Quantization failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        print(f"Quantization failed: {str(e)}")
        logger.exception("Detailed traceback for quantization:")
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

        # Remove token_type_ids if present
        inputs.pop('token_type_ids', None)

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
