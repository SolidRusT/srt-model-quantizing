# app/quantization.py

import os
import logging
from typing import Dict, Any
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)

# At the beginning of the file, add:
import awq
logger.info(f"AutoAWQ version: {awq.__version__}")
print(f"AutoAWQ version: {awq.__version__}")

def run_quantization(model_path: str, quant_config: Dict[str, Any], output_dir: str) -> None:
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
        
        # Test model loading
        if not test_model_loading(model_path):
            raise ValueError("Failed to load model for testing. Aborting quantization.")
        
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
        logger.info(f"Loading model from {model_path}")
        print(f"Loading model from {model_path}")
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None
        )
        logger.info("Model loaded successfully")
        print("Model loaded successfully")

        logger.info(f"Loading tokenizer from {model_path}")
        print(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
        print("Tokenizer loaded successfully")

        # Quantize
        logger.info("Performing AWQ quantization")
        print("Performing AWQ quantization")
        
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Available methods: {dir(model)}")
        
        if hasattr(model, 'quantize'):
            logger.info(f"Quantization config: {quant_config}")
            print(f"Quantization config: {quant_config}")
            quantized_model = model.quantize(tokenizer, quant_config=quant_config)
            if quantized_model is None:
                raise ValueError("Quantization failed: model.quantize returned None")
            logger.info("Quantization completed successfully")
            print("Quantization completed successfully")
        else:
            logger.error("The loaded model does not support the 'quantize' method. It may not be compatible with AWQ quantization.")
            print("The loaded model does not support the 'quantize' method. It may not be compatible with AWQ quantization.")
            raise AttributeError("Model does not support 'quantize' method")

        # Save quantized model
        logger.info(f"Saving quantized model to {output_dir}")
        print(f"Saving quantized model to {output_dir}")
        quantized_model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Quantization completed successfully. Quantized model saved to {output_dir}")
        print(f"Quantization completed successfully. Quantized model saved to {output_dir}")
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

def test_model_loading(model_path: str):
    try:
        logger.info(f"Testing model loading from {model_path}")
        print(f"Testing model loading from {model_path}")
        
        # Check if the model path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            print(f"Model path does not exist: {model_path}")
            return False
        
        # List the contents of the model directory
        logger.info(f"Contents of {model_path}:")
        for item in os.listdir(model_path):
            logger.info(f"- {item}")
            print(f"- {item}")
        
        # Check for the existence of necessary files
        required_files = ['config.json']
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                logger.error(f"Required file {file} not found in {model_path}")
                print(f"Required file {file} not found in {model_path}")
                return False
        
        # Check for either tokenizer.json or tokenizer.model
        if not (os.path.exists(os.path.join(model_path, 'tokenizer.json')) or 
                os.path.exists(os.path.join(model_path, 'tokenizer.model'))):
            logger.error(f"No tokenizer file (tokenizer.json or tokenizer.model) found in {model_path}")
            print(f"No tokenizer file (tokenizer.json or tokenizer.model) found in {model_path}")
            return False
        
        # Check for model weights file
        weight_files = [
            'model.safetensors',
            'model.safetensors.index.json',
            'pytorch_model.bin',
            'pytorch_model.bin.index.json'
        ]
        found_weights = False
        for weight_file in weight_files:
            if os.path.exists(os.path.join(model_path, weight_file)):
                logger.info(f"Found weight file: {weight_file}")
                print(f"Found weight file: {weight_file}")
                found_weights = True
                break
        if not found_weights:
            logger.error(f"No valid model weights found in {model_path}")
            print(f"No valid model weights found in {model_path}")
            return False
        
        # Try to load the config
        try:
            config = AutoConfig.from_pretrained(model_path)
            logger.info(f"Successfully loaded config: {config}")
            print(f"Successfully loaded config: {config}")
        except Exception as config_error:
            logger.error(f"Failed to load config: {str(config_error)}")
            print(f"Failed to load config: {str(config_error)}")
            return False
        
        # Try to load the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info("Successfully loaded tokenizer")
            print("Successfully loaded tokenizer")
        except Exception as tokenizer_error:
            logger.error(f"Failed to load tokenizer: {str(tokenizer_error)}")
            print(f"Failed to load tokenizer: {str(tokenizer_error)}")
            return False
        
        # Verify model weights are readable (without loading the entire model)
        try:
            if os.path.exists(os.path.join(model_path, 'model.safetensors')):
                from safetensors import safe_open
                with safe_open(os.path.join(model_path, 'model.safetensors'), framework="pt", device="cpu") as f:
                    # Read a small portion of the model to verify it's readable
                    for key in list(f.keys())[:5]:
                        _ = f.get_tensor(key)
                logger.info("Successfully verified model weights (safetensors)")
                print("Successfully verified model weights (safetensors)")
            elif os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
                # Verify a few keys in the state dict
                for key in list(state_dict.keys())[:5]:
                    _ = state_dict[key]
                logger.info("Successfully verified model weights (PyTorch)")
                print("Successfully verified model weights (PyTorch)")
            else:
                logger.info("Skipping weight verification for sharded model")
                print("Skipping weight verification for sharded model")
            
            return True
        except Exception as model_error:
            logger.error(f"Failed to verify model weights: {str(model_error)}")
            print(f"Failed to verify model weights: {str(model_error)}")
            logger.exception("Detailed traceback for model weight verification:")
            return False
    except Exception as e:
        logger.error(f"Failed to load model for testing: {str(e)}")
        print(f"Failed to load model for testing: {str(e)}")
        logger.exception("Detailed traceback:")
        return False
