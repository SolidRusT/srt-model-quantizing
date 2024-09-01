# app/converter.py

import os
import glob
import torch
import logging
from safetensors.torch import save_file as safetensors_save_file
from typing import Dict, Any

logger = logging.getLogger(__name__)

def convert_model_to_safetensors(model_path: str) -> str:
    """
    Convert PyTorch model files to safetensors format if necessary.
    If the model is already in safetensors format, it will be left as is.

    Args:
        model_path (str): Path to the directory containing model files.

    Returns:
        str: Path to the directory containing the converted (or original) model files.
    """
    logger.info(f"Checking model files in {model_path}")
    
    pytorch_files = glob.glob(os.path.join(model_path, '*.bin')) + glob.glob(os.path.join(model_path, '*.pt'))
    
    for file in pytorch_files:
        pytorch_path = file
        safetensors_path = os.path.splitext(pytorch_path)[0] + '.safetensors'
        
        if not os.path.exists(safetensors_path):
            state_dict = torch.load(pytorch_path, map_location='cpu')
            safetensors_save_file(state_dict, safetensors_path)
            os.remove(pytorch_path)
        else:
            logger.info(f"Safetensors file already exists for {pytorch_path}. Skipping conversion.")
    
    return model_path

def convert_file(file_path: str) -> None:
    """
    Convert a single PyTorch model file to safetensors format.

    Args:
        file_path (str): Path to the PyTorch model file.
    """
    try:
        logger.info(f"Converting {file_path} to safetensors format")
        state_dict = torch.load(file_path, map_location='cpu')
        safetensors_path = file_path.rsplit('.', 1)[0] + '.safetensors'
        save_file(state_dict, safetensors_path)
        os.remove(file_path)
        logger.info(f"Converted {file_path} to {safetensors_path}")
    except Exception as e:
        logger.error(f"Error converting {file_path}: {str(e)}")
        raise

def load_safetensors_model(model_path: str) -> Dict[str, Any]:
    """
    Load a model from safetensors files, handling split files if necessary.

    Args:
        model_path (str): Path to the directory containing safetensors model files.

    Returns:
        Dict[str, Any]: The loaded model state dict.
    """
    safetensors_files = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    if len(safetensors_files) == 1:
        logger.info(f"Loading single safetensors file: {safetensors_files[0]}")
        return load_file(os.path.join(model_path, safetensors_files[0]))
    else:
        logger.info(f"Loading {len(safetensors_files)} split safetensors files")
        state_dict = {}
        for file in safetensors_files:
            file_path = os.path.join(model_path, file)
            state_dict.update(load_file(file_path))
        return state_dict

def save_safetensors_model(state_dict: Dict[str, Any], output_path: str, max_shard_size: int = 1024 * 1024 * 1024) -> None:
    """
    Save a model state dict to safetensors format, splitting into multiple files if necessary.

    Args:
        state_dict (Dict[str, Any]): The model state dict to save.
        output_path (str): Path to save the safetensors file(s).
        max_shard_size (int): Maximum size of each shard in bytes. Default is 1GB.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()) <= max_shard_size:
        logger.info(f"Saving model to single file: {output_path}")
        save_file(state_dict, output_path)
    else:
        logger.info(f"Model size exceeds {max_shard_size} bytes. Splitting into multiple files.")
        shard_index = 0
        current_shard = {}
        current_size = 0

        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > max_shard_size:
                shard_path = f"{output_path}.{shard_index:05d}-of-{len(state_dict):05d}.safetensors"
                save_file(current_shard, shard_path)
                logger.info(f"Saved shard {shard_index} to {shard_path}")
                shard_index += 1
                current_shard = {}
                current_size = 0

            current_shard[key] = tensor
            current_size += tensor_size

        if current_shard:
            shard_path = f"{output_path}.{shard_index:05d}-of-{len(state_dict):05d}.safetensors"
            save_file(current_shard, shard_path)
            logger.info(f"Saved final shard {shard_index} to {shard_path}")

        logger.info(f"Model split into {shard_index + 1} shards")

def save_file(tensors: Dict[str, torch.Tensor], filename: str) -> None:
    """
    Save tensors to a file using safetensors format.

    Args:
        tensors (Dict[str, torch.Tensor]): Tensors to save.
        filename (str): Name of the file to save to.
    """
    safetensors_save_file(tensors, filename)
