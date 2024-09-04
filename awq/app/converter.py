# app/converter.py

import os
import shutil
import logging
from typing import Dict, Any
from safetensors.torch import save_file as safetensors_save_file, load_file
import torch

logger = logging.getLogger(__name__)

def convert_model_to_safetensors(model_path: str) -> str:
    """
    Convert PyTorch model files to safetensors format or merge sharded safetensors.
    """
    logger.info(f"Converting model at {model_path} to safetensors format")
    
    # Check if model.safetensors already exists
    if os.path.exists(os.path.join(model_path, 'model.safetensors')):
        logger.info("model.safetensors already exists. Skipping conversion.")
        return model_path

    # Check if we have sharded safetensors files
    sharded_safetensors = [f for f in os.listdir(model_path) if f.startswith('model-') and f.endswith('.safetensors')]
    
    if sharded_safetensors:
        logger.info("Found sharded safetensors files. Merging them.")
        try:
            merge_sharded_safetensors(model_path, sharded_safetensors)
        except Exception as e:
            logger.error(f"Error merging sharded safetensors: {str(e)}")
            raise
    else:
        # Check for PyTorch bin files
        pytorch_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
        
        if pytorch_files:
            logger.info("Found PyTorch bin files. Converting to safetensors.")
            try:
                convert_pytorch_to_safetensors(model_path, pytorch_files)
            except Exception as e:
                logger.error(f"Error converting PyTorch files to safetensors: {str(e)}")
                raise
        else:
            logger.error("No PyTorch bin files or safetensors files found.")
            raise FileNotFoundError("No model files found to convert")
    
    # Verify the converted model
    if not verify_safetensors_model(os.path.join(model_path, 'model.safetensors')):
        raise ValueError("Converted safetensors model verification failed")
    
    # Update or remove pytorch_model.bin.index.json
    index_file = os.path.join(model_path, 'pytorch_model.bin.index.json')
    if os.path.exists(index_file):
        os.remove(index_file)
        logger.info("Removed pytorch_model.bin.index.json")
    
    return model_path

def merge_sharded_safetensors(model_path: str, sharded_files: list):
    merged_state_dict = {}
    total_size = 0
    for shard in sorted(sharded_files):
        shard_path = os.path.join(model_path, shard)
        shard_dict = load_file(shard_path)
        merged_state_dict.update(shard_dict)
        total_size += os.path.getsize(shard_path)
        logger.info(f"Loaded shard: {shard}, size: {os.path.getsize(shard_path) / 1024 / 1024:.2f} MB")
    
    output_path = os.path.join(model_path, 'model.safetensors')
    safetensors_save_file(merged_state_dict, output_path)
    logger.info(f"Merged safetensors saved to {output_path}")
    logger.info(f"Total size of merged model: {total_size / 1024 / 1024:.2f} MB")
    
    # Remove sharded files
    for shard in sharded_files:
        os.remove(os.path.join(model_path, shard))
        logger.info(f"Removed shard: {shard}")

def convert_pytorch_to_safetensors(model_path: str, pytorch_files: list):
    merged_state_dict = {}
    total_size = 0
    for pytorch_file in pytorch_files:
        pytorch_path = os.path.join(model_path, pytorch_file)
        logger.info(f"Loading PyTorch file: {pytorch_file}")
        state_dict = torch.load(pytorch_path, map_location='cpu')
        merged_state_dict.update(state_dict)
        total_size += os.path.getsize(pytorch_path)
        logger.info(f"Loaded {pytorch_file}, size: {os.path.getsize(pytorch_path) / 1024 / 1024:.2f} MB")
    
    output_path = os.path.join(model_path, 'model.safetensors')
    logger.info("Saving merged state dict to safetensors format...")
    safetensors_save_file(merged_state_dict, output_path)
    logger.info(f"Converted safetensors saved to {output_path}")
    logger.info(f"Total size of converted model: {total_size / 1024 / 1024:.2f} MB")
    
    # Backup and remove original PyTorch files
    backup_dir = os.path.join(model_path, 'pytorch_backup')
    os.makedirs(backup_dir, exist_ok=True)
    for pytorch_file in pytorch_files:
        src = os.path.join(model_path, pytorch_file)
        dst = os.path.join(backup_dir, pytorch_file)
        shutil.move(src, dst)
        logger.info(f"Moved original PyTorch file to backup: {pytorch_file}")

def verify_safetensors_model(model_path: str) -> bool:
    try:
        _ = load_file(model_path)
        logger.info("Safetensors model verification successful")
        return True
    except Exception as e:
        logger.error(f"Safetensors model verification failed: {str(e)}")
        return False
