# app/converter.py

import json
import os
import shutil
import logging
from typing import Dict, Any
from safetensors.torch import save_file as safetensors_save_file, load_file
import torch
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)

def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    return [names for names in ptrs.values() if len(names) > 1]

def check_file_size(sf_filename, pt_filename):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        logger.warning(f"File size difference exceeds 1% between {sf_filename} and {pt_filename}")

def convert_model_to_safetensors(model_path: str) -> str:
    """
    Convert PyTorch model files to safetensors format or merge sharded safetensors.
    """
    logger.info(f"Converting model at {model_path} to safetensors format")
    
    # Check if safetensors files already exist
    safetensors_files = [f for f in os.listdir(model_path) if f.startswith('model-') and f.endswith('.safetensors')]
    if safetensors_files:
        logger.info("Safetensors files already exist. Skipping conversion.")
        update_safetensors_index(model_path)
        return model_path

    # Check for PyTorch bin files
    pytorch_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
    
    if pytorch_files:
        logger.info(f"Found {len(pytorch_files)} PyTorch bin files. Converting to safetensors.")
        try:
            convert_pytorch_to_safetensors(model_path, pytorch_files)
        except Exception as e:
            logger.error(f"Error converting PyTorch files to safetensors: {str(e)}")
            raise
    else:
        logger.error("No PyTorch bin files found.")
        raise FileNotFoundError("No model files found to convert")
    
    # Update the index file
    update_safetensors_index(model_path)
    
    return model_path

def convert_pytorch_to_safetensors(model_path: str, pytorch_files: list):
    for i, pytorch_file in enumerate(pytorch_files, start=1):
        pt_filename = os.path.join(model_path, pytorch_file)
        sf_filename = os.path.join(model_path, f"model-{i:05d}-of-{len(pytorch_files):05d}.safetensors")
        
        logger.info(f"Converting {pytorch_file} to {os.path.basename(sf_filename)}")
        loaded = torch.load(pt_filename, map_location="cpu")
        loaded = loaded.get("state_dict", loaded)
        shared = shared_pointers(loaded)

        for shared_weights in shared:
            for name in shared_weights[1:]:
                loaded.pop(name)

        loaded = {k: v.contiguous().half() for k, v in loaded.items()}

        safetensors_save_file(loaded, sf_filename, metadata={"format": "pt"})
        logger.info(f"Successfully converted {pytorch_file} to {os.path.basename(sf_filename)}")

        # Optionally, remove the original PyTorch file
        os.remove(pt_filename)
        logger.info(f"Removed original PyTorch file: {pytorch_file}")

def update_safetensors_index(model_path: str):
    pytorch_index_file = os.path.join(model_path, 'pytorch_model.bin.index.json')
    safetensors_index_file = os.path.join(model_path, 'model.safetensors.index.json')
    
    if not os.path.exists(pytorch_index_file):
        logger.error("PyTorch index file not found. Cannot update safetensors index.")
        return
    
    with open(pytorch_index_file, 'r') as f:
        index_data = json.load(f)
    
    safetensors_files = sorted([f for f in os.listdir(model_path) if f.startswith('model-') and f.endswith('.safetensors')])
    
    new_weight_map = {}
    for key, old_file in index_data['weight_map'].items():
        shard_index = int(old_file.split('-')[-1].split('.')[0]) - 1
        new_file = safetensors_files[shard_index]
        new_weight_map[key] = new_file
    
    new_index_data = {
        'metadata': index_data.get('metadata', {}),
        'weight_map': new_weight_map
    }
    
    with open(safetensors_index_file, 'w') as f:
        json.dump(new_index_data, f, indent=2)
    
    logger.info(f"Updated safetensors index file: {safetensors_index_file}")
    
    # Optionally, remove the old PyTorch index file
    os.remove(pytorch_index_file)
    logger.info(f"Removed old PyTorch index file: {pytorch_index_file}")
