# app/converter.py

import os
import shutil
import json
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
    
    # Check if model.safetensors already exists
    if os.path.exists(os.path.join(model_path, 'model.safetensors')):
        logger.info("model.safetensors already exists. Skipping conversion.")
        return model_path

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
            logger.error("No PyTorch bin files or safetensors files found.")
            raise FileNotFoundError("No model files found to convert")
    
    # Verify the converted model
    if not verify_safetensors_model(os.path.join(model_path, 'model.safetensors')):
        raise ValueError("Converted safetensors model verification failed")
    
    # Update or remove pytorch_model.bin.index.json
    index_file = os.path.join(model_path, 'pytorch_model.bin.index.json')
    if os.path.exists(index_file):
        update_index_file(model_path, index_file)
    
    return model_path

def convert_pytorch_to_safetensors(model_path: str, pytorch_files: list):
    for pytorch_file in pytorch_files:
        pt_filename = os.path.join(model_path, pytorch_file)
        sf_filename = os.path.join(model_path, pytorch_file.replace('pytorch_model', 'model').replace('.bin', '.safetensors'))
        
        logger.info(f"Converting {pytorch_file} to safetensors format")
        loaded = torch.load(pt_filename, map_location="cpu")
        loaded = loaded.get("state_dict", loaded)
        shared = shared_pointers(loaded)

        for shared_weights in shared:
            for name in shared_weights[1:]:
                loaded.pop(name)

        loaded = {k: v.contiguous().half() for k, v in loaded.items()}

        safetensors_save_file(loaded, sf_filename, metadata={"format": "pt"})
        check_file_size(sf_filename, pt_filename)

        # Verify conversion
        reloaded = load_file(sf_filename)
        for k, v in loaded.items():
            if not torch.equal(v, reloaded[k]):
                raise RuntimeError(f"Mismatch in tensors for key {k}.")

        logger.info(f"Successfully converted {pytorch_file} to {os.path.basename(sf_filename)}")

        # Optionally, remove the original PyTorch file
        os.remove(pt_filename)
        logger.info(f"Removed original PyTorch file: {pytorch_file}")

def merge_sharded_safetensors(model_path: str, sharded_files: list):
    merged_state_dict = {}
    total_size = 0
    for shard in tqdm(sorted(sharded_files), desc="Merging shards"):
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

def verify_safetensors_model(model_path: str) -> bool:
    try:
        _ = load_file(model_path)
        logger.info("Safetensors model verification successful")
        return True
    except Exception as e:
        logger.error(f"Safetensors model verification failed: {str(e)}")
        return False

def update_index_file(model_path: str, index_file: str):
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    new_map = {k: v.replace('pytorch_model', 'model').replace('.bin', '.safetensors') 
               for k, v in index_data["weight_map"].items()}
    
    new_index_file = os.path.join(model_path, 'model.safetensors.index.json')
    with open(new_index_file, 'w') as f:
        json.dump({**index_data, "weight_map": new_map}, f, indent=4)
    
    os.remove(index_file)
    logger.info(f"Updated index file: {new_index_file}")
