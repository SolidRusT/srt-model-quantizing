# app/model_utils.py

import os
import logging
import json
from typing import Dict, List
import requests
from tqdm import tqdm
from app.config import Config
from safetensors.torch import save_file, load_file
import torch

# Setup logging
logger = logging.getLogger(__name__)

# Add a simple cache for downloaded models
model_cache: Dict[str, str] = {}

def download_model(author: str, model: str) -> str:
    """
    Download the model from a repository, and return the local path.
    """
    cache_key = f"{author}/{model}"
    if cache_key in model_cache:
        logger.info(f"Using cached model for {cache_key}")
        return model_cache[cache_key]

    model_dir = os.path.join(Config.DATA_DIR, f"{author}/{model}")
    os.makedirs(model_dir, exist_ok=True)

    # Download model.safetensors.index.json first
    index_url = f"https://huggingface.co/{author}/{model}/resolve/main/model.safetensors.index.json"
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    download_file(index_url, index_path)

    # Parse the index file to get all model parts
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    model_files = index_data.get('weight_map', {}).values()
    model_files = list(set(model_files))  # Remove duplicates

    # Download all model parts
    for file in model_files:
        file_url = f"https://huggingface.co/{author}/{model}/resolve/main/{file}"
        file_path = os.path.join(model_dir, file)
        download_file(file_url, file_path)

    # Download other necessary files
    other_files = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    for file in other_files:
        file_url = f"https://huggingface.co/{author}/{model}/resolve/main/{file}"
        file_path = os.path.join(model_dir, file)
        download_file(file_url, file_path)

    model_cache[cache_key] = model_dir
    return model_dir

def download_file(url: str, local_path: str) -> None:
    """
    Download a file with a progress bar.
    """
    if os.path.exists(local_path):
        logger.info(f"File already exists: {local_path}")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(local_path, 'wb') as file, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

# Add support for new models here
new_models = [
    {"author": "new_author_1", "model": "new_model_1"},
    {"author": "new_author_2", "model": "new_model_2"},
]

for model in new_models:
    download_model(model["author"], model["model"])


def check_model_files(model_path: str) -> bool:
    """
    Verify if the specified model path contains valid model files.
    """
    required_files = ['config.json', 'tokenizer.json']
    model_files = os.listdir(model_path)
    
    for file in required_files:
        if file not in model_files:
            logger.error(f"Required file {file} not found in {model_path}")
            return False
    
    safetensors_files = [f for f in model_files if f.endswith('.safetensors')]
    pytorch_files = [f for f in model_files if f.endswith('.bin') or f.endswith('.pt')]
    
    if not safetensors_files and not pytorch_files:
        logger.error(f"No valid model files (safetensors or PyTorch) found in {model_path}")
        return False
    
    logger.info(f"Valid model files found in {model_path}")
    return True

def convert_pytorch_to_safetensors(model_path: str) -> None:
    """
    Convert PyTorch model files to safetensors format.
    """
    pytorch_files = [f for f in os.listdir(model_path) if f.endswith('.bin') or f.endswith('.pt')]
    
    for file in pytorch_files:
        pytorch_path = os.path.join(model_path, file)
        safetensors_path = os.path.join(model_path, file.rsplit('.', 1)[0] + '.safetensors')
        
        logger.info(f"Converting {file} to safetensors format")
        state_dict = torch.load(pytorch_path, map_location='cpu')
        save_file(state_dict, safetensors_path)
        
        os.remove(pytorch_path)
        logger.info(f"Removed original PyTorch file: {pytorch_path}")

def get_model_size(model_path: str) -> int:
    """
    Get the total size of all model files in bytes.
    """
    total_size = 0
    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith(('.safetensors', '.bin', '.pt', '.json')):
                total_size += os.path.getsize(os.path.join(root, file))
    
    logger.info(f"Total model size: {total_size / (1024 * 1024):.2f} MB")
    return total_size

def validate_model_checksum(model_path: str, expected_checksum: str) -> bool:
    """
    Validate the model file(s) checksum.
    """
    # Implement checksum validation logic here
    # This is a placeholder and should be replaced with actual checksum calculation
    logger.info(f"Validating checksum for model at {model_path}")
    return True  # Placeholder return value
