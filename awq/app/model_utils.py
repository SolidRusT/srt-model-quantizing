# app/model_utils.py

import os
import logging
from typing import Dict, List
from huggingface_hub import login, hf_hub_download, HfApi
from tqdm import tqdm
from app.config import Config
import json
import torch

# Setup logging
logger = logging.getLogger(__name__)

# Add a simple cache for downloaded models
model_cache: Dict[str, str] = {}

def authenticate_huggingface():
    """
    Authenticate with Hugging Face using the access token from environment variables.
    """
    access_token = os.getenv('HF_ACCESS_TOKEN')
    if not access_token:
        logger.warning("HF_ACCESS_TOKEN not found in environment variables. Some models may not be accessible.")
        return

    try:
        login(access_token)
        logger.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {str(e)}")

def download_model(author: str, model: str) -> str:
    """
    Download the model from Hugging Face, prioritizing safetensors format.
    """
    cache_key = f"{author}/{model}"
    if cache_key in model_cache:
        logger.info(f"Using cached model for {cache_key}")
        return model_cache[cache_key]

    model_dir = os.path.join(Config.DATA_DIR, f"{author}/{model}")
    os.makedirs(model_dir, exist_ok=True)

    try:
        authenticate_huggingface()

        # Download common files
        common_files = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
        for file in common_files:
            try:
                download_path = hf_hub_download(repo_id=f"{author}/{model}", filename=file, cache_dir=model_dir)
                logger.info(f"Downloaded {file} to {download_path}")
            except Exception as e:
                logger.warning(f"Failed to download {file}: {str(e)}")

        # Try to download safetensors files first
        try:
            # Check for safetensors index file
            index_file = hf_hub_download(repo_id=f"{author}/{model}", filename="model.safetensors.index.json", cache_dir=model_dir)
            logger.info(f"Found sharded safetensors model. Index file: {index_file}")
            
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            shard_files = list(set(index_data.get('weight_map', {}).values()))
            
            for shard in shard_files:
                shard_file = hf_hub_download(repo_id=f"{author}/{model}", filename=shard, cache_dir=model_dir)
                logger.info(f"Downloaded safetensors shard: {shard_file}")
            
            logger.info("Successfully downloaded all safetensors shards")
            
        except Exception as safetensors_error:
            logger.info("Safetensors files not found. Attempting to download PyTorch files.")
            
            try:
                # Try single PyTorch file
                model_file = hf_hub_download(repo_id=f"{author}/{model}", filename="pytorch_model.bin", cache_dir=model_dir)
                logger.info(f"Downloaded single PyTorch model file: {model_file}")
            except Exception as single_file_error:
                logger.info("Single PyTorch file not found. Attempting to download sharded PyTorch model.")
                try:
                    # Download PyTorch model index file
                    index_file = hf_hub_download(repo_id=f"{author}/{model}", filename="pytorch_model.bin.index.json", cache_dir=model_dir)
                    logger.info(f"Downloaded PyTorch model index file: {index_file}")
                    
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                    shard_files = list(set(index_data.get('weight_map', {}).values()))
                    
                    for shard in shard_files:
                        shard_file = hf_hub_download(repo_id=f"{author}/{model}", filename=shard, cache_dir=model_dir)
                        logger.info(f"Downloaded PyTorch shard: {shard_file}")
                    
                    logger.info("Successfully downloaded all PyTorch shards")
                except Exception as pytorch_shard_error:
                    logger.error(f"Failed to download any model files: {str(pytorch_shard_error)}")
                    raise

        model_cache[cache_key] = model_dir
        return model_dir
    except Exception as e:
        logger.error(f"An error occurred while downloading model {author}/{model}: {str(e)}")
        raise

def check_model_files(model_path: str) -> bool:
    """
    Verify if the specified model path contains valid model files.
    """
    required_files = ['config.json', 'tokenizer.json']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            logger.error(f"Required file {file} not found in {model_path}")
            return False
    
    # Check for safetensors files first, then fall back to PyTorch files
    if os.path.exists(os.path.join(model_path, 'model.safetensors.index.json')):
        logger.info(f"Found sharded safetensors model")
        return True
    elif any(f.endswith('.safetensors') for f in os.listdir(model_path)):
        logger.info(f"Found safetensors model file(s)")
        return True
    elif os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
        logger.info(f"Found single PyTorch model file")
        return True
    elif os.path.exists(os.path.join(model_path, 'pytorch_model.bin.index.json')):
        logger.info(f"Found sharded PyTorch model")
        return True
    else:
        logger.error(f"No valid model weights found in {model_path}")
        return False

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
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
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
