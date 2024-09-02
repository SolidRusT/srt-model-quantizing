import os
import logging
from typing import Dict
from huggingface_hub import login, snapshot_download, HfFolder
from app.config import Config
import torch
import hashlib
from safetensors.torch import save_file as safetensors_save_file

# Setup logging
logger = logging.getLogger(__name__)

# Add a simple cache for downloaded models
model_cache: Dict[str, str] = {}

def authenticate_huggingface():
    """
    Authenticate with Hugging Face using the access token from environment variables.
    """
    token = os.environ.get("HF_ACCESS_TOKEN") or HfFolder.get_token()
    if token:
        login(token)
        return token
    else:
        logger.error("HF_ACCESS_TOKEN not found in environment variables or Hugging Face cache.")
        return None

def download_model(author: str, model: str) -> str:
    """
    Download the model from Hugging Face, handling the new blob structure.
    """
    try:
        logger.info(f"Attempting to download model {author}/{model}")
        model_path = snapshot_download(repo_id=f"{author}/{model}", local_dir=os.path.join(Config.DATA_DIR, f"{author}-{model}"))
        logger.info(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error downloading model {author}/{model}: {str(e)}")
        raise

def check_model_files(model_path: str) -> bool:
    """
    Verify if the specified model path contains valid model files.
    """
    required_files = ['config.json', 'tokenizer.json']
    for file in required_files:
        file_path = find_file(model_path, file)
        if not file_path:
            logger.error(f"Required file {file} not found in {model_path}")
            return False
    
    # Check for either pytorch_model.bin or sharded model files
    if find_file(model_path, 'pytorch_model.bin'):
        logger.info(f"Found single file model: pytorch_model.bin")
    elif find_file(model_path, 'model.safetensors.index.json'):
        logger.info(f"Found sharded safetensors model")
    elif find_file(model_path, 'pytorch_model.bin.index.json'):
        logger.info(f"Found sharded PyTorch model")
    else:
        logger.error(f"No valid model weights found in {model_path}")
        return False
    
    logger.info(f"All required model files found in {model_path}")
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
        safetensors_save_file(state_dict, safetensors_path)
        
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
    logger.info(f"Validating checksum for model at {model_path}")
    
    calculated_checksum = calculate_directory_checksum(model_path)
    
    if calculated_checksum == expected_checksum:
        logger.info("Checksum validation successful")
        return True
    else:
        logger.warning(f"Checksum mismatch. Expected: {expected_checksum}, Calculated: {calculated_checksum}")
        return False

def calculate_directory_checksum(directory: str) -> str:
    """
    Calculate a checksum for all files in a directory.
    """
    checksums = []
    for root, _, files in os.walk(directory):
        for file in sorted(files):  # Sort to ensure consistent order
            file_path = os.path.join(root, file)
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                checksums.append(file_hash)
    
    return hashlib.sha256("".join(checksums).encode()).hexdigest()

def find_file(directory: str, filename: str) -> str:
    """
    Find a file in the directory structure, following symlinks if necessary.
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            file_path = os.path.join(root, filename)
            if os.path.islink(file_path):
                return os.path.realpath(file_path)
            return file_path
    return ""

def calculate_checksum(file_path: str) -> str:
    """
    Calculate the MD5 checksum of a file.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
