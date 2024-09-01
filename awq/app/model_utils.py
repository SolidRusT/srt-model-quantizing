# app/model_utils.py

import os
import logging
import subprocess
from app.config import Config

# Setup logging
logger = logging.getLogger(__name__)

# Add a simple cache for downloaded models
model_cache = {}

def download_model(author: str, model: str) -> str:
    """
    Download the model from a repository, and return the local path.
    Uses a simple cache to avoid re-downloading.
    """
    cache_key = f"{author}/{model}"
    if cache_key in model_cache:
        logger.info(f"Using cached model for {cache_key}")
        return model_cache[cache_key]

    model_url = f"https://huggingface.co/{author}/{model}/resolve/main/model.pth"
    local_path = os.path.join(Config.DATA_DIR, f"{author}/{model}-AWQ", "model.pth")

    if not os.path.exists(local_path):
        logger.info(f"Downloading model from {model_url} to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        subprocess.run(["curl", "-L", model_url, "-o", local_path], check=True)
    else:
        logger.info(f"Model already exists locally at {local_path}")

    model_cache[cache_key] = local_path
    return local_path

# Add support for new models here
new_models = [
    {"author": "new_author_1", "model": "new_model_1"},
    {"author": "new_author_2", "model": "new_model_2"},
]

for model in new_models:
    download_model(model["author"], model["model"])


def check_pytorch_files(model_path: str) -> bool:
    """
    Verify if the specified model path contains valid PyTorch files.
    """
    if os.path.isfile(model_path):
        file_ext = os.path.splitext(model_path)[-1]
        if file_ext in {'.pth', '.bin', '.pt'}:
            logger.info(f"Model file {model_path} is a valid PyTorch file")
            return True

    logger.error(f"Invalid PyTorch file at {model_path}")
    return False
