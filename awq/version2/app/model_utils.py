# app/model_utils.py

import os
import logging
import subprocess
from app.config import Config

# Setup logging
logger = logging.getLogger(__name__)

def setup_environment(author: str, model: str):
    """
    Set up the environment specific to the author and model.
    """
    base_path = os.path.join(Config.DATA_DIR, f"{author}/{model}-AWQ")
    os.makedirs(base_path, exist_ok=True)
    logger.info(f"Environment set up at: {base_path}")


def download_model(author: str, model: str) -> str:
    """
    Download the model from a repository, and return the local path.
    """
    model_url = f"https://huggingface.co/{author}/{model}/resolve/main/model.pth"
    local_path = os.path.join(Config.DATA_DIR, f"{author}/{model}-AWQ", "model.pth")

    if not os.path.exists(local_path):
        logger.info(f"Downloading model from {model_url} to {local_path}")
        # Using curl for demonstration purposes
        subprocess.run(["curl", "-L", model_url, "-o", local_path], check=True)
    else:
        logger.info(f"Model already exists locally at {local_path}")

    return local_path


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
