# app/config.py

import os
from dotenv import load_dotenv
from huggingface_hub import whoami, HfApi

load_dotenv()  # This loads the variables from .env file

def get_default_quanter():
    try:
        user_info = whoami()
        return user_info['name']
    except Exception:
        return None

class Config:
    # Application Home Directory
    APP_HOME = os.getenv('APP_HOME', '/tmp')

    # Project root directory (parent of app directory)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Templates directory
    TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')

    # Template file paths
    PROCESSING_NOTICE_PATH = os.path.join(TEMPLATES_DIR, 'processing-notice.txt')
    INITIAL_README_PATH = os.path.join(TEMPLATES_DIR, 'initial-readme.txt')

    # Repository and Data Directory Configurations
    DATA_DIR = os.path.join(APP_HOME, 'data')
    LOG_DIR = os.path.join(APP_HOME, 'logs')

    # Quantization Process Configuration
    QUANTER = os.getenv('QUANTER') or get_default_quanter()

    # Environment Settings
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')  # Default to GPU 0

    # Log File Path
    LOG_FILE = os.path.join(LOG_DIR, 'quant-awq.log')

    # Default Quantization Parameters
    QUANT_CONFIG = {
        'zero_point': True,  # Always set to True (default behavior)
        'q_group_size': 128,  # Group size for quantization, can be adjusted
        'w_bit': 4,  # Bit width for quantization, 4-bit is the standard for AWQ
        'version': "GEMM"  # AWQ version, can be "GEMM" or "GEMV"
    }

    # Authentication Settings
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

    # Ensure required directories are created
    @staticmethod
    def setup_directories():
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)

# Call setup directories to ensure they exist
Config.setup_directories()
