# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env file

class Config:
    # Application Home Directory
    APP_HOME = os.getenv('APP_HOME', '/opt/quant-awq')

    # Repository and Data Directory Configurations
    MODEL_REPO_DIR = os.path.join(APP_HOME, 'repos', 'srt-model-quantizing', 'awq')
    DATA_DIR = os.path.join(APP_HOME, 'data')
    LOG_DIR = os.path.join(APP_HOME, 'logs')

    # Quantization Process Configuration
    QUANTER = 'solidrust'
    QUANT_SCRIPT = 'run-quant-awq.py'

    # Environment Settings
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')  # Default to GPU 0

    # Log File Path
    LOG_FILE = os.path.join(LOG_DIR, 'quant-awq.log')

    # Default Quantization Parameters
    QUANT_CONFIG = {
        'zero_point': True,
        'q_group_size': 128,
        'w_bit': 4,
        'version': 'GEMM'
    }

    # Authentication Settings
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

    # Ensure required directories are created
    @staticmethod
    def setup_directories():
        os.makedirs(Config.MODEL_REPO_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)

# Call setup directories to ensure they exist
Config.setup_directories()
