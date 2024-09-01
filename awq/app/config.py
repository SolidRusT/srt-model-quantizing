# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env file

class Config:
    # Application Home Directory
    APP_HOME = os.getenv('APP_HOME', '/opt/quant-awq')

    # Project root directory (parent of app directory)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Repository and Data Directory Configurations
    DATA_DIR = os.path.join(APP_HOME, 'data')
    LOG_DIR = os.path.join(APP_HOME, 'logs')

    # Static content files
    PROCESSING_NOTICE_PATH = os.path.join(PROJECT_ROOT, 'processing-notice.txt')
    INITIAL_README_PATH = os.path.join(PROJECT_ROOT, 'initial-readme.txt')

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
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)

# Call setup directories to ensure they exist
Config.setup_directories()
