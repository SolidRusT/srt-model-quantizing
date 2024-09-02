# exl2/app/config.py

import os
from dotenv import load_dotenv
from huggingface_hub import whoami

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
    LOG_FILE = os.path.join(LOG_DIR, 'quant-exl2.log')

    # Default Quantization Parameters
    QUANT_CONFIG = {
        'bpw': 4.0,  # Default bits per weight
        'hb': 6,     # Default head bits
        'group_size': 128,  # Default group size
        'act_order': False  # Default activation order
    }

    # Exllama2 specific configurations
    EXL2_HOME = os.getenv('EXL2_HOME', '/path/to/exllama2')  # Path to Exllama2 repository
    TEMP_DIR = os.path.join(APP_HOME, 'temp', 'exl2')

    # Bits per weight (BPW) configurations
    BPW_VALUES = ["8.0", "6.5", "5.0", "4.5", "4.0", "3.5", "3.0", "2.5", "2.0"]

    # Authentication Settings
    HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

    # Ensure required directories are created
    @staticmethod
    def setup_directories():
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.TEMP_DIR, exist_ok=True)

    # Method to get head bits based on bits per weight
    @staticmethod
    def get_head_bits(bpw):
        return 8 if bpw in ["8.0", "6.5"] else 6