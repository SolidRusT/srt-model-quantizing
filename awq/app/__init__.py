# app/__init__.py

# Import main functionalities to make them available when importing the app package
from .main import main as run_main
from .model_utils import setup_environment, download_model
from .quantization import run_quantization
from .converter import convert_model_to_safetensors
from .template_manager import process_template

# You can also include versioning information if needed
__version__ = "1.0.0"
