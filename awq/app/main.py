import os
import sys
import argparse
from huggingface_hub import HfApi

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config
from app.model_utils import authenticate_huggingface, download_model, check_model_files
from app.quantization import run_quantization
from app.converter import convert_model_to_safetensors
from app.template_parser import process_template
from app.utils import create_logger

# Initialize the logger
logger = create_logger(Config.LOG_FILE)

def main(author: str, model: str):
    try:
        # Authenticate with Hugging Face
        authenticate_huggingface()

        # Log the start of the process
        logger.info(f"Starting quantization process for {author}/{model}")
        
        try:
            model_path = download_model(author, model)
        except Exception as e:
            logger.error(f"Failed to download model {author}/{model}: {str(e)}")
            return

        # Process initial processing notice template
        processing_notice_path = os.path.join(Config.MODEL_REPO_DIR, 'processing-notice.txt')
        readme_path = os.path.join(Config.DATA_DIR, model + '-AWQ', 'README.md')
        process_template(processing_notice_path, readme_path, author, model)
        logger.info("Initial processing notice added")

        # Check if the model files are valid PyTorch files
        if check_model_files(model_path):
            # Convert the model to safetensors format if necessary
            converted_path = convert_model_to_safetensors(model_path)

            # Quantize the model
            run_quantization(converted_path, Config.QUANT_CONFIG)

            # Process the final README template
            final_readme_path = os.path.join(Config.MODEL_REPO_DIR, 'initial-readme.txt')
            process_template(final_readme_path, readme_path, author, model)
            logger.info("Model quantization and README update completed successfully")
        else:
            logger.error("Invalid PyTorch files. Quantization process aborted.")
    except Exception as e:
        logger.exception(f"An error occurred during the quantization process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model using AWQ")
    parser.add_argument("--author", required=True, help="Author of the model")
    parser.add_argument("--model", required=True, help="Name of the model")

    try:
        args = parser.parse_args()
    except SystemExit:
        # This exception is raised when --help is called or when no arguments are provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    main(args.author, args.model)
