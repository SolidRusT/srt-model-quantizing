import os
import sys
import argparse
import logging
from tqdm import tqdm
from huggingface_hub import HfApi

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config
from app.model_utils import authenticate_huggingface, download_model, check_model_files
from app.quantization import run_quantization, validate_quantized_model  # Add this import
from app.converter import convert_model_to_safetensors
from app.template_parser import process_template
from app.utils import create_logger

# Initialize the logger
logger = create_logger(Config.LOG_FILE)

def main(author: str, model: str):
    try:
        logger.info(f"Starting quantization process for {author}/{model}")
        print(f"Starting quantization process for {author}/{model}")
        
        # Authenticate with Hugging Face
        authenticate_huggingface()

        model_path = os.path.join(Config.DATA_DIR, f"{author}-{model}")
        output_dir = os.path.join(Config.DATA_DIR, f"{author}-{model}-AWQ")
        readme_path = os.path.join(output_dir, 'README.md')

        if not os.path.exists(model_path):
            try:
                logger.info(f"Downloading model {author}/{model}")
                print(f"Downloading model {author}/{model}")
                model_path = download_model(author, model)
                logger.info(f"Model downloaded successfully to {model_path}")
                print(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model {author}/{model}: {str(e)}")
                print(f"Failed to download model {author}/{model}: {str(e)}")
                return

        # Check if the quantized model already exists
        if os.path.exists(output_dir) and os.path.isfile(os.path.join(output_dir, 'model.safetensors')):
            logger.info(f"Quantized model already exists at {output_dir}")
            print(f"Quantized model already exists at {output_dir}")
            
            # Validate the existing quantized model
            if validate_quantized_model(output_dir):
                logger.info("Existing quantized model validation successful")
                print("Existing quantized model validation successful")
                return
            else:
                logger.warning("Existing quantized model failed validation. Proceeding with requantization.")
                print("Existing quantized model failed validation. Proceeding with requantization.")

        # Process initial processing notice template
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processing initial README template for {author}/{model}")
        process_template(Config.PROCESSING_NOTICE_PATH, readme_path, author, model)
        logger.info("Initial processing notice added to README")

        # Check if the model files are valid PyTorch files
        if check_model_files(model_path):
            logger.info("Model files are valid. Proceeding with conversion and quantization.")
            print("Model files are valid. Proceeding with conversion and quantization.")
            
            # Convert the model to safetensors format if necessary
            converted_path = convert_model_to_safetensors(model_path)

            # Quantize the model
            logger.info("Starting model quantization")
            print("Starting model quantization")
            quantization_success = run_quantization(converted_path, Config.QUANT_CONFIG, output_dir)

            if quantization_success:
                # Process the final README template
                logger.info("Updating README with final information")
                process_template(Config.INITIAL_README_PATH, readme_path, author, model)
                logger.info("Model quantization, validation, and README update completed successfully")
                print("Model quantization, validation, and README update completed successfully")
            else:
                logger.error("Quantization process completed, but validation failed. Please check the logs for details.")
                print("Quantization process completed, but validation failed. Please check the logs for details.")

        else:
            logger.error("Invalid PyTorch files. Quantization process aborted.")
            print("Invalid PyTorch files. Quantization process aborted.")
    except Exception as e:
        logger.error(f"An error occurred during the quantization process: {str(e)}")
        print(f"An error occurred during the quantization process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model")
    parser.add_argument("--author", required=True, help="The author of the model on Hugging Face")
    parser.add_argument("--model", required=True, help="The name of the model on Hugging Face")
    args = parser.parse_args()

    main(args.author, args.model)
