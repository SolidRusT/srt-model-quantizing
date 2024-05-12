import argparse
import os
from app.config import Config
from app.model_utils import setup_environment, download_model, check_pytorch_files
from app.quantization import run_quantization
from app.safetensors_conversion import convert_model_to_safetensors
from app.template_manager import process_template
from app.utils import create_logger

# Initialize the logger
logger = create_logger(Config.LOG_FILE)

def main(author: str, model: str):
    # Log the start of the process
    logger.info(f"Starting quantization process for {author}/{model}")

    # Setup environment specific to the author/model
    setup_environment(author, model)
    model_path = download_model(author, model)

    # Process initial processing notice template
    processing_notice_path = os.path.join(Config.MODEL_REPO_DIR, 'processing-notice.txt')
    readme_path = os.path.join(Config.DATA_DIR, model + '-AWQ', 'README.md')
    process_template(processing_notice_path, readme_path, author, model)
    logger.info("Initial processing notice added")

    # Check if the model files are valid PyTorch files
    if check_pytorch_files(model_path):
        # Convert the model to safetensors format if necessary
        converted_path = convert_model_to_safetensors(model_path)

        # Quantize the model
        run_quantization(converted_path, Config.QUANT_CONFIG, logger)

        # Process the final README template
        final_readme_path = os.path.join(Config.MODEL_REPO_DIR, 'initial-readme.txt')
        process_template(final_readme_path, readme_path, author, model)
        logger.info("Model quantization and README update completed successfully")
    else:
        logger.error("Invalid PyTorch files. Quantization process aborted.")

if __name__ == "__main__":
    # Argument parsing for input parameters
    parser = argparse.ArgumentParser(description='Quantize and manage AI models.')
    parser.add_argument('author', help='Author of the model')
    parser.add_argument('model', help='Model identifier')
    args = parser.parse_args()

    # Run the main process with the provided arguments
    main(args.author, args.model)
