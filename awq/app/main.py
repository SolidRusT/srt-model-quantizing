import os
import sys
import argparse
import logging
import re
from huggingface_hub import HfApi, create_repo, Repository, HfFolder, whoami
import shutil

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config
from app.model_utils import (
    authenticate_huggingface,
    download_model,
    check_model_files,
    get_model_size  # Add this import
)
from app.quantization import run_quantization, validate_quantized_model
from app.converter import convert_model_to_safetensors
from app.template_parser import process_template
from app.utils import create_logger

# Initialize the logger
logger = create_logger(Config.LOG_FILE)

def parse_model_string(model_string):
    """Parse the combined author/model string."""
    match = re.match(r'([^/]+)/(.+)', model_string)
    if match:
        return match.group(1), match.group(2)
    else:
        raise ValueError("Invalid model format. Use 'author/model'.")

def main(author: str, model: str, quanter: str = None, expected_checksum: str = None):
    try:
        logger.info(f"Starting quantization process for {author}/{model}")
        print(f"Starting quantization process for {author}/{model}")
        
        # Authenticate with Hugging Face
        token = authenticate_huggingface()
        if not token:
            logger.error("Failed to authenticate with Hugging Face. Please check your token.")
            print("Failed to authenticate with Hugging Face. Please check your token.")
            return

        # Determine quanter if not provided
        if not quanter:
            quanter = Config.QUANTER  # Use the default from Config if not provided via CLI
            logger.info(f"Using default quanter from configuration: {quanter}")
            print(f"Using default quanter from configuration: {quanter}")

        # 1. Download the original model
        model_path = os.path.join(Config.DATA_DIR, f"{author}-{model}")
        if not os.path.exists(model_path):
            try:
                logger.info(f"Downloading model {author}/{model}")
                print(f"Downloading model {author}/{model}")
                model_path = download_model(author, model, expected_checksum)
                logger.info(f"Model downloaded successfully to {model_path}")
                print(f"Model downloaded successfully to {model_path}")
                
                # Add model size information
                try:
                    model_size = get_model_size(model_path)
                    logger.info(f"Model size: {model_size / (1024 * 1024):.2f} MB")
                    print(f"Model size: {model_size / (1024 * 1024):.2f} MB")
                except Exception as e:
                    logger.error(f"Failed to get model size: {str(e)}")
                    print(f"Failed to get model size: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to download model {author}/{model}: {str(e)}")
                print(f"Failed to download model {author}/{model}: {str(e)}")
                return

        # 2. Create or get existing AWQ repo
        awq_repo_name = f"{quanter}/{model}-AWQ"
        try:
            api = HfApi()
            repo_url = api.create_repo(repo_id=awq_repo_name, token=token, exist_ok=True)
            logger.info(f"AWQ repo created or already exists: {repo_url}")
            print(f"AWQ repo created or already exists: {repo_url}")
        except Exception as e:
            logger.error(f"Failed to create AWQ repo: {str(e)}")
            print(f"Failed to create AWQ repo: {str(e)}")
            return

        # 3. Download existing AWQ repo if it exists
        awq_model_path = os.path.join(Config.DATA_DIR, awq_repo_name.split('/')[-1])
        os.makedirs(awq_model_path, exist_ok=True)
        
        # 4. Create and upload processing notice README
        readme_path = os.path.join(awq_model_path, 'README.md')
        try:
            process_template(Config.PROCESSING_NOTICE_PATH, readme_path, author=author, model=model, quanter=quanter)
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=awq_repo_name,
                token=token,
                commit_message="Add processing notice"
            )
            logger.info("Processing notice README created and uploaded")
            print("Processing notice README created and uploaded")
        except Exception as e:
            logger.error(f"Failed to create or upload processing notice: {str(e)}")
            print(f"Failed to create or upload processing notice: {str(e)}")
            # Continue despite this error

        # 5. Check if quantization is needed
        if os.path.exists(os.path.join(awq_model_path, 'model.safetensors')):
            logger.info("AWQ model already exists. Skipping quantization.")
            print("AWQ model already exists. Skipping quantization.")
        else:
            # Check if the model files are valid
            if check_model_files(model_path):
                logger.info("Model files are valid. Proceeding with conversion and quantization.")
                print("Model files are valid. Proceeding with conversion and quantization.")
                
                # Convert the model to a single safetensors file if needed
                logger.info("Checking if model conversion to safetensors format is needed")
                print("Checking if model conversion to safetensors format is needed")
                if os.path.exists(os.path.join(model_path, 'model.safetensors')):
                    logger.info("model.safetensors already exists. Skipping conversion.")
                    print("model.safetensors already exists. Skipping conversion.")
                    converted_path = model_path
                elif os.path.exists(os.path.join(model_path, 'model.safetensors.index.json')):
                    logger.info("Sharded safetensors model found. No conversion needed.")
                    print("Sharded safetensors model found. No conversion needed.")
                    converted_path = model_path
                else:
                    logger.info("Starting model conversion to safetensors format")
                    print("Starting model conversion to safetensors format")
                    converted_path = convert_model_to_safetensors(model_path)
                    logger.info(f"Model converted and saved to {converted_path}")
                    print(f"Model converted and saved to {converted_path}")

                # Add this line to print the model size after conversion
                converted_model_size = get_model_size(converted_path)
                logger.info(f"Converted model size: {converted_model_size / (1024 * 1024):.2f} MB")
                print(f"Converted model size: {converted_model_size / (1024 * 1024):.2f} MB")

                # Check if model weights exist after conversion
                if not (os.path.exists(os.path.join(converted_path, 'model.safetensors')) or 
                        os.path.exists(os.path.join(converted_path, 'model.safetensors.index.json'))):
                    logger.error("No safetensors model weights found after conversion. Aborting quantization.")
                    print("No safetensors model weights found after conversion. Aborting quantization.")
                    return

                # Quantize the model
                logger.info("Starting model quantization")
                print("Starting model quantization")
                try:
                    run_quantization(converted_path, Config.QUANT_CONFIG, awq_model_path)
                except Exception as e:
                    logger.error(f"Quantization failed: {str(e)}")
                    print(f"Quantization failed: {str(e)}")
                    logger.exception("Detailed traceback:")
                    return

        # After quantization
        model_path = os.path.join(awq_model_path, 'model.safetensors')
        sharded_model_index = os.path.join(awq_model_path, 'model.safetensors.index.json')
        
        if os.path.exists(model_path):
            logger.info("AWQ model created successfully.")
            print("AWQ model created successfully.")
        elif os.path.exists(sharded_model_index):
            logger.info("AWQ sharded model created successfully.")
            print("AWQ sharded model created successfully.")
        else:
            logger.error(
                "AWQ model creation failed. Neither 'model.safetensors' nor 'model.safetensors.index.json' found in the output directory."
            )
            print(
                "AWQ model creation failed. Neither 'model.safetensors' nor 'model.safetensors.index.json' found in the output directory."
            )
            return

        # Copy config.json and tokenizer files to AWQ model directory if they don't exist
        for file in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
            src = os.path.join(model_path, file)
            dst = os.path.join(awq_model_path, file)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                logger.info(f"Copied {file} to AWQ model directory")

        # 6. Validate AWQ model
        if validate_quantized_model(awq_model_path):
            # Update README with initial content
            process_template(Config.INITIAL_README_PATH, readme_path, author=author, model=model, quanter=quanter)
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=awq_repo_name,
                token=token,
                commit_message="Update README after successful quantization"
            )
            logger.info("AWQ model validated and README updated")
        else:
            logger.error("AWQ model validation failed")
            print("AWQ model validation failed")
            return

        # 7. Final push of AWQ model to HuggingFace
        api.upload_folder(
            folder_path=awq_model_path,
            repo_id=awq_repo_name,
            token=token,
            commit_message="Upload quantized AWQ model"
        )
        logger.info("AWQ model successfully uploaded to HuggingFace")
        print("AWQ model successfully uploaded to HuggingFace")

    except Exception as e:
        logger.error(f"An error occurred during the quantization process: {str(e)}")
        print(f"An error occurred during the quantization process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model")
    parser.add_argument("model", nargs="?", help="The model in 'author/model' format")
    parser.add_argument("--author", help="The author of the original model on Hugging Face")
    parser.add_argument("--model", dest="model_name", help="The name of the original model on Hugging Face")
    parser.add_argument("--quanter", help="The user or organization to publish the AWQ model under (optional)")
    parser.add_argument("--expected-checksum", help="The expected checksum for the model (optional)")
    args = parser.parse_args()

    if args.model:
        author, model = parse_model_string(args.model)
    elif args.author and args.model_name:
        author, model = args.author, args.model_name
    else:
        parser.error("Either provide 'author/model' as a single argument or use --author and --model separately.")

    main(author, model, args.quanter, args.expected_checksum)
