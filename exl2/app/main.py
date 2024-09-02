import os
import logging
import argparse
from typing import List
from .config import Config
from .quantization import run_quantization, validate_quantized_model
from huggingface_hub import login, HfApi
from exllamav2.conversion.convert_exl2 import convert_model

logger = logging.getLogger(__name__)

def authenticate_huggingface():
    token = Config.HF_ACCESS_TOKEN
    if token:
        login(token)
        return token
    else:
        logger.error("HF_ACCESS_TOKEN not found in environment variables.")
        return None

def download_model(author: str, model: str) -> str:
    api = HfApi()
    model_id = f"{author}/{model}"
    model_path = os.path.join(Config.DATA_DIR, f"{author}-{model}")
    api.snapshot_download(repo_id=model_id, local_dir=model_path)
    return model_path

def main(author: str, model: str, quanter: str = None):
    try:
        logger.info(f"Starting Exllama2 quantization process for {author}/{model}")
        print(f"Starting Exllama2 quantization process for {author}/{model}")
        
        # Authenticate with Hugging Face
        if not authenticate_huggingface():
            return
        
        # Download the model
        model_path = download_model(author, model)
        
        # Define quantization configurations
        bpw_values: List[str] = Config.BPW_VALUES
        
        for bpw in bpw_values:
            branch = f"exl2_{bpw.replace('.', '_')}"
            
            # Create a new branch for this quantization
            api = HfApi()
            repo_id = f"{quanter}/{model}-exl2-{bpw}"
            api.create_branch(repo_id=repo_id, branch=branch)
            
            # Run quantization
            output_dir = os.path.join(Config.DATA_DIR, f"{author}-{model}-exl2-{bpw}")
            hb = Config.get_head_bits(bpw)
            
            # Use Exllama2's convert_model function
            convert_model(
                model_dir=model_path,
                output_dir=output_dir,
                bits=float(bpw),
                head_bits=hb,
                group_size=Config.QUANT_CONFIG['group_size'],
                act_order=Config.QUANT_CONFIG['act_order']
            )
            
            # Validate quantized model
            if validate_quantized_model(output_dir):
                logger.info(f"Quantized model for BPW {bpw} validated successfully")
                print(f"Quantized model for BPW {bpw} validated successfully")
                
                # Upload to Hugging Face Hub
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload Exllama2 quantized model (BPW: {bpw})",
                    branch=branch
                )
                logger.info(f"Uploaded quantized model for BPW {bpw} to Hugging Face Hub")
                print(f"Uploaded quantized model for BPW {bpw} to Hugging Face Hub")
            else:
                logger.error(f"Quantized model for BPW {bpw} failed validation")
                print(f"Quantized model for BPW {bpw} failed validation")
        
        logger.info("Exllama2 quantization process completed")
        print("Exllama2 quantization process completed")
    except Exception as e:
        logger.error(f"Exllama2 quantization process failed: {str(e)}")
        print(f"Exllama2 quantization process failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exllama2 Quantization")
    parser.add_argument("author", help="The author of the model on Hugging Face Hub")
    parser.add_argument("model", help="The name of the model to quantize")
    parser.add_argument("--quanter", help="Specify a custom quanter name", default=Config.QUANTER)
    
    args = parser.parse_args()
    main(args.author, args.model, args.quanter)