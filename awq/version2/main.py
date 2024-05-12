from model_utils import setup_environment, download_model, check_pytorch_files
from quantization import run_quantization
from template_manager import process_template, update_template

def main():
    parser = argparse.ArgumentParser(description='Quantize and manage AI models.')
    parser.add_argument('author', help='Author of the model')
    parser.add_argument('model', help='Model identifier')
    args = parser.parse_args()

    setup_environment(args.author, args.model)
    model_path = download_model(args.author, args.model)
    
    # Process initial processing notice template
    processing_notice_path = os.path.join(Config.MODEL_REPO_DIR, 'processing-notice.txt')
    readme_path = os.path.join(Config.DATA_DIR, args.model + '-AWQ', 'README.md')
    process_template(processing_notice_path, readme_path, args.author, args.model)

    if check_pytorch_files(model_path):
        # Convert, quantize model, and update README
        converted_path = convert_model_to_safetensors(model_path)
        run_quantization(converted_path)
        # Process final README template
        final_readme_path = os.path.join(Config.MODEL_REPO_DIR, 'initial-readme.txt')
        process_template(final_readme_path, readme_path, args.author, args.model)

if __name__ == "__main__":
    main()
