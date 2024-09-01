# Usage Guide for SRT Model Quantizing

This guide provides detailed instructions on how to use the SRT Model Quantizing tool.

## Prerequisites

- Python 3.7 or higher
- Hugging Face account and access token
- Sufficient disk space for model downloads and quantization

## Setting Up

1. Ensure you have set up your Hugging Face token:
   - Set the environment variable: `export HF_ACCESS_TOKEN=your_token_here`
   - Or use the Hugging Face CLI: `huggingface-cli login`

2. Activate your Python virtual environment (if using one).

## Basic Usage

The basic command structure is:

```bash
python app/main.py --author <author> --model <model> [--quanter <quanter>]
```

- `<author>`: The author of the original model on Hugging Face
- `<model>`: The name of the original model on Hugging Face
- `<quanter>` (optional): The user or organization to publish the AWQ model under

If `--quanter` is not provided, the tool will use the username associated with your Hugging Face token.

## Examples

1. Quantize a model and publish under your personal account:
   ```bash
   python app/main.py --author mistralai --model Mistral-7B-Instruct-v0.3
   ```

2. Quantize a model and publish under a specific organization:
   ```bash
   python app/main.py --author cognitivecomputations --model dolphin-2.9.4-gemma2-2b --quanter solidrust
   ```

## Process Overview

1. The tool authenticates with Hugging Face using your token.
2. It downloads the specified model from Hugging Face.
3. A new repository for the AWQ model is created (if it doesn't exist).
4. The model is converted to safetensors format (if necessary).
5. The model is quantized using AutoAWQ.
6. The quantized model is validated.
7. The quantized model and updated README are uploaded to the new repository.

## Idempotent Operation

The tool is designed to be idempotent. If you run the command multiple times:
- It will not re-download the model if it already exists locally.
- It will not re-quantize the model if a quantized version already exists.
- It will validate an existing quantized model before deciding to re-quantize.

## Troubleshooting

- If you encounter authentication errors, ensure your Hugging Face token is correctly set and has the necessary permissions.
- For out-of-memory errors, try using a machine with more RAM or GPU memory.
- Check the logs (in the `logs/` directory) for detailed error messages and the process flow.

## Environment Variables

- `HF_ACCESS_TOKEN`: Your Hugging Face access token.
- `QUANTER`: Default quanter name to use if not provided via CLI argument.

You can set these in a `.env` file in the project root or export them in your shell.

For more specific issues or contributions, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.
