# SRT Model Quantizing

SRT Model Quantizing is a tool for downloading models from Hugging Face, quantizing them using the AutoAWQ method, and uploading them to a Hugging Face-compatible repository.

## Features

- Download models from Hugging Face
- Quantize models using AutoAWQ
- Automatic creation and management of quantized model repositories
- Idempotent operation (safe to run multiple times)
- Automatic determination of the quanter (user or organization)
- Model checksum validation for integrity verification

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/solidrust/srt-model-quantizing.git
   cd srt-model-quantizing
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face token:
   - Create an environment variable `HF_ACCESS_TOKEN` with your Hugging Face token
   - Or, use the Hugging Face CLI to log in: `huggingface-cli login`

## Usage

You can run the quantization process using either of the following methods:

1. Combined author/model format:
   ```bash
   python app/main.py <author>/<model> [--quanter <quanter>] [--expected-checksum <checksum>]
   ```

2. Separate author and model arguments (maintained for backward compatibility):
   ```bash
   python app/main.py --author <author> --model <model> [--quanter <quanter>] [--expected-checksum <checksum>]
   ```

- `<author>/<model>`: The combined author and model name as it appears on Hugging Face
- `<author>`: The author of the original model on Hugging Face
- `<model>`: The name of the original model on Hugging Face
- `<quanter>` (optional): The user or organization to publish the AWQ model under. If not provided, it will be automatically determined from your Hugging Face access token.
- `<checksum>` (optional): The expected checksum for the model.

Example using the combined format:

```bash
python app/main.py mist
```

For more detailed usage instructions, please refer to the [USAGE.md](USAGE.md) file.

### Model Checksum Validation

The tool now includes a checksum validation feature to ensure the integrity of downloaded models. If you have an expected checksum for a model, you can provide it during the quantization process:

```bash
python app/main.py --author <author> --model <model> --expected-checksum <expected-checksum>
```

- `<expected-checksum>`: The expected checksum for the model.

## Docker Usage

To use the Docker container:

1. Build the Docker image:
   ```bash
   docker build -t srt-model-quantizing .
   ```

2. Run the container:
   ```bash
   docker run -v $(pwd)/data:/srt-model-quantizing/data \
              -v $(pwd)/logs:/srt-model-quantizing/logs \
              -e HF_ACCESS_TOKEN=your_token_here \
              srt-model-quantizing \
              python app/main.py --author mistralai --model Mistral-7B-Instruct-v0.3 --quanter your_quanter_name
   ```

   Replace `your_token_here` with your Hugging Face token and `your_quanter_name` with your desired quanter name.

## Troubleshooting

### Model Compatibility Issues

1. "LlamaForCausalLM object has no attribute 'quantize'" error:
   - This error occurs when the loaded model doesn't support AWQ quantization.
   - Solution: Ensure you're using a model that's compatible with AWQ quantization and that you have the correct version of AutoAWQ installed.

2. "You can't move a model that has some modules offloaded to cpu or disk" error:
   - This error occurs when the model is too large to fit entirely in GPU memory and some parts are offloaded.
   - Solution: 
     - Use a GPU with more memory.
     - Implement a strategy for handling large models (e.g., model sharding, which is not currently supported in this tool).
     - Consider using CPU quantization for very large models, although this will be significantly slower.

3. Meta device errors or device mismatch errors:
   - These errors often occur due to insufficient GPU memory.
   - Solution:
     - Use a smaller model.
     - Use a GPU with more memory.
     - Increase your GPU memory allocation if possible.

If you encounter any of these issues, please check our issue tracker or open a new issue with details about your setup, the model you're trying to quantize, and the full error message.

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
