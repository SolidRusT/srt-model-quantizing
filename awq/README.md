# SRT Model Quantizing

SRT Model Quantizing is a tool for downloading models from Hugging Face, quantizing them using the AutoAWQ method, and uploading them to a Hugging Face-compatible repository.

## Features

- Download models from Hugging Face
- Quantize models using AutoAWQ
- Automatic creation and management of quantized model repositories
- Idempotent operation (safe to run multiple times)
- Automatic determination of the quanter (user or organization)

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

Basic usage:

```bash
python app/main.py --author <author> --model <model> [--quanter <quanter>]
```

- `<author>`: The author of the original model on Hugging Face
- `<model>`: The name of the original model on Hugging Face
- `<quanter>` (optional): The user or organization to publish the AWQ model under. If not provided, it will be automatically determined from your Hugging Face access token.

Example:

```bash
python app/main.py --author mistralai --model Mistral-7B-Instruct-v0.3 --quanter solidrust
```

For more detailed usage instructions, please refer to the [USAGE.md](USAGE.md) file.

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
