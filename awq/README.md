# AWQ Quantization

This repository contains the implementation for AWQ (Activation-aware Weight Quantization) model quantization. It provides a pipeline for downloading models from Hugging Face, quantizing them using the AutoAWQ method, and uploading them to a Hugging Face-compatible repository.

## Features

- Download models from Hugging Face Hub
- Quantize models using AutoAWQ
- Upload quantized models to Hugging Face Hub
- Support for both NVIDIA CUDA and AMD ROCm GPUs
- Automatic handling of model conversion to safetensors format

## Requirements

- python 3.11
- CUDA 12.1 (for GPU acceleration)
- HuggingFace API token

## Installation

1. Clone the repository:

```bash
git clone clone https://github.com/solidrust/srt-model-quantizing.git
cd srt-model-quantizing/awq
```

2. Install the dependencies:

```bash
pip install -U -r requirements.txt
```

3. Setup your HuggingFace API token:

```bash
export HUGGINGFACE_TOKEN=<your_token>
```

## Usage

To quantize a model, use the following command:

```bash
python app/main.py <author>/<model> [--quanter <quanter>]
```

Arguments:
- `<author>`: The author of the model on Hugging Face Hub
- `<model>`: The name of the model to quantize
- `--quanter` (optional): Specify a custom quanter name (can be set from .env file as QUANTER)

Example:

```bash
python app/main.py cognitivecomputations/dolphin-2.9.4-gemma2-2b --quanter solidrust
```

## Configuration

The `app/config.py` file contains various configuration options. You can modify these as needed for your environment, but the defaults should work for most users. Most of the configuration is handled through environment variables, which can be set in the `.env` file.

## Testing

To run the unit tests, use:

```bash
python -m unittest discover tests
```

## Contributing

We welcome contributions to this project. Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- This project uses the AutoAWQ library, which is licensed under the MIT License. See the [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) repository for more details.
- Thanks to the HuggingFace team for providing a great platform for sharing and collaborating on AI research.

For more information about the AWQ quantization process and its implementation, please refer to the comments in the source code.
