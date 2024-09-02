# AWQ Quantization

This directory contains the implementation for AWQ (Activation-aware Weight Quantization) model quantization.

## Usage

To use the AWQ quantization tool, make sure you're in the project root directory and run:

```bash
python -m awq <author> <model> [--quanter <quanter>]
```

Arguments:
- `<author>`: The author of the model on Hugging Face Hub
- `<model>`: The name of the model to quantize
- `--quanter` (optional): Specify a custom quanter name

```bash
python -m awq cognitivecomputations/dolphin-2.9.4-gemma2-2b --quanter solidrust
```

## Requirements

Ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

## Configuration

The `app/config.py` file contains various configuration options. You can modify these as needed for your environment.

## Testing

To run the unit tests, use:

```bash
python -m unittest discover tests
```

For more detailed information about the AWQ quantization process and its implementation, please refer to the comments in the source code.

