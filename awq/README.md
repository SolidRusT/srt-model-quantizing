# Model Quantization Script

This script quantizes a causal language model using the AWQ (Automatic Weight Quantization) library and saves the quantized model.

## Requirements

- Python 3.6+
- PyTorch 1.5.0+
- Transformers 4.6.0+
- AWQ

## Usage

You can run the script from the command line with the following arguments:

- `--model_path`: Path to the model to be quantized. This is a required argument.
- `--quant_path`: Path to save the quantized model. This is a required argument.
- `--zero_point`: Whether to use zero point. Default is `True`.
- `--q_group_size`: Quantization group size. Default is `128`.
- `--w_bit`: Weight bit. Default is `4`.
- `--version`: Version. Default is `GEMM`.

Here's an example of how to run the script:

```bash
python your_script.py --model_path /path/to/your/model --quant_path /path/to/save/quantized/model --zero_point True --q_group_size 128 --w_bit 4 --version GEMM
