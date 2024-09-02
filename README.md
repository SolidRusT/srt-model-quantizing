# srt-model-quantizing

srt-model-quantizing is a pipeline for downloading models from Hugging Face, quantizing them, and then uploading them to a Hugging Face-compatible repository. This project is developed by SolidRusT Networks and supports two quantization methods: Exllama2 and AutoAWQ.

## Features

- Supports both Exllama2 and AutoAWQ quantization methods
- Designed for simplicity and ease of use
- Supports both Nvidia CUDA and AMD ROCm GPUs
- Intended for use on Linux servers

## Project Structure

```plaintext
srt-model-quantizing/
├── awq/ # AutoAWQ quantization implementation
│ ├── app/
│ ├── tests/
│ ├── requirements.txt
│ └── README.md
├── exl2/ # Exllama2 quantization implementation
│ ├── app/
│ ├── tests/
│ ├── templates/
│ ├── requirements.txt
│ └── README.md
└── README.md # This file
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/SolidRusT/srt-model-quantizing.git
   cd srt-model-quantizing
   ```

2. Set up virtual environments for each quantization method:

   For AWQ:
   ```
   python -m venv awq_venv
   source awq_venv/bin/activate
   cd awq
   pip install -r requirements.txt
   ```

   For Exllama2:
   ```
   python -m venv exl2_venv
   source exl2_venv/bin/activate
   cd exl2
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face access token:
   ```
   export HF_ACCESS_TOKEN=your_access_token_here
   ```

## Usage

### AWQ Quantization

1. Activate the AWQ virtual environment:
   ```
   source awq_venv/bin/activate
   ```

2. Navigate to the AWQ directory:
   ```
   cd awq
   ```

3. Run the quantization:
   ```
   python app/main.py <author> <model> [--quanter <quanter>]
   ```

   Example:
   ```
   python app/main.py cognitivecomputations/dolphin-2.9.4-gemma2-2b --quanter solidrust
   ```

### Exllama2 Quantization

1. Activate the Exllama2 virtual environment:
   ```
   source exl2_venv/bin/activate
   ```

2. Navigate to the Exllama2 directory:
   ```
   cd exl2
   ```

3. Run the quantization:
   ```
   python app/main.py <author> <model> [--quanter <quanter>]
   ```

   Example:
   ```
   python app/main.py cognitivecomputations/dolphin-2.9.4-gemma2-2b --quanter solidrust
   ```

## Configuration

Both AWQ and Exllama2 implementations have their own `config.py` files in their respective `app` directories. You can modify these files to adjust various settings such as output directories, quantization parameters, and more.

## Testing

To run tests for each implementation, navigate to the respective directory and run:

```bash
python -m unittest discover tests
```

## Contributing

Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
