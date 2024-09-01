# srt-model-quantizing

srt-model-quantizing is a pipeline for downloading models from Hugging Face, quantizing them, and then uploading them to a Hugging Face-compatible repository. This project is developed by SolidRusT Networks and supports two quantization methods: Exllama2 and AutoAWQ.

## Features

- Supports both Exllama2 and AutoAWQ quantization methods
- Designed for simplicity and ease of use
- Attempts to support both Nvidia CUDA and AMD ROCm GPUs
- Intended for use on Linux servers

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SolidRusT/srt-model-quantizing.git
   cd srt-model-quantizing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### AWQ Quantization

To run the AWQ quantization:
