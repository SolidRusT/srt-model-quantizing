# Exllama2 Quantization pipeline

This little script will loop through an array of preferred BPW (bits per weight), and then upload each one to their own branches in git.

## Requirements

- Working CUDA 12+ OR ROCm 6+ environment
- Working Torch library that matches your GPU
- Python 3.11.x virtual environment
