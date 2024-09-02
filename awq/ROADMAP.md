# Development Roadmap

## Current Sprint
- [x] Implement AWQ quantization using AutoAWQ
- [x] Update requirements.txt with necessary dependencies
- [x] Test the updated quantization process with various models
- [x] Implement model validation after quantization
- [x] Implement HuggingFace repo creation for AWQ models
- [x] Implement downloading of existing AWQ repo to data folder
- [x] Update README creation and upload process
- [x] Refactor main.py to follow the new workflow
- [x] Implement idempotent logic for existing AWQ models
- [x] Add error handling for various scenarios in the new workflow
- [x] Update documentation with new workflow details
- [x] Reorganize static content templates into a dedicated directory
- [x] Create Dockerfile for easy setup and deployment
- [x] Implement model checksum validation
- [x] Enhance error handling for device mismatch issues during quantization
- [x] Improve GPU memory reporting and error messages
- [x] Add support for models that may be partially offloaded to CPU/disk
- [x] Implement graceful fallback to CPU quantization for large models
- [x] Improve compatibility checks for AWQ quantization support
- [ ] Optimize CPU quantization performance for large models
- [ ] Implement memory usage estimation before quantization to prevent OOM errors
- [ ] Add progress bar or ETA for CPU quantization
- [ ] Expand test coverage for different model types and sizes
- [ ] Refactor main.py for better readability and maintainability

## Backlog
- [ ] Implement strategy for handling very large models (e.g., model sharding)
- [ ] Add option to specify output version (GEMM, GEMV)
- [ ] Add option to specify output group size (128, 64, 32, etc.)
- [ ] Add error handling for out-of-memory situations
- [ ] Optimize model loading process for large models
- [ ] Move configuration to YAML or JSON file
- [ ] Implement support for additional quantization methods
- [ ] Add support for model checksum validation
- [ ] Improve documentation with more examples and use cases
- [ ] Implement parallel processing for faster quantization
- [ ] Add support for quantizing multiple models in batch
- [ ] Create a user-friendly CLI interface
- [ ] Implement support for multi-GPU setups
- [ ] Add option to specify custom CUDA device
- [ ] Implement graceful degradation for large models (e.g., automatic sharding)
- [ ] Implement automatic model sharding for very large models
- [ ] Add support for distributed quantization across multiple GPUs
- [ ] Implement a web interface for easier model management and quantization
- [ ] Add option to specify custom CUDA device in the command line interface
- [ ] Implement progressive loading for extremely large models

## Completed
- [x] Fix model downloading issue in model_utils.py
- [x] Improve error handling and logging for the download process
- [x] Expand test coverage
- [x] Enhance error handling and logging across all modules
- [x] Improve user feedback and progress indication
- [x] Restore organized folder structure for output files
- [x] Ensure complete processing of all model files
- [x] Implement basic quantization workflow
- [x] Create and update README.md for AWQ models
- [x] Implement automatic determination of quanter when not provided
- [x] Implement model checksum validation
- [x] Add support for latest Llama 3.1 models
