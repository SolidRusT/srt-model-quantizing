# Development Roadmap

## Current Sprint
- [x] Update model_utils.py to support safetensors and split model files
- [x] Implement PyTorch to safetensors conversion in converter.py
- [x] Update quantization.py to handle split safetensors files
- [ ] Enhance error handling and logging for new file formats

## Backlog
- [ ] Implement support for additional quantization methods
- [ ] Create a command-line interface for easier use
- [ ] Improve documentation with examples and use cases
- [ ] Implement progress tracking for long-running operations
- [ ] Add support for model checksum validation

## Completed
- [x] Refactor quantization.py to improve error handling and logging
- [x] Implement unit tests for converter.py
- [x] Update README.md with more detailed usage instructions
- [x] Optimize model loading process in model_utils.py
- [x] Update model_utils.py to support safetensors and split model files
  - Added support for downloading split safetensors files
  - Implemented PyTorch to safetensors conversion
  - Updated file checking process to handle both formats
- [x] Implement PyTorch to safetensors conversion in converter.py
  - Added functionality to convert PyTorch files to safetensors
  - Implemented handling of split safetensors files
  - Added functions to load and save safetensors models, including split files
- [x] Update quantization.py to handle split safetensors files
  - Modified quantization process to work with safetensors format
  - Integrated loading and saving of split safetensors files
  - Added function to calculate total size of quantized model
