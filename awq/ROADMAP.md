# Development Roadmap

## Current Sprint
- [x] Update model_utils.py to support safetensors and split model files
- [x] Implement PyTorch to safetensors conversion in converter.py
- [x] Update quantization.py to handle split safetensors files
- [x] Fix template processing error in main.py and template_parser.py
- [x] Implement proper path handling for template files
- [x] Enhance error handling and logging for new file formats
- [x] Update download_model function to handle Hugging Face Hub's blob structure
- [ ] Test and verify the updated model download and file checking process
- [ ] Clean up repository structure and remove outdated files
- [ ] Implement support for additional quantization methods

## Backlog
- [ ] Create a command-line interface for easier use
- [ ] Improve documentation with examples and use cases
- [ ] Implement progress tracking for long-running operations
- [ ] Add support for model checksum validation
- [ ] Implement parallel processing for large models
- [ ] Integrate template files (processing-notice.txt, initial-readme.txt) into the main application

## Completed
- [x] Refactor quantization.py to improve error handling and logging
- [x] Implement unit tests for converter.py
- [x] Update README.md with more detailed usage instructions
- [x] Optimize model loading process in model_utils.py
- [x] Update model_utils.py to support safetensors and split model files
- [x] Implement PyTorch to safetensors conversion in converter.py
- [x] Update quantization.py to handle split safetensors files
- [x] Implement safetensors model download in model_utils.py
- [x] Fix template processing error in main.py and template_parser.py
- [x] Implement proper path handling for template files
- [x] Enhance error handling and logging for new file formats
- [x] Update download_model function to handle Hugging Face Hub's blob structure
