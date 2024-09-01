# Development Roadmap

## Current Sprint
- [ ] Expand test coverage
  - [ ] Create tests/ directory
  - [ ] Implement tests for all modules
  - [ ] Update run_tests.py script
- [ ] Clean up repository structure
  - [ ] Clarify or remove repos/ directory
  - [ ] Update .gitignore and .cursorignore
- [ ] Enhance error handling and logging across all modules
- [ ] Update and expand documentation
  - [ ] Revise README.md
  - [ ] Expand NOTES.md
  - [ ] Add inline documentation to complex functions
- [ ] Create a command-line interface for easier use
- [ ] Implement progress tracking for long-running operations

## Backlog
- [ ] Optimize model loading process in model_utils.py
- [ ] Investigate parallelization opportunities for large models
- [ ] Move configuration to YAML or JSON file
- [ ] Implement support for additional quantization methods
- [ ] Add support for model checksum validation
- [ ] Improve documentation with more examples and use cases

## Completed
- [x] Update model_utils.py to support safetensors and split model files
- [x] Implement PyTorch to safetensors conversion in converter.py
- [x] Update quantization.py to handle split safetensors files
- [x] Fix template processing error in main.py and template_parser.py
- [x] Implement proper path handling for template files
- [x] Enhance error handling and logging for new file formats
- [x] Update download_model function to handle Hugging Face Hub's blob structure
