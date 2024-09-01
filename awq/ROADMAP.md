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
- [ ] Implement progress tracking for long-running operations
- [ ] Add option to specify output bit-width (2-bit, 3-bit, 4-bit, 8-bit)

## Backlog
- [ ] Add error handling for out-of-memory situations
- [ ] Optimize model loading process for large models
- [ ] Move configuration to YAML or JSON file
- [ ] Implement support for additional quantization methods
- [ ] Add support for model checksum validation
- [ ] Improve documentation with more examples and use cases
- [ ] Implement parallel processing for faster quantization
- [ ] Add support for quantizing multiple models in batch
- [ ] Create a user-friendly CLI interface

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
