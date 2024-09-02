# ROADMAP for srt-model-quantizing

This roadmap outlines the planned improvements and tasks for the srt-model-quantizing project. It is designed to guide our development cycles and maintain alignment across the project.

## 1. Documentation Consolidation and Update
- [x] Merge information from multiple README files into a single, comprehensive README.md in the root directory.
- [x] Include clear instructions for both AWQ and Exllama2 quantization methods.
- [x] Add a section on project structure and how to use the application.

## 2. Repository Restructuring
- [x] Create separate directories for AWQ and Exllama2 quantization methods.
- [x] Move relevant files from the root directory into appropriate subdirectories.
- [x] Create a `common` directory for any code used by both quantization methods.

## 3. Standardize Project Structure
- [x] Apply the structure used in the AWQ version2 folder to the Exllama2 implementation.
- [ ] Ensure both methods have consistent file naming and organization.

## 4. Code Review and Refactoring
- [ ] Identify and eliminate any dead or duplicate code.
- [ ] Standardize coding style across all files (use a linter like flake8 or black).
- [ ] Add proper type hints to improve code readability and maintainability.

## 5. Improve Error Handling and Logging
- [ ] Implement consistent error handling across all scripts.
- [ ] Add more detailed logging to help with debugging and monitoring.

## 6. Update Dependencies
- [ ] Review and update `requirements.txt` files.
- [ ] Ensure all necessary dependencies are listed with appropriate versions.
- [ ] Consider creating separate requirements files for different environments (e.g., CUDA, ROCm).

## 7. Enhance Command-line Interface
- [ ] Create a unified entry point for both quantization methods.
- [ ] Implement robust argument parsing with clear help messages.

## 8. Improve Test Coverage
- [ ] Expand unit tests for both AWQ and Exllama2 implementations.
- [ ] Ensure all critical functions have corresponding unit tests.

## 9. Create Contributing Guidelines
- [ ] Create a CONTRIBUTING.md file.
- [ ] Outline guidelines for contributors.
- [ ] Describe the development workflow and coding standards.

## 10. License Review
- [ ] Ensure all files have appropriate license headers.
- [ ] Verify that the LICENSE file is up-to-date and correctly reflects the project's licensing.

## 11. Clean up Scratch Space
- [ ] Review scripts in the `scratch-space` directory.
- [ ] Integrate useful scripts into the main codebase or remove if unnecessary.

## 12. Implement CI/CD
- [ ] Set up basic CI/CD pipelines for automated testing and linting.

## Future Considerations
- [ ] Explore support for additional quantization methods.
- [ ] Investigate performance optimizations for large model quantization.
- [ ] Consider creating a user interface for easier model selection and quantization.

This roadmap is a living document and may be updated as the project evolves and new priorities emerge.