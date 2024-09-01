# Contributing to SRT Model Quantizing

We welcome contributions to the SRT Model Quantizing project! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/solidrust/srt-model-quantizing.git
   cd srt-model-quantizing
   ```
3. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-or-fix-name
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. Install the development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Making Changes

1. Make your changes in your feature branch.
2. Add or update tests as necessary.
3. Run the tests to ensure they pass:
   ```bash
   python -m unittest discover tests
   ```
4. Update documentation if you're changing functionality.

## Commit Guidelines

- Use clear and meaningful commit messages.
- Reference issue numbers in your commit messages if applicable.

## Submitting Changes

1. Push your changes to your fork on GitHub.
2. Submit a pull request to the main repository.
3. Describe your changes in the pull request description.
4. Link any relevant issues in the pull request description.

## Code Style

- Follow PEP 8 guidelines for Python code.
- Use type hints where possible.
- Keep functions and methods focused and small.

## Testing

- Add unit tests for new functionality.
- Ensure all tests pass before submitting a pull request.
- Aim for high test coverage for new code.

## Documentation

- Update the README.md if you're changing user-facing functionality.
- Update or add docstrings for new functions, classes, or modules.
- Consider updating USAGE.md for significant feature additions or changes.

## Reporting Issues

- Use the GitHub issue tracker to report bugs.
- Clearly describe the issue, including steps to reproduce.
- Include the Python version and operating system you're using.

## Community

- Be respectful and considerate in all interactions.
- Help review pull requests from other contributors.

Thank you for contributing to SRT Model Quantizing!