# Quant-AWQ Application

The Quant-AWQ Application is designed to automate the quantization process for models using the AWQ method, supporting conversions from PyTorch models to `safetensors` format and integrating with the Hugging Face ecosystem.

## Features

- **Automated Quantization Workflow**: Converts and quantizes models to use lower-bit representations.
- **Template-based README Management**: Automatically updates README files using predefined templates to reflect model processing stages.
- **Support for PyTorch Models**: Handles `.bin` and `.pt` files, converting them into the required format before quantization.
- **Logging and Error Handling**: Robust logging and error handling for traceability and troubleshooting.

## Prerequisites

Before you begin, ensure you have Python 3.8+ installed on your system. You will also need the following packages, which can be installed via pip:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the Repository**: Get a copy of the source code on your local machine.

    ```bash
    git clone https://your-repository-url.com/quant-awq.git
    cd quant-awq
    ```

2. **Install Dependencies**: Install the required Python libraries.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, use the following command:

```bash
python app/main.py [author] [model]
```

Replace `[author]` and `[model]` with the appropriate values for the model you wish to quantize.

### Configuration

Edit the `app/config.py` to modify the application settings such as paths and quantization parameters.

## Structure

- `app/`: Main application code.
  - `main.py`: Entry point of the application.
  - `model_utils.py`: Utilities for model downloading and management.
  - `quantization.py`: Core logic for model quantization.
  - `converter.py`: Handles the conversion of PyTorch models to `safetensors`.
  - `template_manager.py`: Manages README file updates from templates.
- `data/`: Directory for storing models and other data.
- `logs/`: Contains application logs.

## Testing

For testing the application, navigate to the project directory and run:

```bash
python -m unittest discover tests
```

Ensure all components function as expected and handle both typical use cases and edge cases.

## Contributing

Contributions to the Quant-AWQ project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
