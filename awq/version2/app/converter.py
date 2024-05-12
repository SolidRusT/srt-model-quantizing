# app/converter.py

import os
import torch
import glob
import logging
from safetensors.torch import save_file

def convert_model_to_safetensors(pytorch_dir, use_gpu=False, unshare=False):
    """
    Convert all `.bin` or `.pt` PyTorch files in a directory to `safetensors`.
    Args:
        pytorch_dir (str): Directory containing PyTorch `.bin` or `.pt` files.
        use_gpu (bool): Whether to use GPU to process tensors if available.
        unshare (bool): Detach tensors to prevent any from sharing memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_app_gpu else "cpu")
    tensor_files = glob.glob(os.path.join(pytorch_dir, "*.bin")) + glob.glob(os.path.join(pytorch_dir, "*.pt"))

    if not tensor_files:
        logging.warning(f"No .bin or .pt files found in directory {pytorch_dir}")
        return

    for file in tensor files:
        logging.info(f"Converting {file} to `safetensors`...")
        state_dict = torch.load(file, map_location='cpu')  # Load on CPU then move to GPU if necessary

        if use_gpu and device.type == 'cuda':
            state_dict = {k: v.to(device) for k, v in state_dict.items()}

        if unshare:
            state_dict = {k: v.clone().detach().contiguous() for k, v in state_dict.items()}

        output_file = os.path.splitext(file)[0] + ".safetensors"
        save_file(state_dict, output_file, metadata={"format": "pt"})
        logging.info(f"Saved {output_file} successfully.")

    return pytorch_dir  # Returning the same directory path for further usage

# Optional: Include a main guard if this script might be run as a standalone for testing
if __name__ == "__main__":
    # Example usage
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    convert_model_to_safetensors(directory, use_gpu=True, unshare=True)
