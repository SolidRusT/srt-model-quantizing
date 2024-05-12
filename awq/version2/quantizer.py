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
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    tensor_files = glob.glob(os.path.join(pytorch_dir, "*.bin")) + glob.glob(os.path.join(pytorch_dir, "*.pt"))

    if not tensor_files:
        logging.warning(f"No .bin or .pt files found in directory {pytorch_dir}")
        return

    for file in tensor_files:
        logging.info(f"Converting {file} to `safetensors`...")
        state_dict = torch.load(file, map_location=device)

        if unshare:
            state_dict = {k: v.clone().detach().contiguous() for k, v in state_dict.items()}

        output_file = os.path.splitext(file)[0] + ".safetensors"
        save_file(state_dict, output_file, metadata={"format": "pt"})
        logging.info(f"Saved {output_file} successfully.")

    return pytorch_dir  # Returning the same directory path for further usage
