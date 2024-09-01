import torch
import argparse, os, glob
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description="Convert .bin/.pt files to .safetensors")
parser.add_argument("--unshare", action="store_true", help="Detach tensors to prevent any from sharing memory")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU to process tensors if available")
parser.add_argument("input_files", nargs="+", type=str, help="Input file(s)")
args = parser.parse_args()

# Checking for available GPUs and setting device accordingly
device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

tensor_files = []
for file_pattern in args.input_files:
    tensor_files.extend(glob.glob(file_pattern))

for file in tensor_files:
    print(f" -- Loading {file}...")
    # Load tensors to specified device
    state_dict = torch.load(file, map_play="cpu")
    if args.use_gpu and torch.cuda.is_available():
        # If GPU is available and the flag is set, transfer tensors to GPU
        state_dict = {k: v.to(device) for k, v in state_dict.items()}

    if args.unshare:
        for k, v in state_dict.items():
            # Ensure each tensor is detached, cloned, and made contiguous
            new_tensor = v.clone().detach().contiguous()
            state_dict[k] = new_tensor
            print(f"Tensor {k} is now contiguous: {new_tensor.is_contiguous()}")

    out_file = os.path.splitext(file)[0] + ".safetensors"
    print(f" -- Saving {out_file}...")

    # Convert tensors back to CPU for saving
    if device.type == 'cuda':
        state_dict = {k: v.cpu() for k, v in state_dict.items()}

    save_file(state_dict, out_file, metadata={"format": "pt"})
