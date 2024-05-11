# huggingface-cli download rombodawg/Llama-3-8B-Base-Coder-v3.5-10k
# python $APP_HOME/repos/srt-model-quantizing/convert-pytorch-simple.py $HOME/.cache/huggingface/hub/models--rombodawg--Llama-3-8B-Base-Coder-v3.5-10k/snapshots/290cefb72390860fb36c87a76dcd9d41b112561b/pytorch_model-00004-of-00004.bin
import torch
import argparse, os, glob
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description="Convert .bin/.pt files to .safetensors")
parser.add_argument("--unshare", action="store_true", help="Detach tensors to prevent any from sharing memory")
parser.add_argument("input_files", nargs="+", type=str, help="Input file(s)")
args = parser.parse_args()

tensor_files = []
for file_pattern in args.input_files:
    tensor_files.extend(glob.glob(file_pattern))

for file in tensor_files:
    print(f" -- Loading {file}...")
    state_dict = torch.load(file, map_location="cpu")

    if args.unshare:
        for k in state_dict.keys():
            state_dict[k] = state_dict[k].clone().detach()

    out_file = os.path.splitext(file)[0] + ".safetensors"
    print(f" -- Saving {out_file}...")
    save_file(state_dict, out_file, metadata={"format": "pt"})
