## Ubuntu 22.04 minimal install
sudo apt install python3.11-full
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
#sudo update-alternatives --config python3
#sudo update-alternatives --set python /usr/bin/python3.11
python -m venv venv-rocm-3.11
source ~/venv-rocm-3.11/bin/activate
pip install --upgrade wheel wandb peft transformers datasets trl bitsandbytes accelerate
# /home/shaun/.cache/huggingface/accelerate/default_config.yaml
accelerate config
# https://github.com/ROCm/HIP/blob/develop/docs/install/install.rst
# https://github.com/ROCm/ROCm#installing-from-amd-rocm-repositories
