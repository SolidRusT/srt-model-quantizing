#! /bin/bash
## to use the latest version of AutoAWQ, we need to install it from source
## this can take about 20mins + depending on the speed of your CPU

# configure the environment
work_dir=$(pwd)
my_home=${HOME}

# create python virtualenv
rm -rf ${my_home}/venv-awq-master
python -m venv ${my_home}/venv-awq-master
source ${my_home}/venv-awq-master/bin/activate
pip install --upgrade pip
pip install wheel
pip install -U transformers>=4.35.0 torch>=2.3.1 pandas protobuf sentencepiece requests tqdm huggingface_hub python-dotenv

# prepare working directory
mkdir -p ${work_dir}/repos

# clone AutoAWQ_kernels
rm -rf ${work_dir}/repos/AutoAWQ_kernels
git clone https://github.com/casper-hansen/AutoAWQ_kernels.git ${work_dir}/repos/AutoAWQ_kernels
cd ${work_dir}/repos/AutoAWQ_kernels
# fix torch version
sed -i 's/"torch==2\.3\.1"/"torch>=2.3.1"/' setup.py
pip install .

# clone AutoAWQ
rm -rf ${work_dir}/repos/AutoAWQ
git clone https://github.com/casper-hansen/AutoAWQ.git ${work_dir}/repos/AutoAWQ
cd ${work_dir}/repos/AutoAWQ
pip install .

# end
deactivate
cd ${work_dir}
