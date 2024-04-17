#!/bin/bash

# nbeerbower/Flammen-Bophades-7B
export MODEL="Flammen-Bophades-7B"
export AUTHOR="nbeerbower"
export QUANTER="solidrust"
export QUANT_SCRIPT="run-quant-awq.py"

export CUDA_VISIBLE_DEVICES=0,1

# Ensure that script stops on first error
set -e

function create_quant_repo() {
    (huggingface-cli repo create --organization ${QUANTER} ${MODEL}-AWQ -y)
}

function clone_quant_repo() {
    git lfs install
    git clone "git@hf.co:${QUANTER}/${MODEL}-AWQ"
}

function quant_model() {
    python ${QUANT_SCRIPT} \
    --model_path ${AUTHOR}/${MODEL} \
    --quant_path ${QUANTER}/${MODEL}-AWQ \
    --zero_point True --q_group_size 128 --w_bit 4 --version GEMM
}

#function test_inference() {
#    python "${EXL2_HOME}/test_inference.py" -m "${MODEL_DIR}/${MODEL}-exl2/" -p "Once upon a time,"
#}

function upload_model_quant() {
    cp quant_config.json "${QUANTER}/${MODEL}-AWQ/"
    cd "${QUANTER}/${MODEL}-AWQ/"
    git add .
    git commit -m "adding AWQ model"
    git pull
    git push
}

# Main Program
create_quant_repo
clone_quant_repo
quant_model
#test_inference
upload_model_quant
