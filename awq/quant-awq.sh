#!/bin/bash
export AUTHOR=$1
export MODEL=$2

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
    --quant_path ${MODEL}-AWQ \
    --zero_point True --q_group_size 128 --w_bit 4 --version GEMM
}

#function test_inference() {
#    python "${EXL2_HOME}/test_inference.py" -m "${MODEL_DIR}/${MODEL}-exl2/" -p "Once upon a time,"
#}

function upload_model_quant() {
    cd "${MODEL}-AWQ/"
    git lfs install
    git add .
    git commit -m "adding AWQ model"
    cp quant_config.json "${MODEL}-AWQ/"
    git add .
    git commit -m "adding quant config"
    cp initial-readme.txt "${MODEL}-AWQ/README.md"
    git add .
    git commit -m "adding initial model card"
    git pull
    git push
}

# Main Program
create_quant_repo
clone_quant_repo
quant_model
#test_inference
upload_model_quant
