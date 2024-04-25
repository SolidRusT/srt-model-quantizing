#!/bin/bash
export AUTHOR=$1
export MODEL=$2

export QUANTER="solidrust"
export QUANT_SCRIPT="run-quant-awq.py"
export APP_HOME=${APP_HOME:-"/opt/quant-awq"}
export SRT_REPO="${APP_HOME}/repos/srt-model-quantizing/awq"
export SRT_DATA="${APP_HOME}/data"

mkdir -p ${APP_HOME}

# Uncomment for single GPU
#export CUDA_VISIBLE_DEVICES=0

# Ensure that script stops on first error
set -e

function garbage_collect() {
  rm -rf ${HOME}/.cache/huggingface/hub/models--*
  rm -rf ${SRT_DATA}/*-AWQ
}

function git_commit() {
  wherami=$(pwd)
  cd ${SRT_DATA}/${MODEL}-AWQ
  git lfs install
  git add .
  git commit -m "$1"
  git pull
  git push
  cd $wherami
}

function update_readme() {
  sed -i "s/{AUTHOR}/${AUTHOR}/g" ${SRT_DATA}/${MODEL}-AWQ/README.md
  sed -i "s/{MODEL}/${MODEL}/g" ${SRT_DATA}/${MODEL}-AWQ/README.md
}

function create_quant_repo() {
  (huggingface-cli repo create --organization ${QUANTER} ${MODEL}-AWQ -y)
}

function processing_notice() {
  cp ${SRT_REPO}/processing-notice.txt ${SRT_DATA}/${MODEL}-AWQ/README.md
  update_readme
  git_commit "add processing notice"
}

function add_quant_config() {
  cp ${SRT_REPO}/quant_config.json ${SRT_DATA}/${MODEL}-AWQ/quant_config.json
  git_commit "adding quant config"
}

function clone_quant_repo() {
  git clone "git@hf.co:${QUANTER}/${MODEL}-AWQ" ${SRT_DATA}/${MODEL}-AWQ
}

function quant_model() {
  python ${SRT_REPO}/${QUANT_SCRIPT} \
    --model_path ${AUTHOR}/${MODEL} \
    --quant_path ${SRT_DATA}/${MODEL}-AWQ \
    --zero_point True --q_group_size 128 --w_bit 4 --version GEMM
  git_commit "adding AWQ model"
}

# Main Program
garbage_collect
create_quant_repo
clone_quant_repo
processing_notice
quant_model
add_quant_config
