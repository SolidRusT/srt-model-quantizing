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
  huggingface-cli upload --commit-message "$1" ${QUANTER}/${MODEL} . .
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

function add_model_card() {
  cp ${SRT_REPO}/initial-readme.txt ${SRT_DATA}/${MODEL}-AWQ/README.md
  update_readme
  git_commit "add default model card"
}

function clone_quant_repo() {
  huggingface-cli download "${QUANTER}/${MODEL}-AWQ" --local-dir "${APP_HOME}/data/${MODEL}-AWQ/"
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
add_model_card
