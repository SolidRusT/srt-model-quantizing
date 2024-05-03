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

function logger() {
  echo "8===D $1" | tee -a "${SRT_DATA}/quant-awq.log"
}

function garbage_collect() {
  logger "Taking out the trash"
  rm -rf ${HOME}/.cache/huggingface/hub/models--*
  rm -rf ${SRT_DATA}/*-AWQ
}

function upload() {
  message=${1:-"empty message"}
  logger "Uploading: ${message}"
  huggingface-cli upload "${QUANTER}/${MODEL}-AWQ" "${SRT_DATA}/${MODEL}-AWQ/" . --commit-message "${message}"
}

function update_readme() {
  logger "Spiff up the README"
  # TODO: huggingface-cli upload username/my-space --remote https://example.com/data.csv
  sed -i "s/{AUTHOR}/${AUTHOR}/g" ${SRT_DATA}/${MODEL}-AWQ/README.md
  sed -i "s/{MODEL}/${MODEL}/g" ${SRT_DATA}/${MODEL}-AWQ/README.md
}

function create_quant_repo() {
  logger "Create a new repo"
  (huggingface-cli repo create --organization ${QUANTER} ${MODEL}-AWQ -y)
}

function processing_notice() {
  logger "add processing notice"
  cp ${SRT_REPO}/processing-notice.txt ${SRT_DATA}/${MODEL}-AWQ/README.md
  update_readme
  upload "add processing notice"
}

function add_quant_config() {
  logger "add quant config"
  cp ${SRT_REPO}/quant_config.json ${SRT_DATA}/${MODEL}-AWQ/quant_config.json
  upload "adding quant config"
}

function add_model_card() {
  logger "add model card"
  cp ${SRT_REPO}/initial-readme.txt ${SRT_DATA}/${MODEL}-AWQ/README.md
  update_readme
  upload "add default model card"
}

function clone_quant_repo() {
  logger "add quant repo"
  huggingface-cli download "${QUANTER}/${MODEL}-AWQ" --local-dir "${APP_HOME}/data/${MODEL}-AWQ/"
}

function quant_model() {
  logger "quantize the model"
  python ${SRT_REPO}/${QUANT_SCRIPT} \
    --model_path ${AUTHOR}/${MODEL} \
    --quant_path ${SRT_DATA}/${MODEL}-AWQ \
    --zero_point True --q_group_size 128 --w_bit 4 --version GEMM
  upload "adding AWQ model"
}

# Main Program
garbage_collect
create_quant_repo
clone_quant_repo
processing_notice
quant_model
add_quant_config
add_model_card
