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
  huggingface-cli download "${QUANTER}/${MODEL}-AWQ" --local-dir "${SRT_DATA}/${MODEL}-AWQ/"
  logger "$(ls -al ${SRT_DATA}/${MODEL}-AWQ)"
}

function download_model() {
  huggingface-cli download "${AUTHOR}/${MODEL}"
  logger "Downloading ${AUTHOR}/${MODEL} ..."
}

function pytorch_check() {
  logger "Checking for pytorch binaries"
  
  # Get the latest snapshot directory
  local snapshot_dir
  snapshot_dir=$(ls -1d ${HOME}/.cache/huggingface/hub/models--${AUTHOR}--${MODEL}/snapshots/* | tail -n 1)
  
  # Check for pytorch .bin files
  local pytorch_bins
  pytorch_bins=$(find "${snapshot_dir}" -name "pytorch_model-0*.bin")

  # Convert pytorch .bin files to safetensors if they exist
  if [[ -n "${pytorch_bins}" ]]; then
    local convert_script="${APP_HOME}/repos/srt-model-quantizing/convert-pytorch-simple.py"
    local bin

    for bin in ${pytorch_bins}; do
      if ! python "${convert_script}" "${bin}"; then
        logger "Error: Conversion failed for %s\n" "${bin}"
        return 1
      fi
    done

    printf "Conversion completed successfully.\n"
  else
    printf "No pytorch .bin files found for author '%s' and model '%s'.\n" "${AUTHOR}" "${MODEL}"
  fi
}

function quant_model() {
  download_model
  pytorch_check
  logger "quantize the model"
  python ${SRT_REPO}/${QUANT_SCRIPT} \
    --model_path ${AUTHOR}/${MODEL} \
    --quant_path ${SRT_DATA}/${MODEL}-AWQ \
    --zero_point True --q_group_size 128 --w_bit 4 --version GEMM
  logger "$(ls -al ${SRT_DATA}/${MODEL}-AWQ)"
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
