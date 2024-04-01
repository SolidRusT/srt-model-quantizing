#!/bin/bash
# Version 0.3.2
# Configuration
export MODEL="Zebrafish-7B"
export AUTHOR="mlabonne"
export QUANTER="Suparious"
export BPW=(3.5 4.25 5.0 6.5 8.0)
export CUDA_VISIBLE_DEVICES=0,1

export OG_REPO="https://huggingface.co/${AUTHOR}/${MODEL}"
export MODEL_DIR="${HOME}/hf_models"
export TEMP_DIR="${HOME}/hf_models/temp/exl2"
export EXL2_HOME="${HOME}/repos/exllamav2"

# Ensure that script stops on first error
set -e

# Functions
function create_quant_repo() {
    (cd "${MODEL_DIR}" && huggingface-cli repo create "${MODEL}-exl2" -y)
}

function clone_quant_repo() {
    git lfs install
    git clone "git@hf.co:${QUANTER}/${MODEL}-exl2" "${MODEL_DIR}/${MODEL}-exl2"
}

function clone_model_repo() {
    git lfs install
    git clone "git@hf.co:${AUTHOR}/${MODEL}" "${MODEL_DIR}"
}

function create_measurement() {
    echo "Measuring Exl2 for ${AUTHOR}'s model"
    echo "${OG_REPO}" > "${MODEL_DIR}/${MODEL}-exl2/original_repo_url.txt"
    python "${EXL2_HOME}/convert.py" -i "${MODEL_DIR}/${MODEL}/" -o "${TEMP_DIR}/" -nr -om "${MODEL_DIR}/${MODEL}-exl2/measurement.json"
    (cd "${MODEL_DIR}/${MODEL}-exl2" && git checkout main && git lfs install && git add . && git commit -m "exl2 quantization measurements" && git push --set-upstream origin main)
}

function quant_model() {
    python "${EXL2_HOME}/convert.py" -i "${MODEL_DIR}/${MODEL}/" -o "${TEMP_DIR}/" -nr -m "${MODEL_DIR}/${MODEL}-exl2/measurement.json" -cf "${MODEL_DIR}/${MODEL}-exl2/" -b "${bpw}" -hb "${hb}"
}

function test_inference() {
    python "${EXL2_HOME}/test_inference.py" -m "${MODEL_DIR}/${MODEL}-exl2/" -p "Once upon a time,"
}

function upload_model_quant() {
    (cd "${MODEL_DIR}/${MODEL}-exl2" && git checkout "${branch}" && huggingface-cli lfs-enable-largefiles . && git add . && git commit -m "exl2 quantization for ${bpw}" && git push --set-upstream origin "${branch}")
}


# Disable error stop for the following operations
set +e

# Prepare directories
mkdir -p "${TEMP_DIR}" "${MODEL_DIR}"

# Re-enable error stop
set -e

## Main Program
create_quant_repo
clone_quant_repo
clone_model_repo
create_measurement

# Run exl2 quantizations and handle branch operations
for bpw in "${BPW[@]}"; do
    branch=$(echo "${bpw}" | tr '.' '_')
    echo "branch: ${branch}, bpw: ${bpw}"

    # Determine head bit (hb) value based on bits per weight (bpw)
    hb=6
    [[ "${bpw}" == "8.0" || "${bpw}" == "6.5" ]] && hb=8
    echo "bpw: ${bpw}, hb: ${hb}"  # For verification

    cd "${MODEL_DIR}/${MODEL}-exl2" && git checkout main && git pull && git fetch --prune
    # Check and switch to the correct branch
    if git rev-parse --verify "refs/heads/${branch}" &>/dev/null; then
        echo "Branch ${branch} already exists locally."
        git checkout "${branch}"
    elif git rev-parse --verify "refs/remotes/origin/${branch}" &>/dev/null; then
        echo "Branch ${branch} exists in remote. Checking it out."
        git checkout -t "origin/${branch}"
    else
        echo "Branch ${branch} does not exist. Creating it."
        git checkout -b "${branch}"
    fi

    # Perform the quantization
    quant_model
    # Test the result
    test_inference
    # Upload to huggingFace
    upload_model_quant
done
