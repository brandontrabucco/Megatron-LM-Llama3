#!/bin/bash

export PYTORCH_DOCKER_IMAGE=${PYTORCH_DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:24.10-py3"}
export HUGGINGFACE_ACCESS_TOKEN=${HUGGINGFACE_ACCESS_TOKEN:-"your-token"}

export DOWNLOAD_MODEL_NAME=${DOWNLOAD_MODEL_NAME:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
export DOWNLOAD_DIR=${DOWNLOAD_DIR:-"Meta-Llama-3.1-8B-Instruct"}

export DATASET_INPUT_NAME=${DATASET_INPUT_NAME:-"large_web_dataset.jsonl"}
export DATASET_NAME=${DATASET_NAME:-"large_web_dataset"}

export DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-32}

export MEGATRON_DIR=${MEGATRON_DIR:-"/home/ubuntu/Megatron-LM"}
export DATASETS_DIR=${DATASETS_DIR:-"/home/ubuntu/datasets"}
export CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-"/home/ubuntu/llama-checkpoints"}

export PYTHON_REQUIREMENTS=(
    transformers
    accelerate
    huggingface-hub
    setuptools==69.5.1
)

read -r -d '' RUN_DOCKER_COMMAND <<- DOCKER_SCRIPT

pip install ${PYTHON_REQUIREMENTS[@]}
cd /workspace/megatron
mkdir -p $DOWNLOAD_DIR

huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN
huggingface-cli download $DOWNLOAD_MODEL_NAME \
    --local-dir $DOWNLOAD_DIR \
    --include "*.json" "*.safetensors"

export CUDA_DEVICE_MAX_CONNECTIONS=1

python tools/preprocess_data.py --workers $DATASET_NUM_WORKERS \
    --input /workspace/datasets/$DATASET_INPUT_NAME \
    --output-prefix /workspace/datasets/$DATASET_NAME \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $DOWNLOAD_DIR

DOCKER_SCRIPT

docker pull $PYTORCH_DOCKER_IMAGE
docker run --gpus all --shm-size 256G --rm \
    -v $MEGATRON_DIR:/workspace/megatron \
    -v $DATASETS_DIR:/workspace/datasets \
    -v $CHECKPOINTS_DIR:/workspace/checkpoints \
    $PYTORCH_DOCKER_IMAGE \
    bash -c "$RUN_DOCKER_COMMAND"