#!/bin/bash

export PYTORCH_DOCKER_IMAGE=${PYTORCH_DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:24.10-py3"}
export HUGGINGFACE_ACCESS_TOKEN=${HUGGINGFACE_ACCESS_TOKEN:-"your-token"}

export DOWNLOAD_MODEL_NAME=${DOWNLOAD_MODEL_NAME:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
export DOWNLOAD_DIR=${DOWNLOAD_DIR:-"Meta-Llama-3.1-8B-Instruct"}
export PRETRAINED_DIR=${PRETRAINED_DIR:-"Meta-Llama-3.1-8B-Instruct-Megatron"}

export MEGATRON_MODEL_SIZE=${MEGATRON_MODEL_SIZE:-"llama3-8Bf"}

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

python tools/checkpoint/convert.py \
    --bf16 --model-type GPT --checkpoint-type hf \
    --loader llama_mistral --saver mcore \
    --target-tensor-parallel-size 8 \
    --model-size $MEGATRON_MODEL_SIZE \
    --load-dir $DOWNLOAD_DIR \
    --save-dir $PRETRAINED_DIR \
    --tokenizer-model $DOWNLOAD_DIR

DOCKER_SCRIPT

docker pull $PYTORCH_DOCKER_IMAGE
docker run --gpus all --shm-size 256G --rm \
    -v $MEGATRON_DIR:/workspace/megatron \
    -v $DATASETS_DIR:/workspace/datasets \
    -v $CHECKPOINTS_DIR:/workspace/checkpoints \
    $PYTORCH_DOCKER_IMAGE \
    bash -c "$RUN_DOCKER_COMMAND"