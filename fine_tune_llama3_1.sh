#!/bin/bash

export PYTORCH_DOCKER_IMAGE=${PYTORCH_DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:24.10-py3"}
export HUGGINGFACE_ACCESS_TOKEN=${HUGGINGFACE_ACCESS_TOKEN:-"your-token"}

export DOWNLOAD_MODEL_NAME=${DOWNLOAD_MODEL_NAME:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
export DOWNLOAD_DIR=${DOWNLOAD_DIR:-"Meta-Llama-3.1-8B-Instruct"}
export PRETRAINED_DIR=${PRETRAINED_DIR:-"Meta-Llama-3.1-8B-Instruct-Megatron"}

export MEGATRON_DIR=${MEGATRON_DIR:-"/home/ubuntu/Megatron-LM"}
export DATASETS_DIR=${DATASETS_DIR:-"/home/ubuntu/datasets"}
export CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-"/home/ubuntu/llama-checkpoints"}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"/home/ubuntu/llama-tensorboard"}

export DATASET_NAME=${DATASET_NAME:-"large_web_dataset"}

export SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-16384}
export TRAIN_STEPS=${TRAIN_STEPS:-10000}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}

export PYTHON_REQUIREMENTS=(
    transformers
    accelerate
    huggingface-hub
    setuptools==69.5.1
)

GPUS_PER_NODE=8

MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT 
)

LLAMA_MODEL_ARGS=(
    --seq-length $SEQUENCE_LENGTH 
    --max-position-embeddings 131072 
    --tokenizer-type HuggingFaceTokenizer 
    --tokenizer-model $DOWNLOAD_DIR 
    --tensorboard-dir /workspace/tensorboard 
    --save /workspace/checkpoints 
    --load $PRETRAINED_DIR 
    --exit-on-missing-checkpoint 
    --use-checkpoint-args 
    --no-load-optim 
    --no-load-rng 
    --finetune 
    --untie-embeddings-and-output-weights 
    --normalization RMSNorm 
    --position-embedding-type rope 
    --no-masked-softmax-fusion 
    --attention-softmax-in-fp32 
    --disable-bias-linear 
    --transformer-impl transformer_engine 
    --group-query-attention 
    --num-query-groups 4 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --rotary-base 500000 
    --rotary-percent 1.0 
    --use-rope-scaling 
    --num-layers 32 
    --hidden-size 4096 
    --ffn-hidden-size 14336 
    --num-attention-heads 32 
    --swiglu --bf16 
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size $GLOBAL_BATCH_SIZE 
    --train-iters $TRAIN_STEPS 
    --lr-decay-iters $TRAIN_STEPS  
    --lr-warmup-fraction 0.001 
    --lr-decay-style cosine 
    --lr 3.0e-5 
    --min-lr 3.0e-6 
    --weight-decay 0.001
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 8 
	--pipeline-model-parallel-size 1 
    --recompute-granularity full 
    --recompute-method uniform 
    --recompute-num-layers 8
)

DATA_ARGS=(
    --data-path /workspace/datasets/$DATASET_NAME 
    --split 949,50,1 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100 
    --save-interval 1000 
    --eval-interval 1000 
    --eval-iters 10 
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

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

DOCKER_SCRIPT

docker pull $PYTORCH_DOCKER_IMAGE
docker run --gpus all --shm-size 256G --rm \
    -v $MEGATRON_DIR:/workspace/megatron \
    -v $DATASETS_DIR:/workspace/datasets \
    -v $CHECKPOINTS_DIR:/workspace/checkpoints \
    -v $TENSORBOARD_DIR:/workspace/tensorboard \
    $PYTORCH_DOCKER_IMAGE \
    bash -c "$RUN_DOCKER_COMMAND"