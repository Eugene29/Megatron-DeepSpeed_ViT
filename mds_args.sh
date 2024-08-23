#!/bin/bash

# Launch-defined arguments.
TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
# NGPU_PER_HOST=1

# export WORLD_SIZE=1
export TSTAMP="${TSTAMP}"
export NHOSTS="${NHOSTS}"
export NGPU_PER_HOST="${NGPU_PER_HOST}"
export PROJECT="datascience"
NGPUS=$((${NHOSTS}*${NGPU_PER_HOST}))
export NGPUS="${NGPUS}"


# Run architecture arguments.
TRAIN_ITERS=25000
EVAL_ITERS=500

if [[ $PYTHONPATH == *"outputs"* ]]; then
    DATA_PATH=~/aevard/datasets/imnet-20
else
    DATA_PATH="~/aevard/datasets/imnet-20/train ~/aevard/datasets/imnet-20/valid"
fi

export TRAIN_ITERS="${TRAIN_ITERS}"
export EVAL_ITERS="${EVAL_ITERS}"
export DATA_PATH="${DATA_PATH}"


# IMG Resolution (i.e. Seq Length)
PATCH_DIM=16
factor=253
IMG_W=$(($PATCH_DIM * $factor))
IMG_H=$(($PATCH_DIM * $factor))
NUM_CLASSES=20

LR=0.01
MIN_LR=0.00001

## ViT-Base - 84M
NLAYERS=8
HSIZE=1024
NUM_HEADS=16

## ViT-Large - 671M
# NLAYERS=16
# HSIZE=2048
# NUM_HEADS=32

export IMG_H="${IMG_H}"
export IMG_W="${IMG_W}"
export PATCH_DIM="${PATCH_DIM}"
export LR="${LR}"
export MIN_LR="${MIN_LR}"
export NLAYERS="${NLAYERS}"
export HSIZE="${HSIZE}"
export SEQ_LEN=$(echo "${IMG_W} * ${IMG_W} / ${PATCH_DIM}^2 + 1" | bc) ## This doesn't matter. It is calculated automatically anyway.  
echo "Sequence length: ${SEQ_LEN}"
# export MICRO_BATCH="${MICRO_BATCH}"
export NUM_HEADS="${NUM_HEADS}"


RUN_NAME="N${NUM_NODES}-${TSTAMP}"
# RUN_NAME="mb${MICRO_BATCH}-gas${GAS}-${RUN_NAME}"
RUN_NAME="VIT-CLASS-${RUN_NAME}"
export RUN_NAME="${RUN_NAME}"

export SP=1
export PP=1
export TP=1

if [ $PP -eq 1 ]; then
    export no_pipeline_parallel=--no-pipeline-parallel
fi
# export GLOBAL_BATCH=$(echo "$MICRO_BATCH * $NGPUS / $PP / $SP / $TP" | bc)
export WANDB_DISABLED=1