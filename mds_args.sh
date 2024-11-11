#!/bin/bash

## COMMUNICATION
TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)

## LIMIT GPU NUM (FOR 1-NODE EXPERIMENTS)
if [ ${SIZE:-"-1"} -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0
    NGPU_PER_HOST=1
    NGPUS=1
elif [ ${SIZE:-"-1"} -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=0,1
    NGPU_PER_HOST=2
    NGPUS=2
fi

## HELPFUL FOR DEBUGGING THROUGH PRINTING OUT GRADIENTS  
if [ ${DEBUG:-""} == "SP" ]; then
    export SP=${SP:-4}
    export DEBUG_FNAME=debug/output_SP.txt
    # export DEBUG_FNAME=None
    > $DEBUG_FNAME

elif [ ${DEBUG:-""} == "DP" ]; then
    export CUDA_VISIBLE_DEVICES=0
    NGPU_PER_HOST=1
    NGPUS=1
    export DEBUG_FNAME=debug/output_DP.txt
    # export DEBUG_FNAME=None
    > $DEBUG_FNAME
fi

export SP=${SP:-1}
export PP=${PP:-1}
export TP=${TP:-1}
if [ $PP -eq 1 ]; then 
    export no_pipeline_parallel=--no-pipeline-parallel
fi

export TSTAMP="${TSTAMP}"
export NHOSTS="${NHOSTS}"
export NGPU_PER_HOST="${NGPU_PER_HOST}"
export PROJECT="datascience"
export NGPUS=$(($NHOSTS * $NGPU_PER_HOST))

## DATA
export DATA=${DATA:-'CIFAR'}

## SET PARALLELISM DEGREES AS ENV VAR AND FOR DS ARGS
MP=$(($SP * $TP * $PP))
DP=$(($NGPUS / $MP))
if [[ $GBS ]]; then
    MBS=$(($GBS / $DP))
elif [[ $MBS ]]; then
    MBS=$(($MBS * $MP)) ## Maintain GBS
    GBS=$(($MBS * $DP)) 
else
    printf "\nERROR: you need to pass in either MBS or GBS\n"; exit 1
fi

## TODO: download and parse data if eagle is unavailable. 
AEVARD_PATH=/eagle/datascience/vsastry/from_andre/aevard/datasets
EKU_PATH=/eagle/datascience/eku/data
if [[ $DATA == 'IMNET' ]]; then
    echo "TRAINING ON IMNET"
    # DATA_PATH="~/aevard/datasets/imnet-20/train ~/aevard/datasets/imnet-20/valid"
    DATA_PATH="$AEVARD_PATH/imnet-20/train $AEVARD_PATH/imnet-20/valid"
    NUM_CLASSES=20
    LR=1e-4
    WEIGHT_DECAY=0
    PATCH_DIM=16
    IMG_W=224
    IMG_H=224

    ## DATA
    NUM_EPOCHS=100
    TRAIN_SIZE=24912
    TRAIN_SAMPLES=$(($NUM_EPOCHS * $TRAIN_SIZE)) ##TODO: Why does IMNET only have 24912 image samples? 
    LR_WARMUP_SAMPLES=1000
    DS_CONFIG_FNAME="IMNET.json"

elif [[ $DATA == 'CIFAR' ]]; then
    echo "TRAINING ON CIFAR"
    DATA_PATH="$EKU_PATH/CIFAR10/train $EKU_PATH/CIFAR10/valid"
    NUM_CLASSES=10
    LR=1e-4
    WEIGHT_DECAY=0
    PATCH_DIM=4
    size_factor=1
    IMG_W=32
    IMG_H=32

    ## DATA
    NUM_EPOCHS=${NUM_EPOCHS:-500}
    TRAIN_SIZE=40000
    EVAL_ITERS=19 ##TODO: Val samples?
    # EVAL_ITERS=$((10000 / 512)) ##TODO: Val samples?
    TRAIN_SAMPLES=$(($NUM_EPOCHS * $TRAIN_SIZE))
    LR_WARMUP_SAMPLES=0
    DS_CONFIG_FNAME="CIFAR.json"

elif [[ $DATA == 'TOY' ]]; then
    echo "TRAINING ON TOY DATASET"
    ##Toy Dataset
    # DATA_PATH="~/aevard/datasets/CIFAR10/train ~/aevard/datasets/CIFAR10/valid"
    DATA_PATH="$EKU_PATH/CIFAR10/train $EKU_PATH/CIFAR10/valid"
    NUM_CLASSES=20
    PATCH_DIM=16
    factor=${factor:-54}

    IMG_W=$(($PATCH_DIM * $factor))
    IMG_H=$(($PATCH_DIM * $factor))
    IMG_D=$(($PATCH_DIM * $factor)) ## image depth for 3dvit.
    NUM_CHANNELS=1 ## 1 for 3dvit, default=3 for 2dvit.
    LR_WARMUP_SAMPLES=0

    ## DATA
    DS_CONFIG_FNAME="TOY.json"
else
    echo "Dataset not implemented"
    exit 1
fi

if [[ $VIT3D ]] && [[ $DATA -ne "TOY" ]]; then
    echo "Currently 3dvit is only supported with Toy dataset"
    exit 1
fi

if [[ $NUM_ITERS ]]; then
    TRAIN_SAMPLES=$(($NUM_ITERS * $GBS))
fi

if [[ -z $ZERO ]]; then
    export ZERO=0
fi

cat <<EOF > "$DS_CONFIG_FNAME"
{
    "train_micro_batch_size_per_gpu": $MBS,
    "steps_per_print": 9999999999,
    "gradient_accumulation_steps": 1,
    "zero_allow_untested_optimizer": false,
    "gradient_clipping": 1.0,
    "communication_data_type": "fp16",
    "fp16": {
                "enabled": true,
                "loss_scale": 0
            },
    "wall_clock_breakdown": false,
    "logging_level": "WARNING",
    "comms_logger": {
                        "enabled": false,
                        "verbose": false,
                        "prof_all": true,
                        "debug": false
                    },
    "flops_profiler": {
                        "enabled": false,
                        "profile_step": 10,
                        "module_depth": -1,
                        "top_modules": 1,
                        "detailed": false,
                        "output_file": null
                        },
    "zero_optimization": {
        "stage": $ZERO
    }
}
EOF
## TODO: add optimal activation_checkpointing config
#  "activation_checkpointing": {
#     "partition_activations": false,
#     "cpu_checkpointing": false,
#     "contiguous_memory_optimization": false,
#     "number_checkpoints": null,
#     "synchronize_checkpoint_boundary": false,
#     "profile": false
#     }

## MODEL CONFIGURATION ##

VIT=${VIT:-"Large"}
echo Using VIT-$VIT
if [[ $VIT == "Tiny" ]]; then
    ## ViT-Tiny (10M)
    NLAYERS=6
    HSIZE=512
    FFN_HSIZE=512
    NUM_HEADS=8
elif [[ $VIT == "Base" ]]; then
    ## ViT-Base (86M)
    NLAYERS=12
    HSIZE=768
    FFN_HSIZE=3072
    NUM_HEADS=12
elif [[ $VIT == "Large" ]]; then
    ## VIT-Large (307M)
    NLAYERS=24
    HSIZE=1024
    FFN_HSIZE=4096
    NUM_HEADS=16
elif [[ $VIT == "Huge" ]]; then
    ## VIT-Huge (632M)
    NLAYERS=32
    HSIZE=1280
    FFN_HSIZE=5120
    NUM_HEADS=16
elif [[ $VIT == "2B" ]]; then
    ## VIT-2B (1.6B in VIT? Why doesn't it fit?)
    echo TBD
    exit 1
    # NLAYERS=10
    # HSIZE=4096
    # FFN_HSIZE=11008
    # HSIZE=16384
    # FFN_HSIZE=16384
    # NUM_HEADS=32
    # ATT_DROPOUT=0.1
    # H_DROPOUT=0.1
else
    echo "VIT not implemented"
    exit 1
fi

## EXPORT
export TRAIN_SAMPLES="${TRAIN_SAMPLES:-5000}"
export EVAL_ITERS="${EVAL_ITERS:-1000}"
export LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-250}"
export EVAL_INTERVAL=${EVAL_INTERVAL:-250}
export DATA_PATH="${DATA_PATH}"
export DATA=$DATA
export GBS=$GBS
export MBS=$MBS
export NUM_CLASSES=$NUM_CLASSES
export IMG_H="${IMG_H}"
export IMG_W="${IMG_W}"
export IMG_D="${IMG_D}"
export PATCH_DIM="${PATCH_DIM}"
export LR="${LR:-1e-4}"
export MIN_LR="${MIN_LR:-0.00001}"
export NLAYERS="${NLAYERS}"
export HSIZE="${HSIZE}"
export NUM_HEADS="${NUM_HEADS}"

## EXPERIMENTAL (This somehow fixes the OOM issue for Ring-Att?)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torch.distributed.DistBackendError: NCCL error in: /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1970, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.20.5
# [rank0]: ncclUnhandledCudaError: Call to CUDA function failed.
# export NCCL_DEBUG=INFO

if [[ $VIT3D ]]; then
    SEQ_LEN=$((IMG_H * IMG_W * IMG_D / PATCH_DIM**3))  
else
    SEQ_LEN=$((IMG_H * IMG_W / PATCH_DIM**2))  
fi
if [[ -z $GLOBAL_MEAN_POOLING ]]; then
    SEQ_LEN=$((SEQ_LEN + 1)) ## Count clf token in seq length. 
fi 
export SEQ_LEN=$SEQ_LEN
## TODO: update seq_len when you add the padded tokens features. 
echo "Sequence length: ${SEQ_LEN}"

## LOGGING
RUN_NAME="N${NUM_NODES}-${TSTAMP}"
RUN_NAME="VIT-CLASS-${RUN_NAME}"
export RUN_NAME="${RUN_NAME}"