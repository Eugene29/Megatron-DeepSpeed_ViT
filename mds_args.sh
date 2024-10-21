#!/bin/bash


## COMMUNICATION
TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)


if [ ${SIZE:-"-1"} -eq 1 ]; then
    ## CUDA DEVICE (for experiments)
    CUDA_VISIBLE_DEVICES=0
    NGPU_PER_HOST=1
    NGPUS=1
elif [ ${SIZE:-"-1"} -eq 2 ]; then
    ## CUDA DEVICE (for experiments)
    CUDA_VISIBLE_DEVICES=0,1
    NGPU_PER_HOST=2
    NGPUS=2
fi

# export WANDB_MODE="disabled"
# DEBUG=
if [ ${DEBUG:-""} == "SP" ]; then
    ## CUDA DEVICE (for experiments)
    # export CUDA_VISIBLE_DEVICES=0,1
    # NGPU_PER_HOST=2
    # NGPUS=2
    ## SP
    ## PARALLELIZATION
    export SP=${SP:-4} ## 1 if the var is not instantiated by mds_submit
    export DEBUG_FNAME=debug/output_SP.txt
    # export DEBUG_FNAME=None
    > $DEBUG_FNAME

elif [ ${DEBUG:-""} == "DP" ]; then
    export CUDA_VISIBLE_DEVICES=0
    NGPU_PER_HOST=1
    NGPUS=1
    ## PARALLELIZATION
    ## TODO: change this into more readable format with string.
    export DEBUG_FNAME=debug/output_DP.txt
    # export DEBUG_FNAME=None
    > $DEBUG_FNAME
fi

export SP=${SP:-1} ## 1 if the var is not instantiated by mds_submit
export PP=${PP:-1}
export TP=${TP:-1}
if [ $PP -eq 1 ]; then 
    export no_pipeline_parallel=--no-pipeline-parallel
fi

##TODO: ORGANIZE GIT COMMIT COMMANDS. 
export TSTAMP="${TSTAMP}"
export NHOSTS="${NHOSTS}"
export NGPU_PER_HOST="${NGPU_PER_HOST}"
export PROJECT="datascience"
export NGPUS=$(($NHOSTS * $NGPU_PER_HOST))

## DATA
export DATA=${DATA:-'CIFAR'}
# export DATA=${DATA:-'CIFAR'}

## SET PARALLELISM DEGREES
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

AEVARD_PATH=/eagle/datascience/vsastry/from_andre/aevard/datasets
EKU_PATH=/eagle/datascience/eku/data/
if [[ $DATA == 'IMNET' ]]; then
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

    NLAYERS=12
    HSIZE=1024
    FFN_HSIZE=1024
    NUM_HEADS=16
    ATT_DROPOUT=0.1
    H_DROPOUT=0.1
    echo "TRAINING ON IMNET"

elif [[ $DATA == 'CIFAR' ]]; then
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
    # TRAIN_SAMPLES=$((10 * 512)) ## TODO: ENABLE FOR PROFILING (5 steps)
    LR_WARMUP_SAMPLES=500
    DS_CONFIG_FNAME="CIFAR.json"

    ## ViT-Tiny
    NLAYERS=6
    HSIZE=512
    FFN_HSIZE=512
    NUM_HEADS=8
    # ATT_DROPOUT=0.1
    # H_DROPOUT=0.1

    # ## Test VIT Large (my version)
    # if [ -n "$PROFILE" ]; then
    # NLAYERS=12
    # HSIZE=2048
    # FFN_HSIZE=2048
    # NUM_HEADS=16
    # fi

    # ATT_DROPOUT=0.1
    # H_DROPOUT=0.1
    ## EOF is super weird but correct. 
    echo "TRAINING ON CIFAR"

elif [[ $DATA == 'Toy' ]]; then
    ##Toy Dataset
    # DATA_PATH="~/aevard/datasets/CIFAR10/train ~/aevard/datasets/CIFAR10/valid"
    DATA_PATH="$EKU_PATH/CIFAR10/train $EKU_PATH/CIFAR10/valid"
    NUM_CLASSES=20
    PATCH_DIM=16
    # factor=2
    # factor=94
    # factor=215
    factor=${factor:-54}

    IMG_W=$(($PATCH_DIM * $factor))
    IMG_H=$(($PATCH_DIM * $factor))
    # IMG_W=$($IMG_SIZE:-$IMG_W)
    # IMG_H=$($IMG_SIZE:-$IMG_H)

    # TRAIN_SAMPLES=$(($NUM_EPOCHS * 10))
    # LR_WARMUP_SAMPLES=1
    ## TODO: make this generalized? 
    # echo $TRAIN_SAMPLES
    # exit 1

    LR_WARMUP_SAMPLES=0

    ## DATA
    DS_CONFIG_FNAME="Toy.json"

    ## ViT-Tiny (1M)
    # NLAYERS=6
    # HSIZE=512
    # FFN_HSIZE=512
    # NUM_HEADS=8

    ## VIT-Large (307M)
    NLAYERS=24
    HSIZE=1024
    FFN_HSIZE=4096
    NUM_HEADS=16
    # NLAYERS=30
    # HSIZE=2048
    # FFN_HSIZE=8192

    ## VIT-2B (1.6B in VIT? Why doesn't it fit?)
    # NLAYERS=10
    # NLAYERS=5
    # HSIZE=4096
    # FFN_HSIZE=11008
    # HSIZE=16384
    # FFN_HSIZE=16384
    # NUM_HEADS=32

    # ATT_DROPOUT=0.1
    # H_DROPOUT=0.1
    echo "TRAINING ON TOYDATASET"
fi

if [[ $NUM_ITER ]]; then
    TRAIN_SAMPLES=$(($NUM_ITER * $GBS))
# elif [[ $NUM_EPOCHS ]]; then
#     TRAIN_SAMPLES=$(($NUM_EPOCHS * $DP))
fi

# echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# echo NGPU_PER_HOST: $NGPU_PER_HOST
# echo NGPUS: $NGPUS
# echo ""
# echo GBS: $GBS
# echo MBS: $MBS
# echo DP: $DP
# echo MP: $MP
# exit 1
# if [ -n "$TRAIN_ITERS" ]; then
#     TRAIN_SAMPLES=$(($DP * $TRAIN_ITERS))
# fi


cat <<EOF > $DS_CONFIG_FNAME
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
                        }
}
EOF
    # "activation_checkpointing": {
    #     "partition_activations": true,
    #     "contiguous_memory_optimization": true
    # }

## OVERWRITE CONFIGS (DEBUG)
# TRAIN_SAMPLES=500
# TRAIN_SAMPLES=512000
# LR_WARMUP_SAMPLES=10

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

## ARCHITECTURE
##TODO: VIT+SP+FA has randomness issue that worsens with respect to the sequence length. 

## ViT-Base - 84M
# NLAYERS=8
# HSIZE=1024
# NUM_HEADS=16 #?

## ViT-Large - 671M
# NLAYERS=16
# HSIZE=2048
# NUM_HEADS=32

export IMG_H="${IMG_H}"
export IMG_W="${IMG_W}"
export PATCH_DIM="${PATCH_DIM}"
export LR="${LR:-1e-4}"
export MIN_LR="${MIN_LR:-0.00001}"
export NLAYERS="${NLAYERS}"
export HSIZE="${HSIZE}"
SEQ_LEN=$(echo "${IMG_W} * ${IMG_W} / ${PATCH_DIM}^2" | bc)  
if [ -z $GLOBAL_MEAN_POOLING ]; then
    SEQ_LEN=$((SEQ_LEN + 1))
fi
export SEQ_LEN
echo "Sequence length: ${SEQ_LEN}"
export NUM_HEADS="${NUM_HEADS}"

## LOGGING
RUN_NAME="N${NUM_NODES}-${TSTAMP}"
RUN_NAME="VIT-CLASS-${RUN_NAME}"
export RUN_NAME="${RUN_NAME}"