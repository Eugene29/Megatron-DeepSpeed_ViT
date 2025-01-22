#!/bin/bash

## COMMUNICATION
NHOSTS=$(wc -l < "${PBS_NODEFILE}")

## LIMIT GPUs VISIBLE (FOR 1-NODE EXPERIMENTS)
if [ ${SIZE:-"-1"} -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0
    ZE_AFFINITY_MASK=0
    NGPU_PER_HOST=1
    NGPUS=1
elif [ ${SIZE:-"-1"} -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=0,1
    ZE_AFFINITY_MASK=0,1
    NGPU_PER_HOST=2
    NGPUS=2
fi

## DEBUG VARIABLES
# HELPFUL FOR DEBUGGING THROUGH PRINTING OUT GRADIENTS  
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
# If DATA_PATH_LOG is passed, will record input tensors consumed
if [[ $DATA_PATH_LOG ]]; then
     > $DATA_PATH_LOG 
fi

export SP=${SP:-1}
export PP=${PP:-1}
export TP=${TP:-1}
export NHOSTS="${NHOSTS}"
export NGPU_PER_HOST="${NGPU_PER_HOST}"
export PROJECT="datascience"
export NGPUS=$(($NHOSTS * $NGPU_PER_HOST))
if [ $PP -eq 1 ]; then 
    export no_pipeline_parallel=--no-pipeline-parallel
fi

## DATA
export DATA=${DATA:-'CIFAR'}

## SET PARALLELISM DEGREES AS ENV VAR AND FOR DS ARGS
MP=$(($SP * $TP * $PP))
DP=$(($NGPUS / $MP))
if [[ $GBS ]]; then
    MBS=$(($GBS / $DP))
elif [[ $MBS ]]; then
    # MBS=$(($MBS * $MP)) ## Maintain GBS across DP and MP
    GBS=$(($MBS * $DP)) 
else
    printf "\nERROR: you need to pass in either MBS or GBS\n"; exit 1
fi
 
if [[ $DATA == 'IMNET' ]]; then
    echo "Not Implemented Error"
    exit 1
    # DATA_PATH="~/aevard/datasets/imnet-20/train ~/aevard/datasets/imnet-20/valid"
    # DATA_PATH="$AEVARD_PATH/imnet-20/train $AEVARD_PATH/imnet-20/valid"
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
    DATA_PATH="$DATA_DIR/CIFAR10/train $DATA_DIR/CIFAR10/valid"
    NUM_CLASSES=10
    LR=1e-4
    WEIGHT_DECAY=0
    PATCH_DIM=4
    # size_factor=1
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
    DATA_PATH="$DATA_DIR/CIFAR10/train $DATA_DIR/CIFAR10/valid" ## Dummy data path
    NUM_CLASSES=20
    PATCH_DIM=16
    factor=${factor:-54}
    NUM_EPOCHS=${NUM_EPOCHS:-500}
    TRAIN_SIZE=40000 ## Dummy Size
    TRAIN_SAMPLES=$(($NUM_EPOCHS * $TRAIN_SIZE))

    IMG_W=$(($PATCH_DIM * $factor))
    IMG_H=$(($PATCH_DIM * $factor))
    IMG_D=$(($PATCH_DIM * $factor)) ## image depth for 3dvit, not used if 2dvit
    NUM_CHANNELS=${NUM_CHANNELS:-1} ## 1 for 2d, 3dvit (TOY).
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

export ZERO=${ZERO:-0}
export hpz=${hpz:-1}

cat <<EOF > "$WORKING_DIR/$DS_CONFIG_FNAME"
{
  "train_batch_size": $GBS,
  "train_micro_batch_size_per_gpu": $MBS,
  "steps_per_print": 10,

  "zero_optimization": {
    "stage": $ZERO,
    "overlap_comm": true,
    "zero_hpz_partition_size": $hpz,
    "contiguous_gradients": true
  },

  "gradient_clipping": 1.0,
  "prescale_gradients": false,

  "communication_data_type": "bf16",
  "bf16": {
    "enabled": true
  },

  "wall_clock_breakdown" : false
}
EOF

## fp16
#   "communication_data_type": "fp16",
#   "fp16": {
#     "enabled": true,
#     "loss_scale": 0,
#     "loss_scale_window": 500,
#     "hysteresis": 2,
#     "min_loss_scale": 1,
#     "initial_scale_power": 11
#   },

## broken
#   "data_types": {
#     "grad_accum_dtype": "fp32" 
#   },

## prevents comm overlap
# ,
#   "comms_logger": {
#     "enabled": true,
#     "verbose": false,
#     "prof_all": true,
#     "debug": false
#   }

## Below configs seems to not do anything
#  "activation_checkpointing": {
#     "partition_activations": false,
#     "cpu_checkpointing": false,
#     "contiguous_memory_optimization": false,
#     "number_checkpoints": null,
#     "synchronize_checkpoint_boundary": false,
#     "profile": false
#     }

# cat <<EOF > "$DS_CONFIG_FNAME"
# {
#     "train_micro_batch_size_per_gpu": $MBS,
#     "steps_per_print": 9999999999,
#     "gradient_accumulation_steps": 1,
#     "zero_allow_untested_optimizer": true,
#     "gradient_clipping": 1.0,
#     "communication_data_type": "fp16",
#     "fp16": {
#                 "enabled": true,
#                 "loss_scale": 0
#             },
#     "wall_clock_breakdown": false,
#     "logging_level": "WARNING",
#     "flops_profiler": {
#                         "enabled": false,
#                         "profile_step": 10,
#                         "module_depth": -1,
#                         "top_modules": 1,
#                         "detailed": false,
#                         "output_file": null
#                         },
#     "zero_optimization": {
#         "stage": $ZERO,
#         "overlap_comm": true
#     },
#     "comms_logger": {
#         "enabled": true,
#         "verbose": false,
#         "prof_all": true,
#         "debug": false
#     }
# }
# EOF
        # "contiguous_gradients": true,
        # "reduce_scatter": true,
        # "allgather_partitions": true,
        # "mics_hierarchical_params_gather": true

## MODEL CONFIGURATION ##
export VIT=${VIT:-"LARGE"}
echo Using VIT-$VIT
# ATT_DROPOUT=0.1
# H_DROPOUT=0.1
if [[ $VIT == "TINY" ]]; then
    ## ViT-TINY (10M)
    NLAYERS=6
    HSIZE=512
    FFN_HSIZE=512
    NUM_HEADS=8
    # PATCH_DIM=4
elif [[ $VIT == "BASE" ]]; then
    ## ViT-BASE (86M)
    NLAYERS=12
    HSIZE=768
    FFN_HSIZE=3072
    NUM_HEADS=12
elif [[ $VIT == "LARGE" ]]; then
    ## VIT-LARGE (307M)
    NLAYERS=24
    HSIZE=1024
    FFN_HSIZE=4096
    NUM_HEADS=16
elif [[ $VIT == "LARGE+" ]]; then
    ## VIT-LARGE (307M)
    NLAYERS=24
    HSIZE=1032
    FFN_HSIZE=4096
    NUM_HEADS=12
elif [[ $VIT == "HUGE" ]]; then
    ## VIT-HUGE (632M)
    NLAYERS=32
    HSIZE=1280
    FFN_HSIZE=5120
    NUM_HEADS=16
elif [[ $VIT == "GIANT" ]]; then
    ## ?B
    NLAYERS=48
    HSIZE=1664
    FFN_HSIZE=8192
    NUM_HEADS=16
elif [[ $VIT == "ENORMOUS" ]]; then
    ## ?B
    NLAYERS=56
    HSIZE=1792
    FFN_HSIZE=15360
    NUM_HEADS=16
elif [[ $VIT == "2B" ]]; then
    ## 1.6B
    NLAYERS=10
    HSIZE=4096
    NUM_HEADS=32
    FFN_HSIZE=11008
elif [[ $VIT == "3B" ]]; then
    ## 2.7B
    NLAYERS=24
    HSIZE=3072
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=32
elif [[ $VIT == "4B" ]]; then
    ## 3.8B
    NLAYERS=48
    HSIZE=2560
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=32
elif [[ $VIT == "5B" ]]; then
    ## 5B
    NLAYERS=28
    HSIZE=$((64 * 60))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=32
elif [[ $VIT == "5.6B" ]]; then
    ## 5.6B
    NLAYERS=28
    HSIZE=$((64 * 64))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=32
elif [[ $VIT == "6B" ]]; then
    ## 6.4B
    NLAYERS=32
    HSIZE=4096
    NUM_HEADS=32
    FFN_HSIZE=$((4 * HSIZE))
elif [[ $VIT == "8B" ]]; then
    ## 8.2B
    NLAYERS=32
    HSIZE=$((64 * 72))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=36
elif [[ $VIT == "9B" ]]; then
    ## 9.2B
    NLAYERS=36
    HSIZE=$((64 * 72))
    FFN_HSIZE=$((4*HSIZE))
    NUM_HEADS=32
elif [[ $VIT == "13B" ]]; then
    ## GPT-3 13B in VIT 12.6B (?)
    NLAYERS=40
    HSIZE=5120
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=40
elif [[ $VIT == "14B" ]]; then
    ## 13.8
    NLAYERS=44
    HSIZE=$((64 * 80))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=40
elif [[ $VIT == "17B" ]]; then
    ## 16.7
    NLAYERS=44
    HSIZE=$((64 * 88))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=44
elif [[ $VIT == "20B" ]]; then
    ## 19.9
    NLAYERS=44
    HSIZE=$((64 * 96))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=48
elif [[ $VIT == "22B" ]]; then
    ## 21.8B
    NLAYERS=48
    HSIZE=6144
    FFN_HSIZE=24576
    NUM_HEADS=48
elif [[ $VIT == "25B" ]]; then
    ## 24.5B
    NLAYERS=48
    HSIZE=$((64 * 102))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=48
elif [[ $VIT == "26B" ]]; then
    ## 25.5B
    NLAYERS=49
    HSIZE=$(( 64 * 104 ))
    FFN_HSIZE=$(( 4 * HSIZE ))
    NUM_HEADS=52
elif [[ $VIT == "28B" ]]; then
    ## 27.6B
    NLAYERS=50
    HSIZE=$(( 64 * 106 ))
    FFN_HSIZE=$(( 4 * HSIZE ))
    NUM_HEADS=53
elif [[ $VIT == "30B" ]]; then
    ## 29.7B
    NLAYERS=50
    HSIZE=$((64 * 110))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=64
elif [[ $VIT == "30+B" ]]; then
    ## ??
    NLAYERS=50
    HSIZE=$((64 * 118))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=64
elif [[ $VIT == "30++B" ]]; then
    ## ??
    NLAYERS=50
    HSIZE=$((64 * 124))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=64
elif [[ $VIT == "42B" ]]; then
    ## 42.4B
    NLAYERS=51
    HSIZE=$((64 * 130))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=64
elif [[ $VIT == "42+B" ]]; then
    ## 42.4B
    NLAYERS=51
    HSIZE=8340
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=60
elif [[ $VIT == "46B" ]]; then
    ## 46.4B
    NLAYERS=51
    HSIZE=$((64 * 136))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=64
elif [[ $VIT == "46+B" ]]; then
    ## ??
    NLAYERS=54
    HSIZE=$((64 * 144))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=64
elif [[ $VIT == "59B" ]]; then
    ## 70B in GPT; 59B in VIT?
    NLAYERS=80
    HSIZE=8192
    FFN_HSIZE=28672
    NUM_HEADS=64
elif [[ $VIT == "112B" ]]; then
    ## 112
    NLAYERS=56
    HSIZE=$((64 * 202))
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=64
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
export NUM_CHANNELS=$NUM_CHANNELS
export IMG_H="${IMG_H}"
export IMG_W="${IMG_W}"
export IMG_D="${IMG_D}"
export PATCH_DIM="${PATCH_DIM}"
export LR="${LR:-1e-4}"
export MIN_LR="${MIN_LR:-0.00001}"
export NLAYERS="${NLAYERS}"
export HSIZE="${HSIZE}"
export NUM_HEADS="${NUM_HEADS}"

if [[ $VIT3D ]]; then
    SEQ_LEN=$((IMG_H * IMG_W * IMG_D / PATCH_DIM**3))  
else
    SEQ_LEN=$((IMG_H * IMG_W / PATCH_DIM**2))  
fi
if [[ -z $GLOBAL_MEAN_POOLING ]]; then
    SEQ_LEN=$((SEQ_LEN + 1)) ## Count clf token in seq length. 
fi 
export SEQ_LEN=$SEQ_LEN
echo "Sequence length: ${SEQ_LEN}"