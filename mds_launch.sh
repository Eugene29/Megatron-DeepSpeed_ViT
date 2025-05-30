#! /bin/bash

## ENVIRONMENT
echo "Launching Megatron Deepspeed VIT."
TZ="America/Chicago" date

## COMMUNICATION
NHOSTS=$(wc -l < "${PBS_NODEFILE}")

get_machine() {
    machine=$(hostname)
    if [[ $(hostname) == x4* ]]; then
        machine="aurora"
    elif [[ $(hostname) == x1* ]]; then
        machine="sunspot"
    elif [[ $(hostname) == x3* ]]; then
        if [[ "${PBS_O_HOST}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
    elif [[ $(hostname) == sophia* ]]; then
        machine="sophia"
    elif [[ $(hostname) == nid* ]]; then
        machine="perlmutter"
    else
        echo "Unknown MACHINE. Setting MACHINE to $(hostname) and continuing..."
    fi
    export MACHINE="${machine}"
    echo "Running on: $machine"
}
get_machine

if [[ $MACHINE == "aurora" ]]; then 
     WANDB_PROJECT_NAME="AuroraViT"
     # Below DATA_DIR is just a placeholder and will only be used if DATA is set to CIFAR. For benchmarking only purposes, it can be set to any existing filepath.
     DATA_DIR="/lus/flare/projects/Aurora_deployment/eku/data"
     ## env with ezpz, etc.
     . /lus/flare/projects/Aurora_deployment/eku/venv/vit/bin/activate
     FA_VERSION="--use-flash-attn-builder"
     NGPU_PER_HOST=12
     set_ccl_vars_on_aurora() {
          export CCL_KVS_MODE=mpi
          export CCL_CONFIGURATION_PATH=""
          export CCL_CONFIGURATION=cpu_gpu_dpcpp
          export CCL_KVS_CONNECTION_TIMEOUT=3600

          export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
          export CCL_PROCESS_LAUNCHER=pmix # Required by Aurora mpich
          if [[ $NHOSTS -gt 1 ]]; then
            ## the following code breaks for node=1
            export FI_PROVIDER=cxi           # Required by Aurora mpich
          fi
          export PALS_PMI=pmix             # Required by Aurora mpich
          export CCL_ATL_TRANSPORT=mpi     # Required by Aurora mpich
          export TORCH_LLM_ALLREDUCE=1
          export CCL_SYCL_ESIMD=1
          export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0 # Required by current oneCCL (MLSL-2881)
          export CCL_ENABLE_SYCL_KERNELS=1
          export CCL_WORKER_AFFINITY=5,13,21,29,37,45,57,65,73,81,89,97
          export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=32768
          export FI_CXI_DEFAULT_CQ_SIZE=1048576
          export FI_CXI_RX_MATCH_MODE=hybrid

          export CCL_ALLGATHERV=topo
          export CCL_ALLREDUCE=topo
          #  export CCL_BCAST=double_tree
          export CCL_BARRIER=ring
          export CCL_ALLREDUCE_SCALEOUT=ring
          #  export CCL_ALLREDUCE_SCALEOUT=rabenseifener
          export CCL_ALLGATHER_SCALEOUT=ring
          export CCL_ALLGATHERV_SCALEOUT=ring
     }

     set_ccl_vars_on_aurora2() {
        export CCL_KVS_MODE=mpi
        export CCL_KVS_CONNECTION_TIMEOUT=600 
        export PALS_PMI=pmix
        export CCL_ATL_TRANSPORT=mpi

        export TORCH_LLM_ALLREDUCE=1
        export CCL_SYCL_ESIMD=1
        export CCL_ATL_SYNC_COLL=1
        export CCL_OP_SYNC=1
        export CCL_ENABLE_AUTO_CACHE=0
        export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=4096

        export CCL_ALLREDUCE=topo
        export CCL_ALLGATHERV=direct
        export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0
        export CCL_ALLREDUCE_SCALEOUT=direct
        export CCL_BCAST=double_tree

        export FI_CXI_DEFAULT_CQ_SIZE=1048576
        export FI_CXI_RX_MATCH_MODE=hybrid
        export FI_MR_CACHE_MONITOR=disabled
        export FI_CXI_OFLOW_BUF_SIZE=8388608
        export FI_CXI_CQ_FILL_PERCENT=30

        export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
        export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
        export NUMEXPR_MAX_THREADS=7
        export OMP_NUM_THREADS=7

        export PALS_PING_PERIOD=480
        export PALS_RPC_TIMEOUT=480
     }
     set_ccl_vars_on_aurora2

elif [[ $MACHINE == "polaris" ]]; then 
     module load conda
     conda activate

     export AWS_DIR=/soft/libraries/aws-ofi-nccl/v1.6.0/
     export NCCL_NET_GDR_LEVEL=PHB
     export NCCL_CROSS_NIC=1
     export NCCL_COLLNET_ENABLE=1
     export NCCL_SOCKET_IFNAME=hsn
     export NCCL_NET="AWS Libfabric"
     export LD_LIBRARY_PATH=$AWS_DIR/lib:$LD_LIBRARY_PATH
     export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH

     export FI_CXI_DISABLE_HOST_REGISTER=1
     export FI_MR_CACHE_MONITOR=userfaultfd
     export FI_CXI_DEFAULT_CQ_SIZE=131072
     export FI_CXI_DEFAULT_TX_SIZE=131072
     export FI_CXI_RDZV_PROTO=alt_read
     export FI_CXI_RX_MATCH_MODE=software
     export FI_CXI_REQ_BUF_SIZE=16MB

     export FI_CXI_RDZV_GET_MIN=0
     export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
     export FI_CXI_RDZV_THRESHOLD=2000

     WANDB_PROJECT_NAME="PolarisViT"
     DATA_DIR="/eagle/datascience/eku/data"
     FA_VERSION="--use-flash-attn-v2"
     NGPU_PER_HOST=4
     ## EXPERIMENTAL (This somehow fixes the OOM issue for Ring-Att?)
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     # torch.distributed.DistBackendError: NCCL error in: /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1970, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.20.5
     # [rank0]: ncclUnhandledCudaError: Call to CUDA function failed.
     # export NCCL_DEBUG=INFO
else
     #### CUSTOMIZE HERE ####
     echo "Not Impelmented Error for $MACHINE Machine. Manually set env variables";
     exit 1
     NGPU_PER_HOST="<number of GPUs per node>"
     NHOSTS="<number of nodes>"
fi

## PYTHONPATH 
# NOTE: Yunchang is not used for ALCF4 Benchmark
WORKING_DIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
# cd $WORKING_DIR
# YUNCHANG="${WORKING_DIR}/long-context-attention" ## Custom yunchang (USP)
# PYTHONPATH="${DEEPSPEED}:${YUNCHANG}:${PYTHONPATH}"
export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH}" ## Add local megatron path
## HOST NODE
export MASTER_ADDR=$(hostname)
export MASTER_PORT=6000

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
export GAS=${GAS:-1}
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
    MBS=$(($GBS / $DP)) ## MBS is $GAS * $MBS? 
elif [[ $MBS ]]; then
    # MBS=$(($MBS * $MP)) ## Maintain GBS across DP and MP
    GBS=$(($MBS * $DP * $GAS)) 
else
    printf "\nERROR: you need to pass in either MBS or GBS\n"; exit 1
fi
 
if [[ $DATA == 'IMNET' ]]; then
    echo "Not Implemented Error"
    exit 1
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
    EVAL_ITERS=19
    TRAIN_SAMPLES=$(($NUM_EPOCHS * $TRAIN_SIZE))
    LR_WARMUP_SAMPLES=0
    DS_CONFIG_FNAME="CIFAR.json"
elif [[ $DATA == 'TOY' ]]; then
    echo "TRAINING ON TOY DATASET"
    ##Toy Dataset
    DATA_PATH="$DATA_DIR $DATA_DIR" ## Dummy data path
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
    MICS=""
    if [[ $MICS_SHARD_SIZE -gt 1 ]]; then
        MICS="_MICS"
    fi
    DS_CONFIG_FNAME="TOY_N${NHOSTS}${MICS}.json"
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
mics_ds_config=""
if [[ $MICS_SHARD_SIZE ]]; then
    mics_ds_config="
    \"mics_hierarchical_params_gather\": true,
    \"mics_shard_size\": $MICS_SHARD_SIZE,"
fi

## DATA TYPE
data_type_ds_config=""
if [[ $fp16 == 1 && $bf16 == 16 ]]; then
    echo "you cannot choose both fp16 and bf16"
    exit 1
elif [[ $fp16 == 1 ]]; then
    data_type_ds_config='
    "communication_data_type": "fp16",
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 500,
        "hysteresis": 2,
        "min_loss_scale": 1,
        "initial_scale_power": 11
    },'
elif [[ $bf16 == 1 ]]; then
    data_type_ds_config='
    "communication_data_type": "bf16",
        "bf16": {
        "enabled": true
    },'
else
    echo "pick either fp16 or bf16"
    exit 1
fi
if [[ $fp16 == 1 ]]; then
    data_type='fp16'
elif [[ $bf16 == 1 ]]; then
    data_type='bf16'
fi
flops_profiler=''
if [[ $PROF_FLOPS -eq 1 ]]; then
    flops_profiler='
    "flops_profiler": {
                    "enabled": true,
                    "profile_step": 3,
                    "module_depth": -1,
                    "top_modules": 1,
                    "detailed": true,
                    "output_file": null
                    },'
fi
comms_logger=''
if [[ $LOG_COMMS -eq 1 ]]; then
    comms_logger='
    "comms_logger": {
        "enabled": true,
        "verbose": false,
        "prof_all": true,
        "debug": false
    },'
fi

## DS CONFIG
cat <<EOF > "$WORKING_DIR/$DS_CONFIG_FNAME"
{
    "train_batch_size": $GBS,
    "train_micro_batch_size_per_gpu": $MBS,
    "steps_per_print": 10,

    "zero_optimization": {
        "stage": $ZERO,
        "overlap_comm": true,
        "zero_hpz_partition_size": $hpz,
        $mics_ds_config
        "contiguous_gradients": true
    },

    "gradient_clipping": 1.0,
    "prescale_gradients": false,

    $data_type_ds_config
    $comms_logger
    $flops_profiler

    "gradient_accumulation_steps": $GAS, 
    "wall_clock_breakdown" : false
}
EOF


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
elif [[ $VIT == "SWIN_LIKE" ]]; then
    ## VIT-LARGE (307M)
    NLAYERS=12
    HSIZE=1152
    FFN_HSIZE=4096
    NUM_HEADS=8
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
    NUM_HEADS=24
elif [[ $VIT == "LARGE++" ]]; then
    ## VIT-LARGE (320M)
    NLAYERS=24
    HSIZE=1056
    FFN_HSIZE=4096
    NUM_HEADS=12
elif [[ $VIT == "HUGE" ]]; then
    ## VIT-HUGE (632M)
    NLAYERS=32
    HSIZE=1280
    FFN_HSIZE=5120
    NUM_HEADS=16
elif [[ $VIT == "GIANT" ]]; then
    ## 1.8B
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
elif [[ $VIT == "HUGE" ]]; then
    ## VIT-HUGE (632M)
    NLAYERS=32
    HSIZE=1280
    FFN_HSIZE=5120
    NUM_HEADS=16
elif [[ $VIT == "1B" ]]; then
    ## 1.0B
    NLAYERS=20
    HSIZE=2052
    NUM_HEADS=12
    FFN_HSIZE=$((4 * 2052))
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
    HSIZE=2568
    FFN_HSIZE=$((4 * HSIZE))
    NUM_HEADS=24
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
elif [[ $VIT == "43B" ]]; then
    ## 42.6B
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
elif [[ $VIT == "61B" ]]; then
    ## 70B in GPT; 59B in VIT?
    NLAYERS=80
    HSIZE=8400
    FFN_HSIZE=28672
    NUM_HEADS=60
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

## ARGUMENTS
ds_json=${WORKING_DIR}/${DS_CONFIG_FNAME}
echo "Working Directory: ${WORKING_DIR}"
echo "PYTHONPATH: $PYTHONPATH"

# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG

CLASSIFIER_ARGS="
     $no_pipeline_parallel \
     --zero-stage ${ZERO} \
     --pipeline-model-parallel-size ${PP} \
     --ds-sequence-parallel-size ${SP} \
     --tensor-model-parallel-size ${TP} \
     --num-layers ${NLAYERS} \
     --hidden-size ${HSIZE} \
     --num-attention-heads ${NUM_HEADS} \
     --patch-dim ${PATCH_DIM} \
     --seq-length ${SEQ_LEN} \
     --max-position-embeddings ${SEQ_LEN} \
     --img-h ${IMG_H} \
     --img-w ${IMG_W} \
     --num-classes ${NUM_CLASSES} \
     --${data_type} \
     --mask-factor 1.0 \
     --lr-decay-style cosine \
     --lr ${LR} \
     --min-lr ${MIN_LR} \
     --weight-decay ${WEIGHT_DECAY:-0.05} \
     --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
     --clip-grad 1.0 \
     --no-gradient-accumulation-fusion \
     --num-workers ${NGPUS} \
     --micro-batch-size ${MBS} \
     --attention-dropout ${ATT_DROPOUT:-0} \
     --hidden-dropout ${H_DROPOUT:-0} \
     --ffn-hidden-size ${FFN_HSIZE:-HSIZE} \
     --save ${WORKING_DIR}/save \
     --train-samples ${TRAIN_SAMPLES} \
     --retro-encoder-attention-dropout 0.0 \
     --retro-encoder-hidden-dropout 0.0 \
     --no-masked-softmax-fusion \
     --no-bias-dropout-fusion \
"

# MODEL ARGUMENTS
if [[ $FA -eq 1 ]]; then
    CLASSIFIER_ARGS="$FA_VERSION $CLASSIFIER_ARGS"
fi
if [[ $NUM_CHANNELS ]]; then
    CLASSIFIER_ARGS="--num-channels $NUM_CHANNELS $CLASSIFIER_ARGS"
fi
if [[ $TPSP -eq 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    CLASSIFIER_ARGS="--sequence-parallel $CLASSIFIER_ARGS"
fi
if [[ $MICS_SHARD_SIZE ]]; then
    CLASSIFIER_ARGS="--use-MICS $CLASSIFIER_ARGS"
fi
if [[ $DATA == "TOY" ]]; then
    CLASSIFIER_ARGS="--use-toy-data $CLASSIFIER_ARGS"
fi
if [[ $SWIN_WIN2IMG_RATIO && $SWIN_WINDOW_SIZE ]]; then
    echo "Please either define window to image ratio or window size"
    exit 1
elif [[ $USE_SWIN -eq 1 ]]; then
    CLASSIFIER_ARGS="\
    --use-swin \
    --swin-window2image-ratio ${SWIN_WIN2IMG_RATIO:-16} \
    --swin-window-size ${SWIN_WINDOW_SIZE} \
    $CLASSIFIER_ARGS"
    pretrain_script="pretrain_swin"
else
    pretrain_script="pretrain_vision_classify"
fi

DATA_ARGS="
     --tokenizer-type NullTokenizer \
     --vocab-size 0 \
     --data-path ${DATA_PATH} \
     --no-data-sharding \
     --split 949,50,1 \
     --eval-iters 0  
"
OUTPUT_ARGS="
     --log-interval 5 \
     --eval-interval $EVAL_INTERVAL \
     --wandb-project $WANDB_PROJECT_NAME \
     --save-interval 2500 \
"
DS_ARGS="
     --deepspeed \
     --deepspeed_config=$ds_json
"
if [[ $ACT_CKPT -eq 1 ]]; then
     DS_ARGS="--deepspeed-activation-checkpointing $DS_ARGS" 
     MEG_ARGS="--checkpoint-activations"
fi

echo "Launching mpiexec."
nsys=""
if [[ $MACHINE == "aurora" ]]; then
    run_cmd="mpiexec --verbose --envall -n ${NGPUS} -ppn ${NGPU_PER_HOST} \
        --hostfile ${PBS_NODEFILE} --cpu-bind depth -d 16 \
        $nsys python \
        ${WORKING_DIR}/${pretrain_script}_ezpz.py \
        ${CLASSIFIER_ARGS} \
        ${DATA_ARGS} \
        ${OUTPUT_ARGS} \
        ${MEG_ARGS} \
        ${DS_ARGS} \
    "
elif [[ $MACHINE == "polaris" ]]; then
     export RDZV_HOST=$(hostname)
     export RDZV_PORT=$RANDOM
     run_cmd="mpiexec --verbose --envall -n ${NHOSTS} -ppn 1 --cpu-bind depth -d ${NGPUS} \
          python3 -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" --nnodes=${NHOSTS} --nproc_per_node=${NGPU_PER_HOST} \
          ${WORKING_DIR}/${pretrain_script}.py \
          ${CLASSIFIER_ARGS} \
          ${DATA_ARGS} \
          ${OUTPUT_ARGS} \
          ${MEG_ARGS} \
          ${DS_ARGS}"
else
     #### CUSTOMIZE HERE ####
     echo "unknown machine. Opting for mpiexec + torchrun to execute multi-gpu/node program. May need to temper run_cmd for functionality/performance";
     export RDZV_HOST=$(hostname)
     export RDZV_PORT=$RANDOM
     run_cmd="mpiexec --verbose --envall -n ${NHOSTS} -ppn 1 --cpu-bind depth -d ${NGPUS} \
          python3 -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" --nnodes=${NHOSTS} --nproc_per_node=${NGPU_PER_HOST} \
          ${WORKING_DIR}/${pretrain_script}.py \
          ${CLASSIFIER_ARGS} \
          ${DATA_ARGS} \
          ${OUTPUT_ARGS} \
          ${MEG_ARGS} \
          ${DS_ARGS}"
fi


echo "run cmd: $run_cmd"
eval $run_cmd