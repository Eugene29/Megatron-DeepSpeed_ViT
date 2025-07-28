#! /bin/bash

# ------------------------------------------------------------------------------------ #
# helper functions

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

setup_aurora_env_and_vars() {
    ## TODO: if hang is observed with num_node=1, conditionally remove cxi
    module restore
    module load frameworks
    NHOSTS=$(wc -l < "${PBS_NODEFILE}")
    WANDB_PROJECT_NAME="AuroraViT"
    FA_VERSION="--use-flash-attn-builder"
    NGPU_PER_HOST=12
    zero_overlap_comm=false
    . /lus/flare/projects/Aurora_deployment/eku/venv/vit/bin/activate

    ## CCL Vars
    export CCL_KVS_MODE=mpi
    export CCL_KVS_CONNECTION_TIMEOUT=600 
    export PALS_PMI=pmix
    export CCL_ATL_TRANSPORT=mpi

    export TORCH_LLM_ALLREDUCE=1
    export CCL_SYCL_ESIMD=1
    export CCL_ATL_SYNC_COLL=1
    export CCL_OP_SYNC=1
    export CCL_ENABLE_AUTO_CACHE=0
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=$((4096 * 8))

    export CCL_ALLGATHERV=topo # direct
    export CCL_ALLGATHERV_SCALEOUT=ring
    export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0
    export CCL_ALLREDUCE=topo
    export CCL_ALLREDUCE_SCALEOUT=ring
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

setup_polaris_env_and_vars() {
    module load conda
    conda activate
    NHOSTS=$(wc -l < "${PBS_NODEFILE}")
    WANDB_PROJECT_NAME="PolarisViT"
    FA_VERSION="--use-flash-attn-v2"
    NGPU_PER_HOST=4
    zero_overlap_comm=true
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CPU_BIND="depth -d 4"
    source /lus/eagle/projects/datascience/eku/venv/base/bin/activate

    # Nvidia Vars
    export AWS_DIR=/soft/libraries/aws-ofi-nccl/v1.9.1-aws
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
}

setup_model_hyperparameter() {
    export VIT=${VIT:-"LARGE"}
    echo Using VIT-$VIT
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
    elif [[ $VIT == "315M" ]]; then
        ## Aurora friendly (head count is dividsible by 12)
        NLAYERS=14
        HSIZE=1536
        FFN_HSIZE=4096
        NUM_HEADS=12
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
    else
        echo "VIT not implemented"
        exit 1
    fi
}

setup_megatron_deepspeed_args() {
    export drop_last_batch_with_GBS=1
    export GLOBAL_MEAN_POOLING=1
    export POS_ENCODING=1

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
    
    echo "TRAINING ON TOY DATASET"
    export DATA=TOY
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

    MICS=""
    if [[ $MICS_SHARD_SIZE -gt 1 ]]; then
        MICS="_MICS"
    fi
    DS_CONFIG_FNAME="TOY_N${NHOSTS}${MICS}.json"

    if [[ $NUM_ITERS ]]; then
        TRAIN_SAMPLES=$(($NUM_ITERS * $GBS))
    fi

    export ZERO=${ZERO:-1}
    export hpz=${hpz:-1}
    mics_ds_config=""
    internode_MICS=false  # 
    if [[ $MICS_SHARD_SIZE ]]; then
        mics_ds_config="
        \"mics_hierarchical_params_gather\": $internode_MICS,
        \"mics_shard_size\": $MICS_SHARD_SIZE,"
    fi

    ## DATA TYPE
    data_type_ds_config=""
    if [[ $fp16 == 1 && $bf16 == 1 ]]; then
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
    cat <<EOF > "$PBS_O_WORKDIR/$DS_CONFIG_FNAME"
{
    "train_batch_size": $GBS,
    "train_micro_batch_size_per_gpu": $MBS,
    "steps_per_print": 10,

    "zero_optimization": {
        "stage": $ZERO,
        "overlap_comm": $zero_overlap_comm,
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

    export TRAIN_SAMPLES="${TRAIN_SAMPLES:-5000}"
    export EVAL_ITERS="${EVAL_ITERS:-1000}"
    export LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-250}"
    export EVAL_INTERVAL=${EVAL_INTERVAL:-250}
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

    ds_json=${PBS_O_WORKDIR}/${DS_CONFIG_FNAME}
    echo "Working Directory: ${PBS_O_WORKDIR}"
    echo "PYTHONPATH: $PYTHONPATH"

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
        --save ${PBS_O_WORKDIR}/save \
        --train-samples ${TRAIN_SAMPLES} \
        --retro-encoder-attention-dropout 0.0 \
        --retro-encoder-hidden-dropout 0.0 \
        --no-masked-softmax-fusion \
        --no-bias-dropout-fusion \
        --no-bias-gelu-fusion \
    "

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

    DATA_ARGS="
        --tokenizer-type NullTokenizer \
        --vocab-size 0 \
        --data-path ${PBS_O_WORKDIR} ${PBS_O_WORKDIR} \
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
}

# ------------------------------------------------------------------------------------ #
# Main

echo "Launching Megatron Deepspeed VIT."
TZ="America/Chicago" date

# 1. Set MACHINE var
get_machine

# 2. Setup environment and cluster-specific variables
if [[ $MACHINE == "aurora" ]]; then
    setup_aurora_env_and_vars
elif [[ $MACHINE == "polaris" ]]; then
    setup_polaris_env_and_vars
else
    #### CUSTOMIZE HERE ####
    NGPU_PER_HOST="<number of GPUs per node>"
    NHOSTS="<number of nodes>"
    FA_VERSION="--use-flash-attn-v2"  # FA version
    zero_overlap_comm="<overlap for zero>"
    CPU_BIND="<cpu-bind for mpiexec>"
    # <activate your environment>
fi

# 3. Set-up python args
setup_model_hyperparameter
setup_megatron_deepspeed_args

# 4. Launch run_cmd
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((RANDOM + 1024))
run_cmd="mpiexec --verbose --envall -n ${NGPUS} -ppn ${NGPU_PER_HOST} \
    --cpu-bind $CPU_BIND \
    python3 ${PBS_O_WORKDIR}/pretrain_vision_classify.py \
    ${CLASSIFIER_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${MEG_ARGS} \
    ${DS_ARGS} \
"
echo "Launching mpiexec."
echo "run cmd: $run_cmd"
eval $run_cmd