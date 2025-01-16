#! /bin/bash

## ENVIRONMENT
echo "Launching Megatron Deepspeed VIT."

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
     ## {Aurora: frameworks-2024.2.1_u1, Polaris: 2024-08-08 base}
     WANDB_PROJECT_NAME="AuroraViT"
     DATA_DIR="/lus/flare/projects/Aurora_deployment/eku/data"
     . /lus/flare/projects/Aurora_deployment/eku/venv/vit/bin/activate ## env with ezpz, etc.
     FA_VERSION="--use-flash-attn-builder"
     NGPU_PER_HOST=12
     set_ccl_vars_on_aurora() {
          export CCL_KVS_MODE=mpi
          export CCL_CONFIGURATION_PATH=""
          export CCL_CONFIGURATION=cpu_gpu_dpcpp
          export CCL_KVS_CONNECTION_TIMEOUT=3600
          export FI_CXI_RX_MATCH_MODE=hybrid
          export CCL_BCAST=double_tree

          export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
          export CCL_PROCESS_LAUNCHER=pmix # Required by Aurora mpich
          export FI_PROVIDER=cxi           # Required by Aurora mpich
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
          export CCL_BCAST=double_tree
     }
     set_ccl_vars_on_aurora ## Gordon Bell Run
     export CCL_ALLGATHERV=topo
     export CCL_ALLREDUCE=topo
     export CCL_BCAST=double_tree
     export CCL_BARRIER=ring
     export CCL_ALLREDUCE_SCALEOUT=ring
     export CCL_ALLGATHER_SCALEOUT=ring
     export CCL_ALLGATHERV_SCALEOUT=ring
elif [[ $MACHINE == "polaris" ]]; then 
     module load conda
     conda activate
     # . /lus/eagle/projects/datascience/eku/venv/vit/bin/activate # if you want sam's ezpz
     ## Huihuo's config
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
     # FA_VERSION="--use-flash-attn-builder" ## TODO: Change back to v2 - why not v3? 
     NGPU_PER_HOST=4
     ## EXPERIMENTAL (This somehow fixes the OOM issue for Ring-Att?)
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     # torch.distributed.DistBackendError: NCCL error in: /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1970, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.20.5
     # [rank0]: ncclUnhandledCudaError: Call to CUDA function failed.
     # export NCCL_DEBUG=INFO
else
     echo "Not Impelmented Error for $MACHINE Machine"; exit 1
fi

## PYTHONPATH 
WORKING_DIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
cd $WORKING_DIR
YUNCHANG="${WORKING_DIR}/long-context-attention" ## Custom yunchang (USP)
DEEPSPEED="${WORKING_DIR}/DeepSpeed" ## Custom DeepSpeed
PYTHONPATH="${DEEPSPEED}:${YUNCHANG}:${PYTHONPATH}"
export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH}" ## Add local megatron path
## HOST NODE
# export MASTER_ADDR=localhost
# export MASTER_PORT=6000

## ARGUMENTS
source "${WORKING_DIR}/mds_args.sh"
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
     --fp16 \
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
     --save /home/eku/polaris/save \
     --train-samples ${TRAIN_SAMPLES} \
     --retro-encoder-attention-dropout 0.0 \
     --retro-encoder-hidden-dropout 0.0 \
     --no-masked-softmax-fusion \
     --no-bias-dropout-fusion \
     --accumulate-allreduce-grads-in-fp32 \
"
## TODO: does --no-async-tensor-model-parallel-allreduce \ make things faster? 

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

DATA_ARGS="
     --tokenizer-type NullTokenizer \
     --vocab-size 0 \
     --data-path ${DATA_PATH} \
     --no-data-sharding \
     --split 949,50,1 \
     --eval-iters 0  
"
## TODO: What really happens if you don't set eval-iter? How to evaluate on entire validation set?
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
     DS_ARGS="--deepspeed-activation-checkpointing $DS_ARGS" ## Useless? 
     MEG_ARGS="--checkpoint-activations"
fi

## TODO: Add prescale grad option?
# prescale_grad="true"

echo "Launching mpiexec."
## If needed to direct stdout/err
# if [[ $SAVE_LOG_TO ]]; then
#      SAVE_LOG_TO="|& tee $SAVE_LOG_TO"
# fi
# nsys="nsys profile -o $log_dir/$time --stats=true --show-output=true"
nsys=""
if [[ $MACHINE == "aurora" ]]; then
          # --cpu-bind depth -d ${NGPUS} \
     ## TODO: Why does cpu bind depth 16 works but not 24 for 2 nodes? 
     ## TODO: torchrun with mpiexec breaks but works great on polaris, why? 
     run_cmd="mpiexec --verbose --envall -n ${NGPUS} -ppn ${NGPU_PER_HOST} --hostfile ${PBS_NODEFILE} \
          --cpu-bind depth -d 16 \
          $nsys python \
          ${WORKING_DIR}/pretrain_vision_classify_ezpz.py \
          ${CLASSIFIER_ARGS} \
          ${DATA_ARGS} \
          ${OUTPUT_ARGS} \
          ${MEG_ARGS} \
          ${DS_ARGS}"
elif [[ $MACHINE == "polaris" ]]; then
     export RDZV_HOST=$(hostname)
     export RDZV_PORT=$RANDOM
          # --hostfile ${PBS_NODEFILE} \
     run_cmd="mpiexec --verbose --envall -n ${NHOSTS} -ppn 1 --cpu-bind depth -d ${NGPUS} \
          python3 -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" --nnodes=${NHOSTS} --nproc_per_node=${NGPU_PER_HOST} \
          ${WORKING_DIR}/pretrain_vision_classify.py \
          ${CLASSIFIER_ARGS} \
          ${DATA_ARGS} \
          ${OUTPUT_ARGS} \
          ${MEG_ARGS} \
          ${DS_ARGS}"

     # run_cmd="mpiexec --verbose --envall -n ${NGPUS} -ppn ${NGPU_PER_HOST} --hostfile ${PBS_NODEFILE} \
     #      --cpu-bind depth -d ${NGPUS} \
     #      $nsys python \
     #      ${WORKING_DIR}/pretrain_vision_classify_ezpz.py \
     #      ${CLASSIFIER_ARGS} \
     #      ${DATA_ARGS} \
     #      ${OUTPUT_ARGS} \
     #      ${MEG_ARGS} \
     #      ${DS_ARGS}"
else
     echo "unknown machine keyerror"; exit 1
fi

## Vanilla torchrun. Doesn't work atm at least on polaris.
# run_cmd="torchrun --nproc-per-node 4 --rdzv_backend c10d --rdzv_endpoint "$RDZV_HOST:$RDZV_PORT" \
#      ${WORKING_DIR}/pretrain_vision_classify.py \
#      ${CLASSIFIER_ARGS} \
#      ${DATA_ARGS} \
#      ${OUTPUT_ARGS} \
#      ${MEG_ARGS} \
#      ${DS_ARGS}"

echo "run cmd: $run_cmd"
eval $run_cmd