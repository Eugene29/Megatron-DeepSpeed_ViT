#! /bin/bash

## ENVIRONMENT
echo "Launching Environment."
# module load conda && conda activate base
. ~/venv/stable/bin/activate ##USER: change env accordingly
## deepspeed version missing error workaround
# cd $HOME/DeepSpeed;
# python -c "import deepspeed; deepspeed.__version__"
# cd ..

## DATA_FILEPATHS CONSUMED
export DATA_PATH_LOG="/home/eku/polaris/logs/data_paths3.log"
> $DATA_PATH_LOG ## clear file

## PYTHONPATH
SCRIPT_DIR=$(dirname $0)
PYTHONPATH=$og_PYTHONPATH

# TEMP_DS=$HOME/DeepSpeed ##TODO: Remove later.
# TEMP_DS=/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/
# PYTHONPATH="$TEMP_DS:${PYTHONPATH}" ##TODO: Remove later.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" ## Adding MEGATRON to pypath ## This should be done automatically? 

## HOST NODE
export MASTER_ADDR=localhost
export MASTER_PORT=6000

## ARGUMENTS
source "${SCRIPT_DIR}/mds_args.sh"
ds_json=${SCRIPT_DIR}/${DS_CONFIG_FNAME}
export MICRO_BATCH=$(jq -r '.train_micro_batch_size_per_gpu' $ds_json) ## I think DS config overwrites anyway.
# MICRO_BATCH=$(($MICRO_BATCH * $SP * $TP)) ##NOTE: Implemented to match the global batch size with DP
## ?? but you need it
export CUDA_DEVICE_MAX_CONNECTIONS=1 

echo "Script Directory: ${SCRIPT_DIR}"
echo "PYTHON PATH: $PYTHONPATH"

# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG
     # --train-iters ${TRAIN_ITERS} \
     # --lr-warmup-iters ${LR_WARMUP_ITERS} \
     # --use-flash-attn-v1 \
CLASSIFIER_ARGS="
     --use-flash-attn-v2 \
     --use_unifiedSP \
     $no_pipeline_parallel \
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
     --no-masked-softmax-fusion \
     --no-bias-dropout-fusion \
     --micro-batch-size ${MICRO_BATCH} \
     --attention-dropout ${ATT_DROPOUT:-0} \
     --hidden-dropout ${H_DROPOUT:-0} \
     --ffn-hidden-size ${FFN_HSIZE:-HSIZE} \
     --save /home/eku/polaris/save \
     --train-samples ${TRAIN_SAMPLES} \
     --retro-encoder-attention-dropout 0.0 \
     --retro-encoder-hidden-dropout 0.0 \
"

DATA_ARGS="
     --tokenizer-type NullTokenizer \
     --vocab-size 0 \
     --data-path ${DATA_PATH} \
     --no-data-sharding \
     --split 949,50,1 \
"

     # --eval-iters 19 \  ##TODO: What really happens if you don't set eval-iter? How to evaluate on entire validation set?
OUTPUT_ARGS="
     --log-interval 25 \
     --eval-interval $EVAL_INTERVAL \
     --wandb-project PolarisViT \
     --save-interval 2500 \
"

DS_ARGS="
     --deepspeed \
     --deepspeed_config=$ds_json
"

# echo "Arguments:"
# echo "${CLASSIFIER_ARGS}"
# echo "${DATA_ARGS}"
# echo "${OUTPUT_ARGS}"
# echo "${DS_ARGS}"


echo "Launching mpiexec."
# run_cmd="python \

run_cmd="mpiexec --verbose --envall -n ${NGPUS} -ppn ${NGPU_PER_HOST} --hostfile ${PBS_NODEFILE} \
     --cpu-bind depth -d 16 python \
     ${SCRIPT_DIR}/pretrain_vision_classify.py \
     ${CLASSIFIER_ARGS} \
     ${DATA_ARGS} \
     ${OUTPUT_ARGS} \
     ${DS_ARGS}"

# run_cmd="deepspeed --hostfile ${PBS_NODEFILE} \
#       --num_gpus ${NGPUS} \
#       ${SCRIPT_DIR}/pretrain_vision_classify.py \
#       ${CLASSIFIER_ARGS} \
#       ${DATA_ARGS} \
#       ${OUTPUT_ARGS} \
#       ${DS_ARGS}"

# printf "run_cmd: \n\n $run_cmd"
# eval $run_cmd
eval $run_cmd