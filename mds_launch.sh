#! /bin/bash

## ENVIRONMENT
echo "Launching Megatron Deepspeed VIT."
. /home/eku/venv/stable_ds15.1/bin/activate ## USER: change env accordingly (env has sam's ezpz repo + deepspeed tag: v0.15.1)

## If DATA_PATH_LOG is passed, will record input tensors consumed
if [[ $DATA_PATH_LOG ]]; then
     > $DATA_PATH_LOG 
fi

## PYTHONPATH 
SCRIPT_DIR=$(dirname $0 | xargs realpath)
cd $SCRIPT_DIR
PYTHONPATH=$og_PYTHONPATH
YUNCHANG=/home/eku/long-context-attention ## Custom yunchang (USP)
PYTHONPATH="$YUNCHANG:$PYTHONPATH"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" ## Add local megatron path

## HOST NODE
export MASTER_ADDR=localhost
export MASTER_PORT=6000

## ARGUMENTS
source "${SCRIPT_DIR}/mds_args.sh"
ds_json=${SCRIPT_DIR}/${DS_CONFIG_FNAME}

echo "Script Directory: ${SCRIPT_DIR}"
echo "PYTHON PATH: $PYTHONPATH"

# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG

CLASSIFIER_ARGS="
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
     --micro-batch-size ${MBS} \
     --attention-dropout ${ATT_DROPOUT:-0} \
     --hidden-dropout ${H_DROPOUT:-0} \
     --ffn-hidden-size ${FFN_HSIZE:-HSIZE} \
     --save /home/eku/polaris/save \
     --train-samples ${TRAIN_SAMPLES} \
     --retro-encoder-attention-dropout 0.0 \
     --retro-encoder-hidden-dropout 0.0 \
"

if [ -n "$unifiedSP" ]; then
     CLASSIFIER_ARGS="--use_unifiedSP $CLASSIFIER_ARGS"
fi
if [ -n "$FA" ]; then
     CLASSIFIER_ARGS="--use-flash-attn-v2 $CLASSIFIER_ARGS"
fi
if [ -n "$NUM_CHANNELS" ]; then
     CLASSIFIER_ARGS="--num-channels $NUM_CHANNELS"
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
     --wandb-project PolarisViT \
     --save-interval 2500 \
"
DS_ARGS="
     --deepspeed \
     --deepspeed_config=$ds_json
"
if [[ $ACT_CKPT ]]; then
     DS_ARGS="--deepspeed-activation-checkpointing $DS_ARGS" ## Useless? 
     MEG_ARGS="--checkpoint-activations"
fi

echo "Launching mpiexec."
# nsys="nsys profile -o $log_dir/$time --stats=true --show-output=true"
nsys=""

run_cmd="mpiexec --verbose --envall -n ${NGPUS} -ppn ${NGPU_PER_HOST} --hostfile ${PBS_NODEFILE} \
     --cpu-bind depth -d 16 \
     $nsys python \
     ${SCRIPT_DIR}/pretrain_vision_classify.py \
     ${CLASSIFIER_ARGS} \
     ${DATA_ARGS} \
     ${OUTPUT_ARGS} \
     ${MEG_ARGS} \
     ${DS_ARGS}"

echo "run cmd: $run_cmd"
eval $run_cmd