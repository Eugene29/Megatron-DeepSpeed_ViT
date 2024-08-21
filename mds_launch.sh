#! /bin/bash

echo "Launching Environment."
# module load conda && conda activate base && . ~/venv/eugene/bin/activate
module load conda && conda activate base && . ~/venv/aevard/bin/activate

PYTHONPATH=$og_PYTHONPATH

## Recent Megatron
export PYTHONPATH="${HOME}/polaris/Megatron-DeepSpeed_DP:${PYTHONPATH}"

echo "Exporting and sourcing other scripts."
export MASTER_ADDR=localhost
export MASTER_PORT=6000

SCRIPT_DIR="${HOME}/polaris/Megatron-DeepSpeed_DP"
echo "Script Directory: ${SCRIPT_DIR}"
source "${SCRIPT_DIR}/mds_args.sh"

# Pre-trains ViT based image classificaation model
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG

echo "Running argument setup."


     # --pipeline-model-parallel-size ${PP} \
CLASSIFIER_ARGS="
     ${no_pipeline_parallel} \
     --ds-sequence-parallel-size ${SP} \
     --tensor-model-parallel-size ${TENSOR_MPS} \
     --num-layers ${NLAYERS} \
     --hidden-size ${HSIZE} \
     --num-attention-heads ${NUM_HEADS} \
     --patch-dim ${PATCH_DIM} \
     --seq-length ${SEQ_LEN} \
     --max-position-embeddings ${SEQ_LEN} \
     --img-h ${IMG_H} \
     --img-w ${IMG_W} \
     --num-classes ${NUM_CLASSES} \
     --mask-factor 1.0 \
     --fp16 \
     --train-iters ${TRAIN_ITERS} \
     --lr-decay-style cosine \
     --lr ${LR} \
     --min-lr ${MIN_LR} \
     --attention-dropout 0.0 \
     --weight-decay 0.05 \
     --lr-warmup-iters 2500 \
     --clip-grad 1.0 \
     --no-gradient-accumulation-fusion \
     --num-workers ${NGPUS} \
     --no-masked-softmax-fusion \
     --no-bias-dropout-fusion \
     --micro-batch-size ${MICRO_BATCH} \
"
     # --global-batch-size ${GLOBAL_BATCH}
     # --save ./ \

DATA_ARGS="
     --tokenizer-type NullTokenizer \
     --vocab-size 0 \
     --data-path ${DATA_PATH} \
     --no-data-sharding \
     --split 949,50,1 \
"

OUTPUT_ARGS="
     --log-interval 250 \
     --eval-interval 2500 \
     --eval-iters ${EVAL_ITERS}
"
     # --save-interval 2500 \

deepspeed_json=${SCRIPT_DIR}/ds_stage1_mb2_gb32_pp1_fp16.json
DS_ARGS="
     --deepspeed \
     --deepspeed_config=$deepspeed_json
"

echo "Arguments:"
echo "${CLASSIFIER_ARGS}"
echo "${DATA_ARGS}"
echo "${OUTPUT_ARGS}"
echo "${DS_ARGS}"

# /home/eku/polaris/mds/Megatron-DeepSpeed_parallel/pretrain_vision_classify.py
     # ${SCRIPT_DIR}/Megatron-DeepSpeed/pretrain_vision_classify.py \

echo "Launching mpiexec."
mpiexec --verbose --envall -n ${NGPUS} -ppn ${NGPU_PER_HOST} --hostfile ${PBS_NODEFILE} \
     --cpu-bind depth -d 16 python \
     ${SCRIPT_DIR}/pretrain_vision_classify.py \
     ${CLASSIFIER_ARGS} \
     ${DATA_ARGS} \
     ${OUTPUT_ARGS} \
     ${DS_ARGS}
     # --use-flash-attn-triton \
     # --use-flash-attn \