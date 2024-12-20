DIR=$(dirname "/eagle/datascience/eku/Megatron-DeepSpeed_ViT/zero3_bench.sh")
LAUNCH_SCRIPT=$(realpath $DIR/mds_launch.sh)
LOG_DIR="$DIR/logs"
PROJ="datascience"
mkdir -p $LOG_DIR

## Model Variables
export drop_last_batch_with_GBS=1
export WANDB_MODE=disabled
export POS_ENCODING=1
export factor=16; ## sequence len = 16^3 = 4096
export PROFILE=1
export DATA=TOY; 
export VIT3D=1
export ZERO=3
export FA=1

## SAFE RUN
LOG_PTH="$LOG_DIR/n${num_nodes}_VIT${safe_model_size}.log"
VIT=$safe_model_size bash $LAUNCH_SCRIPT |& tee $LOG_PTH
## RISKY RUN
LOG_PTH="$LOG_DIR/n${num_nodes}_VIT${risky_model_size}.log"
VIT=$risky_model_size bash $LAUNCH_SCRIPT |& tee $LOG_PTH

## INTERACTIVE LAUNCH (TODO)
# arr_num_nodes=(2)
# arr_num_nodes=(2)