DIR=$(dirname "/lus/flare/projects/Aurora_deployment/eku/Megatron-DeepSpeed_ViT/zero3_bench.sh" | xargs realpath)
LAUNCH_SCRIPT=$DIR/mds_launch.sh
LOG_DIR="$DIR/logs"
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

## SAFER RUN
VIT=$safer_model_size bash $LAUNCH_SCRIPT |& tee "$LOG_DIR/n${num_nodes}_VIT${safer_model_size}_GBS${GBS}.log"

## SAFE RUN
VIT=$safe_model_size bash $LAUNCH_SCRIPT |& tee "$LOG_DIR/n${num_nodes}_VIT${safe_model_size}_GBS${GBS}.log"

## RISKY RUN
VIT=$risky_model_size bash $LAUNCH_SCRIPT |& tee "$LOG_DIR/n${num_nodes}_VIT${risky_model_size}_GBS${GBS}.log"

# ## hpz RUN (SCALE OUT)
