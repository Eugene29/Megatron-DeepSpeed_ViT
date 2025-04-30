# SCRIPT_PTH=${BASH_SOURCE[0]} ## 
SCRIPT_PTH="/lus/flare/projects/Aurora_deployment/eku/Megatron-DeepSpeed_ViT/all2all_workload.sh" ## 
SCRIPT_DIR=$(dirname $SCRIPT_PTH | xargs realpath)
LOGDIR=$SCRIPT_DIR/logs
MAIN_SCRIPT=$SCRIPT_DIR/mds_launch.sh
mkdir -p $LOGDIR

################################ Global ARGUMENTS ################################
num_node=$(wc -l < $PBS_NODEFILE)
export drop_last_batch_with_GBS=1
export GLOBAL_MEAN_POOLING=1
# export WANDB_MODE=disabled
export POS_ENCODING=1
# export MICS_SHARD_SIZE=24
# export PROFILE=1 ## Torch Profiling hangs on Aurora
export PROF_FLOPS=1
export LOG_COMMS=1
# export bf16=1
export fp16=1
export FA=1

################################# EXAMPLE RUNS #################################
## Variables of interest. 
## you can try VIT={LARGE+, 1B, 4B, 8B, 20B, 22B}, Use ZERO to fit bigger model
## Sequence Length = factor^2

export GBS=12; export DATA=TOY; export factor=132; export VIT="8B";
# export GBS=1200; export DATA=CIFAR; export factor=132; export VIT="1B";
export ZERO=3
export ACT_CKPT=1

## SP=12 (Ulysses)
SP=12 NUM_ITERS=15 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log
# SP=4 NUM_ITERS=15 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log

## DP=12
# SP=1 NUM_ITERS=15 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log\