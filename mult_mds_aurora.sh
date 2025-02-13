# SCRIPT_PTH=${BASH_SOURCE[0]} ## 
SCRIPT_PTH="/lus/flare/projects/Aurora_deployment/eku/Megatron-DeepSpeed_ViT/mult_mds_aurora.sh" ## 
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
# export PROFILE=1 ## Profiling hangs on Aurora
# export PROF_FLOPS=1
# export LOG_COMMS=1
export bf16=1
# export fp16=1
export FA=1

################################# EXAMPLE RUNS #################################
# export GBS=12; export DATA=TOY; export factor=66; export VIT="22B"; export ACT_CKPT=1 #export hpz=4
# export GBS=48; export DATA=TOY; export factor=66; export VIT="6B"; #export ACT_CKPT=1 #export hpz=4
# export GBS=1200; export DATA=CIFAR; export factor=66; export VIT="LARGE+"; #export hpz=4
export MBS=512; export DATA=CIFAR; export factor=64; export VIT="LARGE+"; #export hpz=4
# export MBS=1; export DATA=TOY; export factor=2; export VIT="LARGE+"; #export hpz=4

export USE_SWIN=1;
SP=1 ZERO=3 NUM_ITERS=500 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log
#  export 
# SP=1 ZERO=3 NUM_ITERS=15 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log