# SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
SCRIPT_DIR=$(dirname "/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh" | xargs realpath) ## Better for qsubing
LOGDIR=$SCRIPT_DIR/logs
MAIN_SCRIPT=$SCRIPT_DIR/mds_launch.sh
mkdir -p $LOGDIR

################################ Global ARGUMENTS ################################
num_node=$(wc -l < $PBS_NODEFILE)
export drop_last_batch_with_GBS=1
export GLOBAL_MEAN_POOLING=1
export WANDB_MODE=disabled
export POS_ENCODING=1
# export PROFILE=1
# export MICS_SHARD_SIZE=12
# export PROF_FLOPS=1
# export LOG_COMMS=1
# export bf16=1
export fp16=1
export FA=1
export MBS=1

################################# FOM RUNS #################################
export DATA=TOY; export VIT="LARGE"; 

export factor=64; # 4K sequence
SP=1 NUM_ITERS=5 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log

export factor=128; # 16K sequence
SP=1 NUM_ITERS=5 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log