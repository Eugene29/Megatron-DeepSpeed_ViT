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
# export MICS_SHARD_SIZE=24
# export PROFILE=1
export PROF_FLOPS=1
export LOG_COMMS=1
export bf16=1
# export fp16=1
export FA=1

################################# EXAMPLE RUNS #################################
# export MBS=1; export DATA=TOY; export factor=256; export VIT="LARGE++"; #export ACT_CKPT=1; #export hpz=4
# export GBS=2048; export DATA=CIFAR; export VIT="LARGE+";
# export GBS=4; export DATA=TOY; export VIT="3B"; export ZERO=3; #export ACT_CKPT=1;
export GBS=4; export DATA=TOY; export VIT="LARGE+"; #; export ACT_CKPT=1;
export ZERO=3

SP=1 NUM_ITERS=40 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_nodes}_factor${factor}.log