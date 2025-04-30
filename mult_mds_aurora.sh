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
export LOG_COMMS=1
# export bf16=1
export fp16=1
export FA=1  # 1

################################# EXAMPLE RUNS #################################
# export GBS=12; export DATA=TOY; export factor=66; export VIT="22B"; export ACT_CKPT=1 
export MBS=1; export DATA=TOY; export factor=66; export VIT="4B" 
# export MBS=1; export DATA=CIFAR; export factor=2; export VIT="LARGE+"; 
# export MBS=1; export DATA=TOY; export factor=64; export VIT="TINY"; 
# export MBS=2048; export DATA=TOY; export factor=4; export VIT="BASE"; 

# export USE_SWIN=1; 
# export SWIN_WINDOW_SIZE=4; export PATCH_DIM=2
# echo Window Sequence Length: $((SWIN_WINDOW_SIZE**2))
# echo Total sequence length: $(((factor/PATCH_DIM)**2))
# export ACT_CKPT=1; ## Does not work for SWIN atm

# export ZERO=2
# export ACT_CKPT=1

SP=12 ZERO=3 NUM_ITERS=500 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log
# SP=1 NUM_ITERS=500 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log