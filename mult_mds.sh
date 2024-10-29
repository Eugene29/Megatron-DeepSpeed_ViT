export NEWDS=1 ## TODO: parse Deepspeed version to deprecate this flag. 
WORKING_DIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
LOGDIR=$WORKING_DIR/logs
PYSCRIPT=$WORKING_DIR/mds_launch.sh
DATA_PATH_LOG_PREFIX=$WORKING_DIR/logs


################################ ARGUMENTS ################################
# USP_ulysses=1, SP=
# USP_ring=1, SP=
# USP_hybrid=(2,4)                                      ## TBD
# PACKED={1, 5D}                                        ## Fused QKV all2all Ulysses. Need to separtely set SP degree. 1 and 5D is basically the same, just different num dimensions.
# DATA_PATH_LOG=$DATA_PATH_LOG/datafiles_consumed.log   ## Log input tensors
# SIZE=1                                                ## Number of GPU (ONLY WORKS ON 1-NODE)
# drop_last_batch_with_GBS=1                            ## fixes the data order as long as GBS is matching.
# DATA=TOY                                              ## Use Toy dataset
# factor=int                                            ## Size of your Toy image: width = height = factor * 16, channel=3
# PROFILE=1                                             ## Turn on pytorch profiler (train iter is automatically set to 10)
# GBS=2048                                              ## global batch size
# MBS=2048                                              ## micro batch size, automatically multiplied by Model Parallelism degree (i.e. MBS:= MBS * MP)
# POS_ENCODING=1                                        ## Use positioanl encoding instead of positional embedding
# WANDB_MODE=disabled                                   ## Disable WANDB
# GLOBAL_MEAN_POOLING=1                                 ## Use Global mean pooling instead of clf token 
# NUM_ITERS                                             ## Num train iteration
# FA=1                                                  ## Turn on Flash Attention
# DEBUG={SP, DP}                                        ## Triggers debug mode: run for 1 iteration and record forward activations, output, and gradients. 


################################ CURRENT CONSTRAINTS ################################
# 1. GLOBAL_MEAN_POOLING is required for SP (for now)
# 2. Pass at least GBS or MBS


################################ Global ARGUMENTS ################################
export GLOBAL_MEAN_POOLING=1
# export WANDB_MODE=disabled
# export CUDA_DEVICE_MAX_CONNECTIONS=1 ## TODO: What is this??
export drop_last_batch_with_GBS=1
# export PROFILE=1


################################ EXAMPLE RUNS ################################
export DATA=CIFAR
export GBS=2048
SIZE=1 NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log
SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
SP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds3.log

export GBS=4096
SIZE=1 NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log
SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
SP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds3.log

# # export DATA=TOY; export factor=2
# # export GBS=4
# # ## TODO: Depending on GBS, one of the three will have a different output for Toy dataset even though input is the same?
# # SP=1   NUM_ITERS=30 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
# # SP=4   NUM_ITERS=30 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds3.log
# # SIZE=1 NUM_ITERS=30 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log

## ACTIVATION CKPT ##
export DATA=CIFAR
export GBS=4096

SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1            bash $PYSCRIPT |& tee $LOGDIR/mds2.log ## 30.43 GB
SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 ACT_CKPT=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log ## 12.74 GB

################################ EXAMPLE RUNS with Data Logging ################################
# SIZE=1 NUM_ITERS=2 FA=1 POS_ENCODING=1 DATA_PATH_LOG=$DATA_PATH_LOG_PREFIX/data_consumed_DP1.log bash $PYSCRIPT |& tee $LOGDIR/mds1.log
# SP=1   NUM_ITERS=2 FA=1 POS_ENCODING=1 DATA_PATH_LOG=$DATA_PATH_LOG_PREFIX/data_consumed_DP4.log bash $PYSCRIPT |& tee $LOGDIR/mds2.log

# export DATA=TOY; export factor=297 ## GPU1 max: 297?
# export GBS=4
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ulysses=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ring=1    bash $PYSCRIPT |& tee $LOGDIR/mds2.log
# FA=1 SIZE=$GBS           POS_ENCODING=1                bash $PYSCRIPT |& tee $LOGDIR/mds3.log

# export GBS=2
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ulysses=1 bash $PYSCRIPT |& tee $LOGDIR/mds4.log
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ring=1    bash $PYSCRIPT |& tee $LOGDIR/mds5.log
# FA=1 SIZE=$GBS SP=1      POS_ENCODING=1                bash $PYSCRIPT |& tee $LOGDIR/mds6.log

# export GBS=1
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ulysses=1 bash $PYSCRIPT |& tee $LOGDIR/mds7.log
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ring=1    bash $PYSCRIPT |& tee $LOGDIR/mds8.log
# FA=1 SIZE=$GBS SP=1      POS_ENCODING=1                bash $PYSCRIPT |& tee $LOGDIR/mds9.log



## qsub example:
# qsub -A datascience -q debug -l select=1 -l logltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
# qsub -A datascience -q debug-scaling -l select=1 -l walltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh