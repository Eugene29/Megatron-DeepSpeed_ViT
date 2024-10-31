# export NEWDS=1 ## TODO: parse Deepspeed version to deprecate this flag. 
# SCRIPT_PTH=${BASH_SOURCE[0]}
SCRIPT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
WORKING_DIR=$(dirname $SCRIPT_PTH | xargs realpath)
LOGDIR=$WORKING_DIR/logs
MAIN_SCRIPT=$WORKING_DIR/mds_launch.sh
mkdir -p $LOGDIR

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
# ZERO
# ACT_CKPT


################################ CURRENT CONSTRAINTS ################################
# 1. GLOBAL_MEAN_POOLING is required for SP (for now)
# 2. Pass at least GBS or MBS
# 3. ZERO123 has different loss than ZERO=0. But it is also observed in the example script and needs investigation.


################################ Global ARGUMENTS ################################
export GLOBAL_MEAN_POOLING=1
# export WANDB_MODE=disabled
# export CUDA_DEVICE_MAX_CONNECTIONS=1 ## TODO: What is this??
export drop_last_batch_with_GBS=1
export PROFILE=1

# unset NCCL_COLLNET_ENABLE
# unset NCCL_CROSS_NIC
# unset NCCL_NET
# unset NCCL_NET_GDR_LEVEL

################################ EXAMPLE RUNS ################################
# export DATA=CIFAR
# export GBS=1
# SIZE=1 NUM_ITERS=20 FA=1 POS_ENCODING=1               bash $MAIN_SCRIPT |& tee $LOGDIR/mds1.log
# SP=1   NUM_ITERS=10 FA=1 POS_ENCODING=1               bash $MAIN_SCRIPT |& tee $LOGDIR/mds2.log
# SP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1               bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log
# SP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1 USP_ulysses=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds4.log

# SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1               bash $MAIN_SCRIPT |& tee $LOGDIR/mds1.log
# TP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1               bash $MAIN_SCRIPT |& tee $LOGDIR/mds4.log
# SP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1               bash $MAIN_SCRIPT |& tee $LOGDIR/mds2.log
# SP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1 USP_ulysses=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log

export GBS=2048
# SP=1   FA=1 POS_ENCODING=1            bash $MAIN_SCRIPT |& tee $LOGDIR/mds5.log
# TP=8   FA=1 POS_ENCODING=1            bash $MAIN_SCRIPT |& tee $LOGDIR/mds_TP.log ## Almost matching. Is it expected and good enough?
# SP=8   FA=1 POS_ENCODING=1            bash $MAIN_SCRIPT |& tee $LOGDIR/mds6.log
# SIZE=1 FA=1 POS_ENCODING=1 ACT_CKPT=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds4.log


# export GBS=4096
# # SIZE=1 NUM_ITERS=20 FA=1 POS_ENCODING=1 ACT_CKPT=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds7.log ## OOM
# SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds8.log
# SP=4   NUM_ITERS=20 FA=1 POS_ENCODING=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds9.log

# ## ACTIVATION CKPT & ZERO ##
# export DATA=CIFAR
# export GBS=4096
# SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1                   bash $MAIN_SCRIPT |& tee $LOGDIR/mds10.log
# SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 ACT_CKPT=1        bash $MAIN_SCRIPT |& tee $LOGDIR/mds11.log
# SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 ACT_CKPT=1 ZERO=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds12.log ## Loss differs than Zero=0 and Memory increased from iter 15
# SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 ACT_CKPT=1 ZERO=2 bash $MAIN_SCRIPT |& tee $LOGDIR/mds13.log ## Loss differs than Zero=0 and Memory increased from iter 15
# SP=1   NUM_ITERS=20 FA=1 POS_ENCODING=1 ACT_CKPT=1 ZERO=3 bash $MAIN_SCRIPT |& tee $LOGDIR/mds14.log ## Loss differs than Zero=0 and Memory increased from iter 15

################################################################ Benchmark ################################################################
export DATA=TOY; export factor=100 ## Seq_len=128^2, Img_dim=(2048^2, 3), Patch_dim=16
export PROFILE=1
export GBS=2; export SIZE=1
# SP=1      FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds1.log
# SP=1      FA=1  POS_ENCODING=1  ACT_CKPT=1             bash $MAIN_SCRIPT |& tee $LOGDIR/mds2.log

export GBS=4; export SIZE=2
# SP=1      FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log
# SP=1      FA=1  POS_ENCODING=1                 ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds4.log
# SP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds5.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1          bash $MAIN_SCRIPT |& tee $LOGDIR/mds6.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ring=1             bash $MAIN_SCRIPT |& tee $LOGDIR/mds7.log 
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1  ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds8.log
# TP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds9.log

export GBS=8; export SIZE=4
# SP=1      FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds10.log
# SP=1      FA=1  POS_ENCODING=1                 ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds11.log
SP=$SIZE  FA=1  POS_ENCODING=1  ACT_CKPT=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds12.log
SP=$SIZE  FA=1  POS_ENCODING=1  ACT_CKPT=1  USP_ulysses=1          bash $MAIN_SCRIPT |& tee $LOGDIR/mds13.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ring=1             bash $MAIN_SCRIPT |& tee $LOGDIR/mds14.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1  ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds15.log
# TP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds16.log

export GBS=16; export SIZE=8
# SP=1      FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds17.log
# SP=1      FA=1  POS_ENCODING=1                 ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds18.log
# SP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds19.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1          bash $MAIN_SCRIPT |& tee $LOGDIR/mds20.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ring=1             bash $MAIN_SCRIPT |& tee $LOGDIR/mds21.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1  ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds22.log
# TP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds23.log

################################################################ EXAMPLE RUNS with Data Logging ################################################################
# export DATA=TOY
# export GBS=1; export factor=100
# SIZE=1 NUM_ITERS=2 FA=1 POS_ENCODING=1 DATA_PATH_LOG=$LOGDIR/data_consumed_DP1.log bash $MAIN_SCRIPT |& tee $LOGDIR/mds1.log
# SP=1   NUM_ITERS=2 FA=1 POS_ENCODING=1 DATA_PATH_LOG=$LOGDIR/data_consumed_DP4.log bash $MAIN_SCRIPT |& tee $LOGDIR/mds2.log
# SP=4   NUM_ITERS=4 FA=1 POS_ENCODING=1               bash $MAIN_SCRIPT |& tee $LOGDIR/mds4.log
# SP=4   NUM_ITERS=4 FA=1 POS_ENCODING=1 USP_ulysses=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds5.log

# SP=1   NUM_ITERS=2 FA=1 POS_ENCODING=1 DATA_PATH_LOG=$LOGDIR/data_consumed_DP2SP16.log bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log
# SP=4   NUM_ITERS=2 FA=1 POS_ENCODING=1 DATA_PATH_LOG=$LOGDIR/data_consumed_DP2SP16.log bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log
# SP=16   NUM_ITERS=2 FA=1 POS_ENCODING=1 DATA_PATH_LOG=$LOGDIR/data_consumed_DP2SP16.log bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log

# export DATA=TOY; export factor=297 ## GPU1 max: 297?
# export GBS=4
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ulysses=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds1.log
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ring=1    bash $MAIN_SCRIPT |& tee $LOGDIR/mds2.log
# FA=1 SIZE=$GBS           POS_ENCODING=1                bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log

# export GBS=2
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ulysses=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds4.log
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ring=1    bash $MAIN_SCRIPT |& tee $LOGDIR/mds5.log
# FA=1 SIZE=$GBS SP=1      POS_ENCODING=1                bash $MAIN_SCRIPT |& tee $LOGDIR/mds6.log

# export GBS=1
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ulysses=1 bash $MAIN_SCRIPT |& tee $LOGDIR/mds7.log
# FA=1 SIZE=$GBS SP=$GBS   POS_ENCODING=1  USP_ring=1    bash $MAIN_SCRIPT |& tee $LOGDIR/mds8.log
# FA=1 SIZE=$GBS SP=1      POS_ENCODING=1                bash $MAIN_SCRIPT |& tee $LOGDIR/mds9.log



## qsub example:
# qsub -A datascience -q debug -l select=1 -l logltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
# qsub -A datascience -q debug-scaling -l select=1 -l walltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh