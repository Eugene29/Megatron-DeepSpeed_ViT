# export NEWDS=1 ## TODO: parse Deepspeed version to deprecate this flag. 
# SCRIPT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
SCRIPT_PTH=${BASH_SOURCE[0]}
WORKING_DIR=$(dirname $SCRIPT_PTH | xargs realpath)
LOGDIR=$WORKING_DIR/logs
MAIN_SCRIPT=$WORKING_DIR/mds_launch.sh
mkdir -p $LOGDIR

################################ ARGUMENTS ################################
# USP_ulysses=1, SP=                                    ## Turn on USP's Ulysses. Separately set degree by SP=_
# USP_ring=1, SP=                                       ## Turn on USP's Ulysses. Separately set degree by SP=_
# USP_hybrid=(2,4)                                      ## TBD
# PACKED={1, 5D}                                        ## Fused QKV all2all Ulysses. Need to separtely set SP degree. 1 and 5D is basically the same, just different num dimensions.
# DATA_PATH_LOG=$DATA_PATH_LOG/datafiles_consumed.log   ## Log input tensors
# SIZE=1                                                ## Number of GPU (ONLY WORKS ON 1-NODE)
# drop_last_batch_with_GBS=1                            ## fixes the data order as long as GBS is matching.
# DATA=TOY                                              ## Use Toy dataset
# factor=int                                            ## Size of your Toy image: width = height = factor * patch_dim (default=16), channel=3
# PROFILE=1                                             ## Turn on pytorch profiler (train iter is automatically set to 10)
# GBS=2048                                              ## global batch size
# MBS=2048                                              ## micro batch size, automatically multiplied by Model Parallelism degree (i.e. MBS:= MBS * MP)
# POS_ENCODING=1                                        ## Use positioanl encoding instead of positional embedding
# WANDB_MODE=disabled                                   ## Disable WANDB
# GLOBAL_MEAN_POOLING=1                                 ## Use Global mean pooling instead of clf token 
# NUM_ITERS                                             ## Num train iteration
# FA=1                                                  ## Turn on Flash Attention
# DEBUG={SP, DP}                                        ## Triggers debug mode: run for 1 iteration and record forward activations, output, and gradients. 
# ZERO                                                  ## {0, 1, 2, 3} - 0 by default
# ACT_CKPT                                              ## set ACT_CKPT to anything to turn on activation checkpointing
# VIT3D                                                 ## Switch to 3DVIT. Must use Toy dataset for now. By default, dataset size is [GBS, p*factor, p*factor, p*factor, 1] 
#                                                          where p=16 (patch size) by default. One can change in mds_args.sh
# VIT                                                   ## Size of VIT {TINY, BASE, LARGE, HUGE}. Refer to mds_args.sh for their sizes
# TPSP                                                  ## Upgrade from TP to TP-SP
# LOG_RESULTS                                           ## log results (tflops, mem fpt, samples/sec) in a json file

################################ CURRENT LIMITATION ################################
# 1. Pass at least GBS or MBS
# 2. ZERO123 has different loss than ZERO=0. Whether convergence is impacted needs to be tested.
# 3. Setting Environment Variables to 1 turns on the variables. (e.g. FA=0 to turn off or 1 to turn on Flash Attention)

################################ Global ARGUMENTS ################################
# export GLOBAL_MEAN_POOLING=1
export WANDB_MODE=disabled
# export CUDA_DEVICE_MAX_CONNECTIONS=1 ## TODO: What is this??
export drop_last_batch_with_GBS=1
export PROFILE=1

# unset NCCL_COLLNET_ENABLE
# unset NCCL_CROSS_NIC
# unset NCCL_NET
# unset NCCL_NET_GDR_LEVELs
# libfabric

################################ EXAMPLE RUNS ################################
export GBS=4; export DATA=TOY; export factor=64; export VIT=HUGE
SP=1 ZERO=3 NUM_ITERS=10 FA=1 POS_ENCODING=1 bash $MAIN_SCRIPT |& tee $LOGDIR/benchmark.log ## check that this runs as expectedly. 