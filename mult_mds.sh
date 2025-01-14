# SCRIPT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh ## Use this if submitting this script as qsub (also needed if using batch scripts?)
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
# DATA_PATH_LOG=$filepath                               ## Log input tensors. Default=$DATA_PATH_LOG/datafiles_consumed.log
# SIZE=int                                              ## Number of GPU (ONLY WORKS ON 1-NODE)
# drop_last_batch_with_GBS=1                            ## fixes the data order as long as GBS is matching.
# DATA={TOY, CIFAR, IMNET(testing..)}                   ## Use Toy dataset
# factor=int                                            ## Image_dim / patch_dim. Controls the size of your Toy image
# PROFILE={0,1}                                         ## Turn on pytorch profiler (train iter is automatically set to 10)
# GBS=int                                               ## global batch size
# MBS=int                                               ## micro batch size, automatically multiplied by Model Parallelism degree (i.e. MBS:= MBS * MP)
# POS_ENCODING={0,1}                                    ## Use positioanl encoding instead of positional embedding
# WANDB_MODE=disabled                                   ## Disable WANDB
# GLOBAL_MEAN_POOLING=1                                 ## Use Global mean pooling instead of clf token 
# NUM_ITERS=int                                         ## Num train iteration
# FA={0,1}                                              ## Turn on Flash Attention
# DEBUG={SP, DP}                                        ## Triggers debug mode: run for 1 iteration and record forward activations, output, and gradients. 
# ZERO={0,1,2,3}                                        ## Stages of DeepSpeed Zero. 0 by default
# ACT_CKPT={0,1}                                        ## set ACT_CKPT to anything to turn on activation checkpointing
# VIT3D={0,1}                                           ## Switch to 3DVIT. Must use Toy dataset for now. By default, dataset size is [GBS, p*factor, p*factor, p*factor, 1] 
#                                                          where p=16 (patch size) by default. One can change in mds_args.sh
# VIT=string                                            ## Size of VIT. Refer to mds_args.sh for possible models
# TPSP={0,1}                                            ## Upgrade from TP to TP-SP
# LOG_RESULTS={0,1}                                     ## log results (tflops, mem fpt, samples/sec) in a json file

################################ Notes ################################
# 1. Pass either GBS or MBS
# 2. ZERO123 has different loss than ZERO=0, also observed in LLM. Whether convergence is impacted needs to be tested.
# 4. 

################################ Global ARGUMENTS ################################
num_node=$(wc -l < $PBS_NODEFILE)
export drop_last_batch_with_GBS=1
# export GLOBAL_MEAN_POOLING=1
export WANDB_MODE=disabled
export POS_ENCODING=1
export PROFILE=1
export FA=1

################################# EXAMPLE RUNS #################################
# export MBS=2; export DATA=TOY; export factor=64; export VIT="13B"; #export hpz=4
export MBS=1; export DATA=TOY; export factor=64; export VIT="28B"; #export hpz=4

SP=1 ZERO=3 NUM_ITERS=10 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_MBS${MBS}.log
# SP=8 ZERO=3 NUM_ITERS=10 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_MBS${MBS}.log
# SP=8 ZERO=3 NUM_ITERS=10 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_MBS${MBS}.log