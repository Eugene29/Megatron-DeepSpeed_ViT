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

################################ CURRENT CONSTRAINTS ################################
# 1. GLOBAL_MEAN_POOLING is required for SP (for now)
# 2. Pass at least GBS or MBS
# 3. ZERO123 has different loss than ZERO=0. But it is also observed in the example script and needs investigation.
# 4. FA=0 or 1 both turns on FA. unset FA to disable. 


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
################################################################ Benchmark ################################################################
# export DATA=TOY; ## Seq_len=128^2, Img_dim=(2048^2, 3), Patch_dim=16
# export NUM_ITERS=5; 
# export NUM_CHANNELS=1 ## Q. how come changing number of channels can incur higher memory? 
# export FA=1
# export POS_ENCODING=1
# export PROFILE=1
# export LOG_RESULTS=1

# experiment () {
#     ## Size: Num GPU
#     ## GBS: GBS per GPU. Actual GBS := GBS * SIZE
#     ## factor: Image Size = 
#     SIZE=$1
#     GBS=$(($2 * SIZE)) ## Weak Scaling
#     factor=$3
#     export SIZE=$SIZE
#     export GBS=$GBS
#     export factor=$factor
    
#     SP=1                     method="DP"   bash $MAIN_SCRIPT  |&  tee  $LOGDIR/3dvit_DP${SIZE}_factor${factor}.log    ## SP
#     if [[ $SIZE -ne 1 ]]; then
#         TP=$SIZE                 method="TP"   bash $MAIN_SCRIPT  |&  tee  $LOGDIR/3dvit_TP${SIZE}_factor${factor}.log    ## TP
#         TP=$SIZE  TPSP=1         method="TPSP" bash $MAIN_SCRIPT  |&  tee  $LOGDIR/3dvit_TPSP${SIZE}_factor${factor}.log  ## TP-SP
#         SP=$SIZE                 method="SPU"  bash $MAIN_SCRIPT  |&  tee  $LOGDIR/3dvit_SPU${SIZE}_factor${factor}.log   ## SP
#         SP=$SIZE  USP_ulysses=1  method="USPU" bash $MAIN_SCRIPT  |&  tee  $LOGDIR/3dvit_USPU${SIZE}_factor${factor}.log  ## USP-Uly
#         SP=$SIZE  USP_ring=1     method="USPR" bash $MAIN_SCRIPT  |&  tee  $LOGDIR/3dvit_USPR${SIZE}_factor${factor}.log  ## USP-Ring
#     fi
# }

# for SIZE in 8; do
#     ##           Size  GBS   factor
#     experiment  $SIZE  1024  8
#     experiment  $SIZE  256   16
#     experiment  $SIZE  64    32
#     experiment  $SIZE  16    64
#     experiment  $SIZE  4     128
#     experiment  $SIZE  1     256
# done

# export SIZE=2
# export GBS=$((4 * SIZE)); export factor=128;

# export GBS=4; export SIZE=2
# factor=9999 SP=1      FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds3.log
# SP=1      FA=1  POS_ENCODING=1                 ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds4.log
# SP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds5.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1          bash $MAIN_SCRIPT |& tee $LOGDIR/mds6.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ring=1             bash $MAIN_SCRIPT |& tee $LOGDIR/mds7.log 
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1  ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds8.log
# TP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds9.log

# export GBS=8; export SIZE=4
# SP=1      FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds10.log
# SP=1      FA=1  POS_ENCODING=1                 ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds11.log
# SP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds12.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_hybrid=1  ZERO=2        bash $MAIN_SCRIPT |& tee $LOGDIR/mds13.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ring=1             bash $MAIN_SCRIPT |& tee $LOGDIR/mds14.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1  ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds15.log
# TP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds16.log

# export GBS=16; export SIZE=8
# SP=1      FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds17.log
# SP=1      FA=1  POS_ENCODING=1                 ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds18.log
# SP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds19.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1          bash $MAIN_SCRIPT |& tee $LOGDIR/mds20.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ring=1             bash $MAIN_SCRIPT |& tee $LOGDIR/mds21.log
# SP=$SIZE  FA=1  POS_ENCODING=1  USP_ulysses=1  ZERO=2  bash $MAIN_SCRIPT |& tee $LOGDIR/mds22.log
# TP=$SIZE  FA=1  POS_ENCODING=1                         bash $MAIN_SCRIPT |& tee $LOGDIR/mds23.log

################################################################ VIT3D Benchmark ################################################################
## BASE
# num_nodes=$(wc -l < $PBS_NODEFILE)
# num_gpu=$(( 4 * num_nodes ))
# export VIT3D=1; export VIT=BASE; export GBS=$num_gpu; export DATA=TOY; export NUM_ITERS=10; 
# export POS_ENCODING=1; export FA=1; 
# export PROFILE=1

# factor = [4, 8, 16, 32, 64, 128] 
# IV = [DP, TP, TP-SP, SP, USP-Uly, USP-Ring]
# for factor in 4 8 16 32; do ## Cannot fit 64
#     export factor=$factor
#     SP=1  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP${num_gpu}.log ## DP
#     for MP in 2 4; do
#         DP=$(( GBS / MP ))
#         TP=$MP                  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_TP${MP}_DP${DP}_factor${factor}.log    ## TP
#         TP=$MP  TP-SP=1         bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_TPSP${MP}_DP${DP}_factor${factor}.log  ## TP-SP
#         SP=$MP                  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_SPU${MP}_DP${DP}_factor${factor}.log   ## SP
#         SP=$MP  USP_ulysses=1   bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_USPU${MP}_DP${DP}_factor${factor}.log  ## USP-Uly
#         SP=$MP  USP_ring=1      bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_USPR${MP}_DP${DP}_factor${factor}.log  ## USP-Ring
#     done
# done

# export factor=4   
# GBS=8192  SP=1                bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP${num_gpu}_factor${factor}.log ## DP
# export factor=8   
# GBS=1024  SP=1                bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP${num_gpu}_factor${factor}.log ## DP
# export factor=16  
# GBS=128   SP=1                bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP${num_gpu}_factor${factor}.log ## DP
# export factor=32  
# GBS=16    SP=1                bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP${num_gpu}_factor${factor}.log ## DP
# export factor=64  
# GBS=4     SP=1  ACT_CKPT=1    bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP${num_gpu}_factor${factor}_ACT.log ## DP

# export factor=64
# for GBS in 2; do
#     for MP in 2; do
#         DP=$(( GBS / MP ))
#         TP=$MP  GBS=$GBS                  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_TP${MP}_DP${DP}_factor${factor}.log    ## TP
#         TP=$MP  GBS=$GBS  TPSP=1          bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_TPSP${MP}_DP${DP}_factor${factor}.log  ## TP-SP
#         SP=$MP  GBS=$GBS                  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_SPU${MP}_DP${DP}_factor${factor}.log   ## SP
#         SP=$MP  GBS=$GBS  USP_ulysses=1   bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_USPU${MP}_DP${DP}_factor${factor}.log  ## USP-Uly
#         SP=$MP  GBS=$GBS  USP_ring=1      bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_USPR${MP}_DP${DP}_factor${factor}.log  ## USP-Ring
#     done
# done

# factor=128  GBS=1  SP=4  FA=1  USP_ulysses=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP2SP2.log ## DP


# export factor=32
# export GBS=32
# # GBS=32  SP=1  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_DP${num_gpu}.log ## DP
# for MP in 2; do
#     DP=$(( GBS / MP ))
#     # TP=$MP                  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_TP${MP}_DP${DP}_factor${factor}.log    ## TP
#     TP=$MP  TP-SP=1         bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_TPSP${MP}_DP${DP}_factor${factor}.log  ## TP-SP
#     SP=$MP                  bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_SPU${MP}_DP${DP}_factor${factor}.log   ## SP
#     # SP=$MP  USP_ulysses=1   bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_USPU${MP}_DP${DP}_factor${factor}.log  ## USP-Uly
#     SP=$MP  USP_ring=1      bash $MAIN_SCRIPT |& tee $LOGDIR/3dvit_USPR${MP}_DP${DP}_factor${factor}.log  ## USP-Ring
# done

## LARGE
# export VIT3D=1; export VIT=LARGE; export GBS=1; 
# export SIZE=1; export factor=44;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_l1.log 
# export SIZE=2; export factor=52;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_l2.log 
# export SIZE=4; export factor=62;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_l3.log
# export SIZE=8; export factor=78;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_l4.log
# export SIZE=16; export factor=88;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_l5.log

## HUGE
# export VIT3D=1; export VIT=HUGE; export GBS=1; 
# export SIZE=1; export factor=36;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_h1.log
# export SIZE=2; export factor=42;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_h2.log
# export SIZE=4; export factor=52;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_h3.log
# export SIZE=8; export factor=64;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_h4.log
# export SIZE=16; export factor=72;
# SP=$SIZE  FA=1  POS_ENCODING=1  bash $MAIN_SCRIPT |& tee $LOGDIR/mds_h5.log

# unset VIT3D; unset VIT; unset GBS ## Prevent VIT3D turned on for other runs

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