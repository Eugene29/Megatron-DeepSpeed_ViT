## Welcome to ALCF ViT repo

## Clone & Init Submodule:
```
  git clone --recursive https://github.com/Eugene29/Megatron-DeepSpeed_ViT.git ## Clone module + submodule
  cd Megatron-DeepSpeed_ViT
  git submodule update --init --recursive ## Init & Update submodule
```

## Environment
Only base environment is needed for polaris cluster while for aurora, we employ sam's ezpz library. A suitable virtual environment (in flare file-system) is activated automatically on aurora. 

## Notes:
Main script for entry is `mult_mds.sh`. In here, you'll need to modify `SCRIPT_DIR`. There is also descriptions for possible flags. Any other fixed ENV and MDS-related variables can be changed in `mult_launch.sh`. 

## POSSIBLE ENV VARIABLES
```
USP_ulysses=1, SP=                                    ## Turn on USP's Ulysses. Separately set degree by SP=_
USP_ring=1, SP=                                       ## Turn on USP's Ulysses. Separately set degree by SP=_
USP_hybrid=(2,4)                                      ## TBD
PACKED={1, 5D}                                        ## Fused QKV all2all Ulysses. Need to separtely set SP degree. 1 and 5D is basically the same, just different num dimensions.
DATA_PATH_LOG=$filepath                               ## Log input tensors. Default=$DATA_PATH_LOG/datafiles_consumed.log
SIZE=int                                              ## Number of GPU (ONLY WORKS ON 1-NODE)
drop_last_batch_with_GBS=1                            ## fixes the data order as long as GBS is matching.
DATA={TOY, CIFAR, IMNET(testing..)}                   ## Use Toy dataset
factor=int                                            ## Image_dim / patch_dim. Controls the size of your Toy image
PROFILE={0,1}                                         ## Turn on pytorch profiler (train iter is automatically set to 10)
GBS=int                                               ## global batch size
MBS=int                                               ## micro batch size, automatically multiplied by Model Parallelism degree (i.e. MBS:= MBS * MP)
POS_ENCODING={0,1}                                    ## Use positioanl encoding instead of positional embedding
WANDB_MODE=disabled                                   ## Disable WANDB
GLOBAL_MEAN_POOLING=1                                 ## Use Global mean pooling instead of clf token 
NUM_ITERS=int                                         ## Num train iteration
FA={0,1}                                              ## Turn on Flash Attention
DEBUG={SP, DP}                                        ## Triggers debug mode: run for 1 iteration and record forward activations, output, and gradients. 
ZERO={0,1,2,3}                                        ## Stages of DeepSpeed Zero. 0 by default
ACT_CKPT={0,1}                                        ## set ACT_CKPT to anything to turn on activation checkpointing
VIT3D={0,1}                                           ## Switch to 3DVIT. Must use Toy dataset for now. By default, dataset size is [GBS, p*factor, p*factor, p*factor, 1] 
                                                         where p=16 (patch size) by default. One can change in mds_args.sh
VIT=string                                            ## Size of VIT. Refer to mds_args.sh for possible models
TPSP={0,1}                                            ## Upgrade from TP to TP-SP
LOG_RESULTS={0,1}                                     ## log results (tflops, mem fpt, samples/sec) in a json file
MICS_SHARD_SIZE                                       ## Size of your MICS partition group
fp16                                                  ## enable fp16
bf16                                                  ## use datatype bf16
LOG_COMMS                                             ## log/profile communications through deepspeed
PROF_FLOPS                                            ## profile flop counts with detail through deepspeed

################################ Notes ################################
1. Pass either GBS or MBS
2. Pass either fp16 or bf16
2. ZERO123 has different loss than ZERO=0, also observed in LLM. Whether convergence is impacted needs to be tested.
```