## Clone 
If not already cloned,
```
  git clone --recursive https://github.com/Eugene29/Megatron-DeepSpeed_ViT.git  # Clone module & submodule
```
## Init Submodule
```
  cd Megatron-DeepSpeed_ViT
  git submodule update --init --recursive  # Init & Update submodule
```

## Main Script for Entry:
Main script for entry is `mult_mds_aurora.sh` or `mult_mds_polaris.sh`. You'll need to modify `SCRIPT_DIR`. Environment variables not in `mult_mds_*.sh` can be found and configured in `mds_launch.sh`. 

## For ALCF4 Benchmark:
- Please run mult_mds_polaris_alcf4.sh
- Environment variable DATA_DIR is just a placeholder and will only be used when using CIFAR dataset (i.e. DATA=CIFAR). For benchmarking only purposes, it can be set to any existing filepath.

## POSSIBLE ENV VARIABLES
```
DATA={TOY, CIFAR}           ## Use Toy dataset
factor=int                  ## Represents image_dim/patch_dim of toy dataset to manipulate sequence length.
NUM_ITERS=int               ## Num train iteration
FA={0,1}                    ## Enable Flash Attention
ZERO={0,1,2,3}              ## Stages of DeepSpeed Zero. 0 by default
ACT_CKPT={0,1}              ## Enable activation checkpointing
VIT3D={0,1}                 ## Switch to 3DVIT. Must use Toy dataset (for now).
VIT=string                  ## Choose different VIT model size. Refer to mds_launch.sh all options.
TPSP={0,1}                  ## Upgrade from TP to TP-SP
LOG_RESULTS={0,1}           ## log results (tflops, mem fpt, samples/sec) in a json file
MICS_SHARD_SIZE             ## Size of your MICS partition group (Needs custom DeepSpeed with MICS fix)
fp16                        ## enable fp16
bf16                        ## use datatype bf16
LOG_COMMS                   ## log/profile communications through deepspeed
PROF_FLOPS                  ## profile flop counts with detail through deepspeed
PROFILE={0,1}               ## Enable pytorch profiler. Trace is saved in your LOG_DIR.
GBS=int                     ## global batch size
MBS=int                     ## micro batch size
POS_ENCODING={0,1}          ## Use positioanl encoding instead of positional embedding
WANDB_MODE=disabled         ## Disable WANDB
GLOBAL_MEAN_POOLING=1       ## Use Global mean pooling instead of clf token
USP_ulysses=1, SP=          ## Turn on USP's Ulysses. Separately set SP degree
USP_ring=1, SP=             ## Turn on USP's Ring Attention. Separately set SP degree
USP_hybrid=(2,4)            ## TBD
SIZE=int                    ## Restraint Number of GPU (ONLY WORKS ON 1-NODE)
drop_last_batch_with_GBS=1  ## fixes the data order as long as GBS is matching.
################################ Notes ################################
1. Pass either GBS or MBS
2. Pass either fp16 or bf16
3. Without ZERO3, only fp16 can be used (for now).
```

## Environment
Only base environment is needed for polaris cluster while for aurora, sam's ezpz library will be needed on top of the base environment. A suitable virtual environment is activated automatically on aurora.

Environment outside of polaris and aurora is WIP..
