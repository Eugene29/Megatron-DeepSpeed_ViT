## Clone 
```
  git clone https://github.com/Eugene29/Megatron-DeepSpeed_ViT.git
```

## Entry Scripts
To generally experiment with the repo, checkout `mult_exp.sh`.

To run ALCF4 benchmark, checkout `alcf4_bench*.sh` scripts. 

If you are running outside of **Polaris** or **Aurora** cluster, customize cluster specific variables at `#### CUSTOMIZE HERE ####` in `mds_launch.sh`. To set-up environment from scratch, refer to `requirements.txt`.

## List of Shorcut Variables to Set-up Megatron-DeepSpeed 
```
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
PROF_FLOPS                  ## profile flop counts with detail through ds-flops-profiler
PROFILE={0,1}               ## profile through pytorch profiler.
GBS=int                     ## global batch size
MBS=int                     ## micro batch size
POS_ENCODING={0,1}          ## Use positioanl encoding instead of positional embedding
WANDB_MODE=disabled         ## Disable WANDB
GLOBAL_MEAN_POOLING=1       ## Use Global mean pooling instead of clf token in VIT
drop_last_batch_with_GBS=1  ## fixes the data order as long as GBS is matching.
################################ Notes ################################
1. Pass either GBS or MBS
2. Pass either fp16 or bf16
3. Without ZERO3, only fp16 can be used (for now).
```
