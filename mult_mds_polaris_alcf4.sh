#!/bin/bash -l
#PBS -A datascience
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -l select=32
#PBS -q prod
#PBS -N vit_alcf4
#PBS -k doe

module use /soft/modulefiles
module load conda; conda activate base
SCRIPT_DIR=$(dirname "/eagle/datascience/vsastry/Vit_Pipeline/alcf4ViTbenchmark/latest_vit/Megatron-DeepSpeed_ViT/mult_mds.sh" | xargs realpath) ## Better for qsubing
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
export ranks_per_node=4
export MBS=1

################################# FOM RUNS #################################
export VIT3D=1; export factor=16; export IMG_W=128; export IMG_H=128; export DATA=TOY; export VIT="LARGE+"; #; export ACT_CKPT=1;
export ZERO=3
# SEq : 4k
SP=1 NUM_ITERS=40 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_nodes}_factor${factor}.log

# SEq : 13K
export factor=24;
SP=1 NUM_ITERS=40 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_nodes}_factor${factor}.log

#Seq : 32K
export factor=32;
SP=1 NUM_ITERS=40 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_nodes}_factor${factor}.log

#Seq : 262K
export factor=64;
export SP=4
SP=4 NUM_ITERS=20 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_nodes}_factor${factor}.log

# Seq: 2M 
export factor=120;
export SP=24
SP=24 NUM_ITERS=20 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_nodes}_factor${factor}.log
