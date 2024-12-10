# export NEWDS=1 ## TODO: parse Deepspeed version to deprecate this flag. 
# SCRIPT_PTH=${BASH_SOURCE[0]}
SCRIPT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh
WORKING_DIR=$(dirname $SCRIPT_PTH | xargs realpath)
LOGDIR=$WORKING_DIR/logs
MAIN_SCRIPT=$WORKING_DIR/mds_launch.sh

export GBS=${GBS:-2048}
export GLOBAL_MEAN_POOLING=1
export drop_last_batch_with_GBS=1
export DATA=${DATA:-"CIFAR"}

## Avoid multi-node hangs
unset NCCL_COLLNET_ENABLE
unset NCCL_CROSS_NIC
unset NCCL_NET
unset NCCL_NET_GDR_LEVEL

## Basic Config
FA=1 POS_ENCODING=1 bash $MAIN_SCRIPT |& tee $LOGDIR/$LOGFNAME ## TODO: get rid of nested qsub? 

