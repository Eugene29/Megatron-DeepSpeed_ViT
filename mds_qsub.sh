# export NEWDS=1 ## TODO: parse Deepspeed version to deprecate this flag. 
# SCRIPT_PTH=${BASH_SOURCE[0]}
# SCRIPT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh
# WORKING_DIR=$(dirname $SCRIPT_PTH | xargs realpath)
# LOGDIR=$WORKING_DIR/logs
# MAIN_SCRIPT=$WORKING_DIR/mds_launch.sh

# export GBS=${GBS:-2048}
# export GLOBAL_MEAN_POOLING=1
# export drop_last_batch_with_GBS=1
# export DATA=${DATA:-"CIFAR"}


## Basic Config
# FA=1 POS_ENCODING=1 bash $MAIN_SCRIPT |& tee $LOGDIR/$LOGFNAME ## TODO: get rid of nested qsub? 

PROJ="Aurora_deployment"
# QUEUE="debug"
QUEUE="lustre_scaling"
hrs=1
BASH_SCRIPT="/lus/flare/projects/Aurora_deployment/eku/Megatron-DeepSpeed_ViT/mult_mds.sh"

num_nodes=64
qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
# num_nodes=128
# qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
# num_nodes=256
# qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
# num_nodes=512
# qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT