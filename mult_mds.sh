export NEWDS=1
LOGDIR=$(dirname $0 | xargs realpath)/logs
PYSCRIPT=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_launch.sh

# DEBUG=SP bash $PYSCRIPT
# DEBUG=DP bash $PYSCRIPT

# DEBUG=SP SP=4 unifiedSP=1 bash $PYSCRIPT |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/SP.log
# DEBUG=SP SP=4 bash $PYSCRIPT |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/SP.log
# DEBUG=DP unifiedSP=1 bash $PYSCRIPT |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/DP.log
# DEBUG=DP NUM_EPOCHS=1 unifiedSP=1 bash $PYSCRIPT |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/DP.log

# DEBUG=DP SIZE=1 bash $PYSCRIPT |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/DP.log
# DEBUG=DP bash $PYSCRIPT |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/DP.log

# bash /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/SP.log
# bash /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh |& tee  /eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/DP.log

# qsub -A datascience -q debug -l select=1 -l logltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
# qsub -A datascience -q debug-scaling -l select=1 -l walltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh

## ARGUMENTS
# export WANDB_MODE="disabled"
# export PROFILE=1
export GLOBAL_MEAN_POOLING=1 # global mean pooling instead of clf token
export CUDA_DEVICE_MAX_CONNECTIONS=1 ## TODO: What is this??
export MBS=512 ## If SP, MBS := MBS * SP
# export DATA="Toy"; export factor=700 ## Toy dataset with flexible img_size
# 94 (32) -> 95 (OOM)
# 660 (34) -> 665 (OOM) w/ FA

NUM_EPOCHS=5 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# NUM_EPOCHS=5 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# NUM_EPOCHS=20 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# factor=70 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log

# NUM_EPOCHS=60 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# SP=4 unifiedSP=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# SP=4 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# SP=4 unifiedSP=1 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# unifiedSP=1 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log

# DEBUG=SP NUM_EPOCHS=1 SP=4 unifiedSP=1 bash $PYSCRIPT
# DEBUG=SP NUM_EPOCHS=10 SP=4 bash $PYSCRIPT
# DEBUG=DP NUM_EPOCHS=10 unifiedSP=1 SIZE=1 bash $PYSCRIPT
# DEBUG=DP NUM_EPOCHS=10 SIZE=1 bash $PYSCRIPT
# DEBUG=SP NUM_EPOCHS=10 SP=4 unifiedSP=1 FA=1 bash $PYSCRIPT
# DEBUG=DP NUM_EPOCHS=10 SIZE=1 FA=1 bash $PYSCRIPT