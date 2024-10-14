export NEWDS=1
LOGDIR=$(dirname $0 | xargs realpath)/logs
DIR=$(dirname $0 | xargs realpath)
PYSCRIPT=$DIR/mds_launch.sh
mkdir -p $LOGDIR

. /home/eku/venv/stable/bin/activate

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

# qsub -A datascience -q debug -l select=1 -l walltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
# qsub -A datascience -q debug-scaling -l select=1 -l walltime=01:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh

## 
export WANDB_MODE="disabled"
# export PROFILE=1
# export DATA="Toy" ## Toy dataset with flexible img_size
export GLOBAL_MEAN_POOLING=1 # global mean pooling instead of clf token
# bash $PYSCRIPT |& tee $LOGDIR/mds.log ## DP
# SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log ## DP

# factor=60 bash $PYSCRIPT |& tee $LOGDIR/mds.log
ZE_AFFINITY_MASK=0,1,2,3 NUM_EPOCHS=20 bash $PYSCRIPT |& tee $LOGDIR/mds.log

# ZE_AFFINITY_MASK=0,1,2,3 NUM_EPOCHS=20 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# ZE_AFFINITY_MASK=0,1,2,3 SP=4 NUM_EPOCHS=20 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# FA=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log

# bash $PYSCRIPT |& tee $LOGDIR/mds.log
# SP=4 unifiedSP=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# SP=4 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# SP=4 unifiedSP=1 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# NUM_EPOCHS=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# unifiedSP=1 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log

# DEBUG=SP NUM_EPOCHS=1 SP=4 unifiedSP=1 bash $PYSCRIPT
# DEBUG=SP NUM_EPOCHS=10 SP=4 bash $PYSCRIPT
# DEBUG=DP NUM_EPOCHS=10 unifiedSP=1 SIZE=1 bash $PYSCRIPT
# DEBUG=DP NUM_EPOCHS=10 SIZE=1 bash $PYSCRIPT
# DEBUG=SP NUM_EPOCHS=10 SP=4 unifiedSP=1 FA=1 bash $PYSCRIPT
# DEBUG=DP NUM_EPOCHS=10 SIZE=1 FA=1 bash $PYSCRIPT