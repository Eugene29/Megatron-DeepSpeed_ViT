export NEWDS=1
LOGDIR=$(dirname $0 | xargs realpath)/logs
PYSCRIPT=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_launch.sh
DATA_PATH_LOG_PREFIX=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/logs/


## DEBUG
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

## BENCHMARKING & RUNS ARGUMENTS
# export WANDB_MODE="disabled"
# export PROFILE=1
export GLOBAL_MEAN_POOLING=1 # global mean pooling instead of clf token
# export CUDA_DEVICE_MAX_CONNECTIONS=1 ## TODO: What is this??
export GBS=4 ## always works after fix. 
# export MBS=1 ## MBS := MBS * SP in mds_args.sh to keep the Global Batch Size the same for DP & SP, 
# export DATA="Toy" #; export factor=250 ## Toy dataset with flexible img_size
# SIZE=1 NUM_ITER=15 FA=1 TOY_DATALOG=$DATA_PATH_LOG_PREFIX/data_consumed_DP1.log bash $PYSCRIPT |& tee $LOGDIR/mds1.log

## NOTES:
## VIT-TINY
# Max seq w/ FA 94 (32) -> 95 (OOM)
# Max seq w/out FA 660 (34) -> 665 (OOM)
# GBS only works when MP-degree = World Size

# NUM_EPOCHS=1 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
# factor=16 SIZE=1 NUM_EPOCHS=1 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log

# factor=64 NUM_EPOCHS=1 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log
# factor=64 SIZE=1 NUM_EPOCHS=1 FA=1 POS_ENCODING=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log

# factor=800 SIZE=1 NUM_EPOCHS=1 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
# factor=660 NUM_EPOCHS=1 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
# factor=850 NUM_EPOCHS=1 FA=1 SIZE=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
# factor=700 NUM_EPOCHS=1 FA=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds3.log
# factor=750 NUM_EPOCHS=1 FA=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds4.log
# factor=800 NUM_EPOCHS=1 FA=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds5.log
# factor=1200 NUM_EPOCHS=1 FA=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds6.log

# NUM_EPOCHS=1 FA=1 SP=4 USP_ulysses=1 PACKED=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log
# NUM_EPOCHS=1 FA=1 SP=4 USP_ulysses=1 PACKED=5D bash $PYSCRIPT |& tee $LOGDIR/mds2.log

# NUM_EPOCHS=30 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds.log
export GBS=512
SIZE=1 NUM_ITER=15 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds1.log
SP=1   NUM_ITER=15 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds2.log
SP=4   NUM_ITER=15 FA=1 bash $PYSCRIPT |& tee $LOGDIR/mds3.log

## Data check
# SIZE=1 NUM_ITER=15 FA=1 TOY_DATALOG=$DATA_PATH_LOG_PREFIX/data_consumed_DP1.log bash $PYSCRIPT |& tee $LOGDIR/mds1.log
# SP=1   NUM_ITER=15 FA=1 TOY_DATALOG=$DATA_PATH_LOG_PREFIX/data_consumed_DP4.log bash $PYSCRIPT |& tee $LOGDIR/mds2.log
# SP=4   NUM_ITER=15 FA=1 TOY_DATALOG=$DATA_PATH_LOG_PREFIX/data_consumed_SP4.log bash $PYSCRIPT |& tee $LOGDIR/mds3.log

# NUM_EPOCHS=25 FA=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# NUM_EPOCHS=25 FA=1 USP_ulysses=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log
# NUM_EPOCHS=1 USP_ring=1 SP=4 bash $PYSCRIPT |& tee $LOGDIR/mds.log

## Possible args:
# USP_ulysses=1, SP=2
# USP_ring=1, SP=2
# USP_hybrid=(2,4) ## TBD
# PACKED={1, 5D} ## Fused QKV all2all Ulysses
# DATA_PATH_LOG=$DATA_PATH_LOG/datafiles_consumed.log
# SIZE=1

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