SCRIPT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/grid_benchmark.sh
WORKING_DIR=$(dirname $SCRIPT_PTH | xargs realpath)
LOGDIR=$WORKING_DIR/logs
MAIN_SCRIPT=$WORKING_DIR/mds_launch.sh
mkdir -p $LOGDIR

################################ Global ARGUMENTS ################################
export WANDB_MODE=disabled
export drop_last_batch_with_GBS=1
export PROFILE=1
export LOG_RESULTS=1
export FA=1
export VIT3D=1
export DATA=TOY
export POS_ENCODING=1
export GLOBAL_MEAN_POOLING=1
export SIZE=$SIZE

echo WANDB_MODE: $WANDB_MODE
echo drop_last_batch_with_GBS: $drop_last_batch_with_GBS
echo PROFILE: $PROFILE
echo LOG_RESULTS: $LOG_RESULTS
echo FA: $FA
echo GBS: $GBS
echo VIT3D: $VIT3D
echo DATA: $DATA
echo POS_ENCODING: $POS_ENCODING

# export CUDA_DEVICE_MAX_CONNECTIONS=1 ## TODO: What is this??

## Below is to prevent multi-node hangs on Polaris-cluster
unset NCCL_COLLNET_ENABLE
unset NCCL_CROSS_NIC
unset NCCL_NET
unset NCCL_NET_GDR_LEVEL

################################ Scaling EXPERIMENT ################################
factors=(4 8 16 32 64 128 192)
models=(BASE HUGE 4B 12B)
num_models=${#models[@]}
PATCH_DIM=16

## MP = max(4, SIZE)
if [[ $SIZE -ge 4 ]]; then
    MP=4
else
    MP=$SIZE
fi
export GBS=$((SIZE / MP))

for model in ${models[@]}; do
    for factor in ${factors[@]}; do
        export VIT=$model
        export factor=$factor

        TP=$MP        method="TP"  bash $MAIN_SCRIPT |& tee $LOGDIR/benchmark.log
        # No model parallelisms on 1 GPU
        if [[ $SIZE -ne 1 ]]; then
            TP=$MP ZERO=2 method="TP_Z2"  bash $MAIN_SCRIPT |& tee $LOGDIR/benchmark.log
            SP=$MP        method="SPU"    bash $MAIN_SCRIPT |& tee $LOGDIR/benchmark2.log
            SP=$MP ZERO=2 method="SPU_Z2" bash $MAIN_SCRIPT |& tee $LOGDIR/benchmark3.log
        fi
    done
done

echo "grid benchmark finished!"