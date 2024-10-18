SCRIPT_DIR=/eagle/datascience/eku/MDS-VIT-CLEAN/examples_deepspeed/sequence_parallel
# OUTPUT_PTH=$SCRIPT_DIR/output
LOG_DIR=$SCRIPT_DIR/custom_log
ds_script=$SCRIPT_DIR/ds_pretrain_gpt_1.3B_seq_parallel_32k.sh

mkdir -p $LOG_DIR

export train_iter=400
# export train_iter=10
export drop_last_batch_with_GBS=1 ## reconcile the data order between DP and MP
num_gpu=4

## Zero-DP
# sp_size=1 zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_DP4.log bash $ds_script |& tee $LOG_DIR/ex_ds2.log
# sp_size=1 zero_stage=1 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_ZERO1.log bash $ds_script |& tee $LOG_DIR/ex_ds6.log
# sp_size=1 zero_stage=2 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_ZERO2.log bash $ds_script |& tee $LOG_DIR/ex_ds7.log
# sp_size=1 zero_stage=3 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_ZERO3.log bash $ds_script |& tee $LOG_DIR/ex_ds8.log

## parallelisms
SIZE=1 zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_DP1.log bash $ds_script |& tee $LOG_DIR/ex_ds1.log
sp_size=$num_gpu zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_SP4.log bash $ds_script |& tee $LOG_DIR/ex_ds3.log
tp_size=$num_gpu zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_TP4.log bash $ds_script |& tee $LOG_DIR/ex_ds4.log
pp_size=$num_gpu zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_PP4.log bash $ds_script |& tee $LOG_DIR/ex_ds5.log