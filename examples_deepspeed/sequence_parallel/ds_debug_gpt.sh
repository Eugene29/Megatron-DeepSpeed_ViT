OUTPUT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/examples_deepspeed/sequence_parallel/output
LOG_DIR=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/examples_deepspeed/sequence_parallel/log
ds_script=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/examples_deepspeed/sequence_parallel/ds_pretrain_gpt_1.3B_seq_parallel_32k.sh

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_PTH

# ## DP=1
# SIZE=1 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_DP.log bash $ds_script

# ## DP=4
zero_stage=0 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_DP4.log bash $ds_script |& tee $LOG_DIR/ex_ds1.log

zero_stage=1 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_DP4.log bash $ds_script |& tee $LOG_DIR/ex_ds2.log

zero_stage=2 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_DP4.log bash $ds_script |& tee $LOG_DIR/ex_ds3.log

zero_stage=3 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_DP4.log bash $ds_script |& tee $LOG_DIR/ex_ds4.log

# ## SP=4
# sp_size=4 zero_stage=0 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_SP.log bash $ds_script |& tee $LOG_DIR/ex_ds1.log ## 10.51

# sp_size=4 zero_stage=1 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_SP.log bash $ds_script |& tee $LOG_DIR/ex_ds2.log ## 7.48

# sp_size=4 zero_stage=2 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_SP.log bash $ds_script |& tee $LOG_DIR/ex_ds3.log ## 7.48

# sp_size=4 zero_stage=3 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_SP.log bash $ds_script |& tee $LOG_DIR/ex_ds4.log ## 8.98