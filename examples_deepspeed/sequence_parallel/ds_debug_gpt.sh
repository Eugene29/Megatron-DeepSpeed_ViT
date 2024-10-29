OUTPUT_PTH=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/examples_deepspeed/sequence_parallel/output
ds_script=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/examples_deepspeed/sequence_parallel/ds_pretrain_gpt_1.3B_seq_parallel_32k.sh

# ## DP=1
# SIZE=1 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_DP.log bash $ds_script

# ## DP=4
# DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_DP4.log bash $ds_script

# ## SP=4
sp_size=4 DATA_PATH_LOG=$OUTPUT_PTH/tokens_consumed_SP.log bash $ds_script
# 10.51