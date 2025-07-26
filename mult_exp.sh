# PBS_O_WORKDIR should be root of the repo
[[ -z $PBS_O_WORKDIR ]] && PBS_O_WORKDIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
echo PBS_O_WORKDIR: $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
mkdir -p $PBS_O_WORKDIR/logs

################################ Global ARGUMENTS ################################
num_node=$(wc -l < $PBS_NODEFILE)
export drop_last_batch_with_GBS=1
export GLOBAL_MEAN_POOLING=1
export WANDB_MODE=disabled
export POS_ENCODING=1
export NUM_ITERS=5
# export PROFILE=1
# export MICS_SHARD_SIZE=12
# export PROF_FLOPS=1
# export LOG_COMMS=1
# export bf16=1
export fp16=1
export FA=1
export MBS=1

################################# RUNS #################################
export DATA=TOY;
export VIT="315M";

# export factor=64; # 4K sequence
# SP=1 bash $PBS_O_WORKDIR/mds_launch.sh |& tee $PBS_O_WORKDIR/logs/n${num_node}_factor${factor}.log

export factor=128; # 16K sequence
SP=1 bash $PBS_O_WORKDIR/mds_launch.sh |& tee $PBS_O_WORKDIR/logs/n${num_node}_factor${factor}.log
# SP=4 bash $PBS_O_WORKDIR/mds_launch.sh |& tee $PBS_O_WORKDIR/logs/n${num_node}_factor${factor}.log