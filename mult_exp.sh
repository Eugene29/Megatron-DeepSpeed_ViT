[[ -z $PBS_O_WORKDIR ]] && PBS_O_WORKDIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
echo PBS_O_WORKDIR: $PBS_O_WORKDIR  # PBS_O_WORKDIR should be root of the repo
cd $PBS_O_WORKDIR
mkdir -p $PBS_O_WORKDIR/logs

# ------------------------------------------------------------------------------------ #
# Global Arguments
export WANDB_MODE=disabled
export NUM_ITERS=5
# export PROFILE=1  # enable torch profiler
# export PROF_FLOPS=1  # ds-flops-profiler from deepspeed
# export LOG_COMMS=1  # comms logger from deepspeed
export VIT="315M";
# export bf16=1
export fp16=1
export MBS=1
export FA=1

# ------------------------------------------------------------------------------------ #
# Launch mds_launch.sh
num_node=$(wc -l < $PBS_NODEFILE)

export factor=64; # 4K sequence
SP=1 bash $PBS_O_WORKDIR/mds_launch.sh |& tee $PBS_O_WORKDIR/logs/n${num_node}_factor${factor}.log

export factor=128; # 16K sequence
SP=1 bash $PBS_O_WORKDIR/mds_launch.sh |& tee $PBS_O_WORKDIR/logs/n${num_node}_factor${factor}.log
SP=4 bash $PBS_O_WORKDIR/mds_launch.sh |& tee $PBS_O_WORKDIR/logs/n${num_node}_factor${factor}.log