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
export VIT3D=1;
# export bf16=1
export fp16=1
export ZERO=1
export MBS=1
export FA=1

# ------------------------------------------------------------------------------------ #
# FOM RUNS
num_node=$(wc -l < $PBS_NODEFILE)

# seq : 4k
SP=1 NUM_ITERS=40 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log

# seq : 13K
export factor=24;
SP=1 NUM_ITERS=40 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log

# seq : 32K
export factor=32;
SP=1 NUM_ITERS=40 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log

# seq : 262K
export factor=64;
SP=4 NUM_ITERS=20 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log

# seq: 2M 
export factor=120;
SP=24 NUM_ITERS=20 bash $MAIN_SCRIPT |& tee $LOGDIR/n${num_node}_factor${factor}.log