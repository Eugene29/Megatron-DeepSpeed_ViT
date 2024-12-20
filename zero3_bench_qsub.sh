DIR=$(dirname ${BASH_SOURCE[0]})
BASH_SCRIPT=$(realpath $DIR/zero3_bench.sh)
LOG_DIR="$DIR/logs"
PROJ="datascience"
mkdir -p $LOG_DIR

## INTERACTIVE LAUNCH (TODO)
# arr_num_nodes=(2)
# arr_num_nodes=(2)

## QSUB (Multi-node)
hrs=1 
num_gpus_per_node=4 ## Polaris
# safe_model_sizes=( 8B 13B 17B 22B ) ## safe model sizes for nodes 2, 4, 8, 16
# risky_model_sizes=( 9B 14B 20B 25B )
# arr_num_nodes=( 2 4 8 16 )
safe_model_sizes=( 22B 26B ) ## safe model sizes for nodes 2, 4, 8, 16
risky_model_sizes=( 25B 28B )
arr_num_nodes=( 16 32 )
# safe_model_sizes=( 5B ) ## safe model sizes for nodes 2, 4, 8, 16
# risky_model_sizes=( 5.6B )
# arr_num_nodes=( 1 )

# wait_til_queue_is_available() {
#     while [[ $(qstat -u eku | awk 'END {print $10}') =~ ^(E|R|Q)$ ]]; do
#         echo "Found debug-s with E, R, or Q status"
#         sleep 5m
#     done
#     echo "No debug-s with E, R, or Q status"
# }

# wait_til_queue_is_available


# for i in ${!arr_num_nodes[@]}; do
for (( i=${#arr_num_nodes[@]}-1; i>=0; i-- )); do
    if [[ $num_nodes -lt 10 ]]; then
        QUEUE="debug-scaling"
        # QUEUE="debug" ##
    else
        QUEUE="prod"
    fi

    num_nodes=${arr_num_nodes[i]}
    export GBS=$((num_nodes * num_gpus_per_node)) ## Equivalent to num_gpus and DP degree
    export safe_model_size=${safe_model_sizes[i]}
    export risky_model_size=${risky_model_sizes[i]}
    # SAVE_SAFE_LOG_TO=""
    # SAVE_RISKY_LOG_TO="$LOG_DIR/n${num_nodes}_VIT${risky_model_size}"
    # SAFE_ENV_ARGS="GBS=$GBS,VIT=$safe_model_size"
    # RISKY_ENV_ARGS="GBS=$GBS,VIT=$risky_model_size"
    # echo SAFE_ENV_ARGS: $SAFE_ENV_ARGS
    # wait_til_queue_is_available
    # export VIT=$safe_model_size
    # export SAVE_LOG_TO="$LOG_DIR/n${num_nodes}_VIT${safe_model_size}.log" ## 
    qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
    # bash $BASH_SCRIPT

    # export SAVE_LOG_TO=$LOG_DIR/$SAVE_SAFE_LOG_TO
    # qsub -V -v $SAFE_ENV_ARGS -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home "$BASH_SCRIPT"
    # qsub -V -v $RISKY_ENV_ARGS -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home "$BASH_SCRIPT"
done