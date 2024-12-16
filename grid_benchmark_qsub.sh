BASH_SCRIPT=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/grid_benchmark.sh ## Work on qsub
PROJ="datascience"

## reset stored results
# echo "{}" > /eagle/projects/datascience/eku/Megatron-DeepSpeed_ViT/logs/results.json ## reset results.json

## INTERACTIVE LAUNCH
arr_num_gpus=(16)
# arr_num_gpus=(4)
for num_gpus in ${arr_num_gpus[@]}; do
    SIZE=$num_gpus bash $BASH_SCRIPT
done

## QSUB (Multi-node)
# hrs=3 ##
# arr_num_nodes=(2 4 8 16)
# # arr_num_nodes=(16)
# for num_nodes in ${arr_num_nodes[@]}; do
#     if [[ $num_nodes -lt 10 ]]; then
#         QUEUE="preemptable"
#     else
#         QUEUE="prod"
#     fi
#     export SIZE=$((num_nodes * 4))
#     qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home "$BASH_SCRIPT"
# done

## QSUB (Single-node)
# hrs=2 ##
# num_nodes=1
# SIZES=(1)
# QUEUE="preemptable"

# for SIZE in ${SIZES[@]}; do
#     export SIZE
#     qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home "$BASH_SCRIPT"
# done

# export SIZE
# qsub -V -A $PROJ -q $QUEUE -l select=$num_nodes -l walltime=$hrs:00:00,filesystems=eagle:home "$BASH_SCRIPT"