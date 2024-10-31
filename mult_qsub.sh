
# export NUM_EPOCHS=500 ## Default
PYSCRIPT=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh ## Work on qsub

# QUEUE="debug"
# QUEUE="debug-scaling"
# QUEUE="prod"
QUEUE="preemptable"
PROJ="datascience"

hrs=8
nodes=2 #10
MP=8
MBS=128
# export NUM_ITERS=20

NGPUS=$(( nodes * 4 ))
DP=$(( NGPUS / MP ))
export GBS=$(( MBS * NGPUS ))

echo NGPUS: $NGPUS
echo DP: $DP
echo GBS: $GBS

## Utilize both DP and MP to speed up experiments. 
qsub -V -v LOGFNAME=mds_TP${MP}DP${DP}.log,TP=$MP -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $PYSCRIPT
qsub -V -v LOGFNAME=mds_SP${MP}DP${DP}.log,SP=$MP -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $PYSCRIPT
qsub -V -v LOGFNAME=mds_DP${NGPUS}.log          -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $PYSCRIPT

## INTERACTIVE
# qsub -I -l select=2 LOGFNAME=mds_TP8_DP4.log TP=8 bash /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh
# LOGFNAME=mds_TP8_DP4.log TP=8 bash /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh
