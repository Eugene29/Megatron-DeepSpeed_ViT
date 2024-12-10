### Useful script for submitting many jobs/experiments

### Q. What can we configure in this Script?
### A. We can configure:
###     hrs
###     nodes
###     -           Choose TP or SP by passing -v TP= or SP= (need to have no space between variables)
###     MP:         Choose TP or SP degree with MP
###     GBS or MBS  Pass either GBS or MBS
###     LOGFNAME:   Log std. output and std. error

# export NUM_EPOCHS=500 ## Default
BASH_SCRIPT=/eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh ## Work on qsub

# QUEUE="debug"
# QUEUE="debug-scaling"
# QUEUE="prod"
QUEUE="preemptable"
PROJ="datascience"

hrs=8
nodes=1 #10
MP=4
export GBS=2048
# MBS=128
# export GBS=$(( MBS * NGPUS ))
# export NUM_ITERS=20

NGPUS=$(( nodes * 4 ))
DP=$(( NGPUS / MP ))

echo NGPUS: $NGPUS
echo DP: $DP
echo GBS: $GBS

## Example Runs
## Q. Double check whether TP has a higher worse convergence
qsub -V -v LOGFNAME=qsub_n${nodes}_TP${MP}_DP${DP}.log,TP=$MP -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
qsub -V -v LOGFNAME=qsub_n${nodes}_SP${MP}_DP${DP}.log,SP=$MP -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
qsub -V -v LOGFNAME=qsub_n${nodes}_DP${NGPUS}.log             -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT

## Q. Does scaling it to two nodes make a difference in convergence somehow?
# Submit different configuration jobs
MP=8
nodes=2
qsub -V -v LOGFNAME=qsub_n${nodes}_TP${MP}_DP${DP}.log,TP=$MP -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
qsub -V -v LOGFNAME=qsub_n${nodes}_SP${MP}_DP${DP}.log,SP=$MP -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT
qsub -V -v LOGFNAME=qsub_n${nodes}_DP${NGPUS}.log             -A $PROJ -q $QUEUE -l select=$nodes -l walltime=$hrs:00:00,filesystems=eagle:home $BASH_SCRIPT


## IGNORE BELOW
## INTERACTIVE 
# qsub -I -l select=2 LOGFNAME=mds_TP8_DP4.log TP=8 bash /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh
# LOGFNAME=mds_TP8_DP4.log TP=8 bash /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mds_qsub.sh
