#! /bin/bash -l

### Helper script for mds_qsub.sh
DIR=$(dirname $0) ## $0 works as intended since we are not in job env yet.

## ARGUMENTS (-DEFAULTS)
NUM_NODES=${NUM_NODES:-1}
DURATION=${DURATION:-1:00:00}

## OTHER CONFIGS
PROJECT=datascience
QUEUE=debug
# QUEUE=debug-scaling
TSTAMP=$(date "+%m.%d.%H%M")
RUN_NAME=$TSTAMP
# RUN_NAME="N${NUM_NODES}-${TSTAMP}"
# RUN_NAME="ViT-CLF-${RUN_NAME}"

## Dynamically overwrite arguments
echo "DIR=$DIR"
echo "QUEUE=$QUEUE"
echo "PROJECT=datascience"
echo "DURATION=$DURATION"
echo "TSTAMP=$TSTAMP"
echo "NUM_NODES=$NUM_NODES"
echo "RUN_NAME: ${RUN_NAME}"

## -V enables job-submit to inherit env variable, enabling inline variable assignment. 
qsub \
  -q "${QUEUE}" \
  -A "${PROJECT}" \
  -V \
  -N "${RUN_NAME}" \
  -l select="$NUM_NODES" \
  -l walltime="${DURATION}" \
  -l filesystems=eagle:home:grand \
   ${DIR}/mds_launch.sh |& tee job_logs/${RUN_NAME}.log
echo "JOB IS SUBMITTED!" ## Q. I don't remember seeing these being printed?