#! /bin/bash -l


# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG
DIR=$(dirname $0)

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
echo "JOB IS SUBMITTED!"