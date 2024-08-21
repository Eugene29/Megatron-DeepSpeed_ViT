 #! /bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
PROJECT="datascience"

# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG
DIR="$( cd -P "$( dirname "$BASH_SOURCE[0]" )" >/dev/null 2>&1 && pwd )"

QUEUE=$1
NUM_NODES=$2
DURATION=$3
TSTAMP=$(date "+%Y-%m-%d-%H%M%S")

RUN_NAME="N${NUM_NODES}-${TSTAMP}"
RUN_NAME="VIT-CLASS-${RUN_NAME}"


echo "QUEUE=$QUEUE"
echo "PROJECT=datascience"
echo "DURATION=$DURATION"
echo "TSTAMP=$TSTAMP"
echo "NUM_NODES=$NUM_NODES"
echo "RUN_NAME: ${RUN_NAME}"

qsub \
  -q "${QUEUE}" \
  -A "${PROJECT}" \
  -V \
  -N "${RUN_NAME}" \
  -l select="$NUM_NODES" \
  -l walltime="${DURATION}" \
  -l filesystems=eagle:home:grand \
  "${DIR}/mds_launch.sh"

