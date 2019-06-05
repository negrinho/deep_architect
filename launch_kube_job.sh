#! /bin/bash
# First argument is the yaml,
# Second argument is the name of the config,
# third argument is repetition number,
# fourth argument is the number of parallel jobs,
# fifth argument is config file

JOB_NAME="${2//_/-}"
SEARCH_NAME=$2
REPETITION=$3
NUM_JOBS=$4
CONFIG_FILE=$5
export JOB_NAME
export SEARCH_NAME
export NUM_JOBS
export CONFIG_FILE
export REPETITION
envsubst < $1 | kubectl apply -f -
