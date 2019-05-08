#! /bin/bash


JOB_NAME="${2//_/-}"
SEARCH_NAME=$2
export JOB_NAME
export SEARCH_NAME
envsubst < $1 | kubectl apply -f -
