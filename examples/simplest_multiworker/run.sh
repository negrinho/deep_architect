#!/bin/bash
if [ $# -ne 2 ]; then
    echo "./run_local.sh cfg_name num_workers"
    exit 1
fi
cfg_name="$1"
num_workers=$2
# generate architectures
python examples/simplest_multiworker/master.py \
    --config_filepath "examples/simplest_multiworker/configs/$cfg_name.json"
# run the workers
u=$(($num_workers - 1))
for worker_id in $(seq 0 $u); do
    python examples/simplest_multiworker/worker.py \
        --config_filepath "examples/simplest_multiworker/configs/$cfg_name.json" \
        --worker_id $worker_id \
        --num_workers $num_workers &
done

# To run with GPU support, use export CUDA_VISIBLE_DEVICES=$worker_id