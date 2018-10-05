if [ $# -ne 2 ]; then
    echo "./run_local.sh cfg_name num_workers"
    exit 1
fi
cfg_name="$1"
num_workers=$2
folderpath=dev/negrinho/multiworker/simplest
cfg_filepath="$folderpath/configs/$cfg_name.json"
# generate architectures
ipython $folderpath/master.py -- \
    --config_filepath "$cfg_filepath"

u=$(($num_workers - 1))
for worker_id in $(seq 0 $u); do
    ipython $folderpath/worker.py -- \
        --config_filepath "$cfg_filepath" \
        --worker_id $worker_id \
        --num_workers $num_workers &
done

# To run with GPU support, use export CUDA_VISIBLE_DEVICES=$worker_id
# TODO: it is not clear that it is using the GPU.