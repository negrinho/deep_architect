
# command, job name, num cpus, memory in mbs, time in minutes
# limits: 4GB per cpu, 48 hours,
# NOTE: read https://www.psc.edu/bridges/user-guide/running-jobs for more details.
ut_submit_bridges_cpu_job_with_resources() {
    script='#!/bin/bash'"
#SBATCH --nodes=1
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=$3
#SBATCH --mem=$4MB
#SBATCH --time=$5
#SBATCH --job-name=\"$2\"
$1" && echo \"$script\" > _run.sh && chmod +x _run.sh && sbatch _run.sh && rm _run.sh;
}

# 1: command, 2: job name, 3: num cpus, 4: num_gpus, 5: memory in mbs, 6: time in minutes
# limits: 7GB per gpu, 48 hours, 16 cores per gpu,
# NOTE: read https://www.psc.edu/bridges/user-guide/running-jobs for more details.
ut_submit_bridges_gpu_job_with_resources() {
    script='#!/bin/bash'"
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:k80:$4
#SBATCH --cpus-per-task=$3
#SBATCH --mem=$5MB
#SBATCH --time=$6
#SBATCH --job-name=\"$2\"
$1" && echo \"$script\" > _run.sh && chmod +x _run.sh && sbatch _run.sh && rm _run.sh;
}



cfg_name="$1"
num_workers=2
folderpath=dev/negrinho/multiworker/simplest
cfg_folderpath="$folderpath/configs/"
ipython $folderpath/master.py -- \
    --config_filepath "$cfg_folderpath/$1.json"
for worker_id in {1..}
    export CUDA_VISIBLE_DEVICES=$worker_id &&
    ipython $folderpath/worker.py -- \
        --config_filepath $FOLDERPATH/configs/debug.json \
        --worker_id $worker_id

### NOTE: this is this.

# TODO: needs to use singularity.
# let us make sure that it works locally.

# needs to use singularity.