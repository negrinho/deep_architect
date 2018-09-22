# Main/Searcher 
import deep_architect.utils as ut
import deep_architect.search_logging as log 
from subprocess import check_output
import os

def done(job_id): 
    out = check_output(['squeue', '-j', job_id, '-O', 'state']) 
    return (len(out.split()) == 1) # another way is if checkout is ['STATE', 'PENDING']. But also risk not finishing job

def valid_filepath(result_fp): 
    return False if (not os.path.isfile(result_fp) or os.path.getsize(result_fp) == 0) else True 
        
def create_and_submit_job(i, exp_name, config_fp, evaluation_fp): 
    jobname = "%s_arch%d" % (exp_name, i)
    out_fp = "%s/slurm.out" % evaluation_fp
    result_fp = "%s/result.json" % evaluation_fp
    script = [
        '#!/bin/bash', 
        '#SBATCH --nodes=1', 
        '#SBATCH --partition=GPU-shared', 
        '#SBATCH --gres=gpu:p100:1',
	    '#SBATCH --ntasks-per-node=1', 
        '#SBATCH --time=00:15:00', 
        '#SBATCH --job-name=%s' % jobname, 
        '#SBATCH --output=%s' % out_fp, 
        'python evaluator.py --config %s > %s' % (config_fp, result_fp) 
    ]
    script_fp = "%s/run%d.sh" % (evaluation_fp, i) 
    ut.write_textfile(script_fp, script) 
    check_output(["chmod", "+x", script_fp]) 

    # submit bash job and obtain id 
    out = check_output(['sbatch', script_fp]) 
    return (out.split()[3], result_fp)

def create_sample(num_classes, num_samples, exp_name):
    # create sample script
    logger = log.SearchLogger('logs', exp_name, delete_if_exists=True, create_parent_folders=True)
    result_fp = 'logs/%s/sample_info.json' % exp_name
    out_fp = 'logs/%s/sample_slurm.out' % exp_name
    jobname = 'sampling_architecture_for_%s_experiment' % exp_name
    script = [
        '#!/bin/bash', 
        '#SBATCH --nodes=1', 
        '#SBATCH --partition=GPU-shared', 
        '#SBATCH --gres=gpu:p100:1',
	    '#SBATCH --ntasks-per-node=1', 
        '#SBATCH --time=00:30:00', 
        '#SBATCH --job-name=%s' % jobname,
        '#SBATCH --output=%s' % out_fp,
        'module load keras/2.2.0_tf1.7_py3_gpu ', 
        'source activate',
        'export PYTHONPATH=${PYTHONPATH}:/pylon5/ci4s8dp/maxle18/darch/', 
        'python examples/tutorials/multiworker/searcher.py --num_classes %d --num_samples %d --exp_name %s --result_fp %s' % (num_classes, num_samples, exp_name, result_fp) 
    ]
    script_fp = 'logs/%s/sample.sh' % exp_name 
    ut.write_textfile(script_fp, script) 
    check_output(["chmod", "+x", script_fp]) 

    # submit batch job and obtain id 
    job_id = check_output(['sbatch', script_fp]).split()[3]
    while True: # perhaps need to have a time guard here 
        if (done(job_id) and valid_filepath(result_fp)): 
            return ut.read_jsonfile(result_fp)
    return dict()
    
# bridges-server based main 
def main():

    num_classes = 10
    num_samples = 3 # number of architecture to sample 
    exp_name = 'mnist_multiworker'
    best_val_acc, best_architecture = 0., -1
    
    # create and submit batch job to sample architectures and get information back  
    architectures = create_sample(num_classes, num_samples, exp_name)
    
    # create and submit batch job to evaluate each architecture
    jobs = []
    for i in architectures:
        (config_fp, evaluation_fp) = architectures[i]["config_filepath"], architectures[i]["evaluation_filepath"]
        (job_id, result_fp) = create_and_submit_job(int(i), exp_name, config_fp, evaluation_fp)
        jobs.append((int(i), job_id, result_fp))

    # extract result and remove job 
    while(len(jobs) > 0) :
        for (arch_id, result_fp, job_id) in jobs: 
            if (done(job_id) and valid_filepath(result_fp)): 
                results = ut.read_jsonfile(fp) 
                val_acc = float(results['val_accuracy'])
                print("Finished evaluating architecture %d, validation accuracy is %f" % (arch_i, val_acc))
                if val_acc > best_val_acc: 
                    best_architecture, best_val_acc = arch_id, val_acc
            jobs.remove((arch_id, eval_logger, job_id))
    
    print("Best validation accuracy is %f with architecture %d" % (best_val_acc, best_architecture)) 

if __name__ == "__main__": 
    main() 
