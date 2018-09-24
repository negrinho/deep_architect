import subprocess
import os
import deep_architect.utils as ut

def is_environment_variable_defined(name):
    return name in os.environ

def set_environment_variable(name, value, abort_if_notexists=True):
    assert (not abort_if_notexists) or (name in os.environ)
    os.environ[name] = value

def get_environment_variable(name, abort_if_notexists=True):
    assert (not abort_if_notexists) or (name in os.environ)
    return os.environ.get(name, None)

def get_gpu_information():
    gpus = []
    try:
        convert_to_gigabytes = lambda x: ut.convert_between_byte_units(x,
            src_units='megabytes', dst_units='gigabytes')
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader']).decode('utf-8')

        gpu_s_lst = out.strip().split('\n')
        for i, s in enumerate(gpu_s_lst):
            utilization_s, memory_s, total_memory_s = s.split(', ')
            gpus.append({
                'gpu_id' : i,
                'gpu_utilization_in_percent' : float(utilization_s.split()[0]),
                'gpu_memory_utilization_in_gigabytes' : convert_to_gigabytes(float(memory_s.split()[0])),
                'gpu_total_memory_in_gigabytes' : convert_to_gigabytes(float(total_memory_s.split()[0]))
            })
    except OSError:
        pass
    return gpus

def get_available_gpu(memory_threshold_in_gigabytes, utilization_threshold_in_percent):
    gpus = get_gpu_information()
    for g in gpus:
        if (g['gpu_utilization_in_percent'] <= utilization_threshold_in_percent) and (
            g['gpu_memory_utilization_in_gigabytes'] <= memory_threshold_in_gigabytes):
            return g['gpu_id']
    return None

def get_total_num_gpus():
    try:
        n = len(subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n

def set_visible_gpus(gpu_ids):
    n = get_total_num_gpus()
    assert all([i < n and i >= 0 for i in gpu_ids])
    set_environment_variable('CUDA_VISIBLE_DEVICES', ",".join(map(str, gpu_ids)),
        abort_if_notexists=False)
