### running locally and on a server
import psutil
import subprocess

def cpus_total():
    return psutil.cpu_count()

def cpus_free():
    frac_free = (1.0 - 0.01 * psutil.cpu_percent())
    return int( np.round( frac_free * psutil.cpu_count() ) )

def convert_between_byte_units(x, src_units='b', dst_units='mb'):
     units = ['b', 'kb', 'mb', 'gb', 'tb']
     assert (src_units in units) and (dst_units in units)
     return x / float(
         2 ** (10 * (units.index(dst_units) - units.index(src_units))))

def memory_total(units='mb'):
    return convert_between_byte_units(psutil.virtual_memory().total, dst_units=units)
    
def memory_free(units='mb'):
    return convert_between_byte_units(psutil.virtual_memory().available, dst_units=units)

def gpus_total():
    try:
        n = len(subprocess.check_output(['nvidia-smi', '-L']).strip().split('\n'))
    except OSError:
        n = 0
    return n

def gpus_free():
    return len(gpus_free_ids())

def gpus_free_ids():
    try:
        out = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used', 
            '--format=csv,noheader'])

        ids = []

        gpu_ss = out.strip().split('\n')
        for i, s in enumerate( gpu_ss ):
            
            p_s, m_s = s.split(', ')
            p = float( p_s.split()[0] )
            m = float( m_s.split()[0] )

            # NOTE: this is arbitrary for now. maybe a different threshold for 
            # utilization makes sense. maybe it is fine as long as it is not 
            # fully utilized. memory is in MBs.
            if p < 0.1 and m < 100.0:
                ids.append(i)

        return ids

    except OSError:
        ids = []
    
    return ids

def gpus_set_visible(ids):
    n = gpus_total()
    assert all([i < n and i >= 0 for i in ids])
    subprocess.call(['export', 'CUDA_VISIBLE_DEVICES=%s' % ",".join(map(str, ids))])

