import deep_architect.search_logging as sl
import deep_architect.utils as ut

### Loading the data.
def process_logs(log_lst):
    ds = []
    for i, log in enumerate(log_lst):
        d = log['results']
        # d.pop('sequences')
        # d['num_training_epochs'] = len(d['num_training_epochs'])
        d['evaluation_id'] = i
        ds.append(d)
    return ds

def get_logs():
    logs_folderpath = "/home/darshan/darch/temp/logs"
    # path_lst = ['logs/test_cifar10_short', 'logs/test_cifar10_medium', 'logs/test']
    path_lst = [ut.join_paths([logs_folderpath, p]) for p in ['random_enas_space', 'random_zoph_sp1', 'random_zoph_sp3']]
    print(path_lst)
    path_to_log = {p : process_logs(sl.read_search_folder(p)) for p in path_lst}
    path_to_log = {p.split('/')[-1]: path_to_log[p] for p in path_to_log}
    return path_to_log