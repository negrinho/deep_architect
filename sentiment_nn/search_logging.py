
import sys
sys.path.append('../')

import json
import os
import shutil
import darch.surrogates as su
from six import iteritems

def read_jsonfile(fpath):
    with open(fpath, 'r') as f:
        d = json.load(f)
        return d

def write_jsonfile(d, fpath, sort_keys=False):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=sort_keys)

def path_prefix(path):
    return os.path.split(path)[0]

def join_paths(paths):
    return os.path.join(*paths)

def path_exists(path):
    return os.path.exists(path)

def file_exists(path):
    return os.path.isfile(path)

def folder_exists(path):
    return os.path.isdir(path)

def create_folder(folderpath, 
        abort_if_exists=True, create_parent_folders=False):

    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))
    assert not (abort_if_exists and folder_exists(folderpath))

    if not folder_exists(folderpath):
        os.makedirs(folderpath)

def delete_folder(folderpath, abort_if_nonempty=True, abort_if_notexists=True):
    assert folder_exists(folderpath) or (not abort_if_notexists)

    if folder_exists(folderpath):
        assert len(os.listdir(folderpath)) == 0 or (not abort_if_nonempty)
        shutil.rmtree(folderpath)
    else:
        assert not abort_if_notexists

### TODO: this can be more complicated by having different folders, but over
# complicated for now.
# class SearchLogger:
#     def __init__(self, out_folderpath, search_name):
#         self.out_folderpath = out_folderpath
#         self.search_name = search_name
#         self.current_eval_id = 1
    
#         self.search_folderpath = join_paths([out_folderpath, search_name])
#         self.search_data_folderpath = join_paths([
#             self.search_folderpath, 'search_data'])
#         self.evaluations_folderpath = join_paths([
#             self.search_folderpath, 'evaluations'])
        
#         assert folder_exists(folderpath) and not folder_exists(self.search_folderpath)
#         create_folder(self.search_folderpath)
#         create_folder(self.search_data_folderpath)
#         create_folder(self.evaluations_folderpath)

#         self.current_eval_folderpath = join_paths([
#             self.evaluations_folderpath, 'x%d' % self.current_eval_id])
#         self.current_eval_userdata_folderpath = join_paths([
#             self.current_eval_folderpath, 'user_data'])
#         create_folder(self.current_eval_folderpath)
#         create_folder(self.current_eval_userdata_folderpath)

#     def log(self, inputs, outputs, hs, cfg_d, vs, r):
#         feats = su.extract_features(inputs, outputs, hs)
#         feats = {'module_feats' : feats[0], 'connection_feats' : feats[1], 
#             'module_hyperp_feats' : feats[2], 'other_hyper_feats' : feats[3]}
#         write_jsonfile(feats, join_paths(self.c))



#     def log(self, d):
#         with open(self.path, "w") as fi:
#             tbd = {}
#             for val in self.include:
#                 tbd[val] = str(d.get(val, "NA"))
#             json.dump(tbd, fi, indent=4)

# # NOTE: something that seems reasonable is to move the user loggers to
# # some other place. this context dictionary, it should be able to do 
# # something with them. everything should be placed in the context dictionary.

# NOTE: the logger is extremely tied to the evaluation of a model and to the 
# results returned. a lot more sophistication is needed with some cases.

class SearchLogger:
    def __init__(self, folderpath, search_name):
        self.search_name = search_name
        self.current_eval_id = 1
    
        self.search_folderpath = join_paths([folderpath, search_name])        
        assert folder_exists(folderpath) and not folder_exists(self.search_folderpath)
        create_folder(self.search_folderpath)

    def log(self, inputs, outputs, hs, cfg_d, vs, r):
        current_eval_folderpath = join_paths([
            self.search_folderpath, 'x%d' % self.current_eval_id])
        create_folder(current_eval_folderpath)
        cef = current_eval_folderpath

        d = {}
        feats = su.extract_features(inputs, outputs, hs)
        feats = {'module_feats' : feats[0], 'connection_feats' : feats[1], 
            'module_hyperp_feats' : feats[2], 'other_hyper_feats' : feats[3]}
        d['hyperp_values'] = vs
        d['features'] = feats
        # d['searcher_token'] = cfg_d
        d['results'] = {k: float(v) for k, v in iteritems(r)}
        print d
        write_jsonfile(d, join_paths([cef, 'log.json']))
        vi.draw_graph(outputs.values(), True, True, out_folderpath=cef, print_to_screen=False)
        self.current_eval_id += 1

def load_logs(folderpath, search_name):
    search_folderpath = join_paths([folderpath, search_name])
    assert folder_exists(search_folderpath)

    d_logs = {}
    i = 1
    while True:
        eval_name = 'x%d' % i
        eval_folderpath = join_paths([search_folderpath, eval_name])
        if folder_exists(eval_folderpath):
            log_filepath = join_paths([eval_folderpath, 'log.json'])
            d_logs[eval_name] = read_jsonfile(log_filepath)
            i += 1
        else:
            break 
    return d_logs

# the most important part of this model is to make sure that it works in the 
# simplest things. it can be done at the sequence level.

import data as da
import evaluator as ev
import darch.searchers as se
import search_space as ss
import darch.visualization as vi

if __name__ == '__main__':
    delete_folder('test', abort_if_nonempty=False)
    data = da.load_data(True)
    searcher = se.MCTSearcher(ss.ss1_fn, 0.1)
    logger = SearchLogger('.', 'test')
    for _ in xrange(16):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs, data)
        print vs, r, cfg_d
        searcher.update(r['val_acc'], cfg_d)
        logger.log(inputs, outputs, hs, cfg_d, vs, r)
        # vi.draw_graph(outputs.values(), True)
