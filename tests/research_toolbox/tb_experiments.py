### command line arguments
import argparse
import tb_io

class CommandLineArgs:
    def __init__(self, args_prefix=''):
        self.parser = argparse.ArgumentParser()
        self.args_prefix = args_prefix
    
    def add(self, name, type, default=None, optional=False, help=None,
            valid_choices=None, list_valued=False):
        valid_types = {'int' : int, 'str' : str, 'float' : float}
        assert type in valid_types

        nargs = None if not list_valued else '*'
        type = valid_types[type]

        self.parser.add_argument('--' + self.args_prefix + name, 
            required=not optional, default=default, nargs=nargs, 
            type=type, choices=valid_choices, help=help)

    def parse(self):
        return self.parser.parse_args()
    
    def get_parser(self):
        return self.parser

### managing large scale experiments.
import itertools as it
import stat

def get_valid_name(folderpath, prefix):
    idx = 0
    while True:
        name = "%s%d" % (prefix, idx)
        path = join_paths([folderpath, name])
        if not path_exists(path):
            break
        else:
            idx += 1   
    return name

# generating the call lines for a code to main.
def generate_call_lines(main_filepath,
        argnames, argvals, 
        output_filepath=None, profile_filepath=None):

    sc_lines = ['python -u \\'] 
    # add the profiling instruction.
    if profile_filepath is not None:
        sc_lines += ['-m cProfile -o %s \\' % profile_filepath]
    sc_lines += ['%s \\' % main_filepath]
    # arguments for the call
    sc_lines += ['    --%s %s \\' % (k, v) 
        for k, v in it.izip(argnames[:-1], argvals[:-1])]
    # add the output redirection.
    if output_filepath is not None:
        sc_lines += ['    --%s %s \\' % (argnames[-1], argvals[-1]),
                    '    > %s 2>&1' % (output_filepath) ] 
    else:
        sc_lines += ['    --%s %s' % (argnames[-1], argvals[-1])] 
    return sc_lines

# all paths are relative to the current working directory or to entry folder path.
def create_run_script(main_filepath,
        argnames, argvals, script_filepath, 
        # entry_folderpath=None, 
        output_filepath=None, profile_filepath=None):
    
    sc_lines = [
        '#!/bin/bash',
        'set -e']
    # # change into the entry folder if provided.
    # if entry_folderpath is not None:
    #     sc_lines += ['cd %s' % entry_folderpath]
    # call the main function.
    sc_lines += generate_call_lines(**retrieve_values(locals(), 
        ['main_filepath', 'argnames', 'argvals', 
        'output_filepath', 'profile_filepath'])) 
    # change back to the previous folder if I change to some other folder.
    # if entry_folderpath is not None:
    #     sc_lines += ['cd -']
    write_textfile(script_filepath, sc_lines, with_newline=True)
    # give run permissions.
    st = os.stat(script_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(script_filepath, st.st_mode | exec_bits)

# NOTE: could be done more concisely with a for loop.
def create_runall_script(exp_folderpath):
    fo_names = list_folders(exp_folderpath, recursive=False, use_relative_paths=True)
    # print fo_names
    num_exps = len([n for n in fo_names if path_last_element(n).startswith('cfg') ])

    # creating the script.
    sc_lines = ['#!/bin/bash']
    sc_lines += [join_paths([exp_folderpath, "cfg%d" % i, 'run.sh']) 
        for i in xrange(num_exps)]

    # creating the run all script.
    out_filepath = join_paths([exp_folderpath, 'run.sh'])
    # print out_filepath
    write_textfile(out_filepath, sc_lines, with_newline=True)
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)

# NOTE: for now, this relies on the fact that upon completion of an experiment
# a results.json file is generated.
def create_runall_script_with_parallelization(exp_folderpath):
    fo_names = list_folders(exp_folderpath, recursive=False, use_relative_paths=True)
    # print fo_names
    num_exps = len([n for n in fo_names if path_last_element(n).startswith('cfg') ])

    ind = ' ' * 4
    # creating the script.
    sc_lines = [
        '#!/bin/bash',
        'if [ "$#" -lt 0 ] && [ "$#" -gt 3 ]; then',
        '    echo "Usage: run.sh [worker_id num_workers] [--force-rerun]"',
        '    exit 1',
        'fi',
        'force_rerun=0',
        'if [ $# -eq 0 ] || [ $# -eq 1 ]; then',
        '    worker_id=0',
        '    num_workers=1',
        '    if [ $# -eq 1 ]; then',
        '        if [ "$1" != "--force-rerun" ]; then',
        '            echo "Usage: run.sh [worker_id num_workers] [--force-rerun]"',
        '            exit 1',
        '        else',
        '            force_rerun=1',
        '        fi',
        '    fi',
        'else',
        '    worker_id=$1',
        '    num_workers=$2',
        '    if [ $# -eq 3 ]; then',
        '        if [ "$3" != "--force-rerun" ]; then',
        '            echo "Usage: run.sh [worker_id num_workers] [--force-rerun]"',
        '            exit 1',
        '        else',
        '            force_rerun=1',
        '        fi',
        '    fi',
        'fi',
        'if [ $num_workers -le $worker_id ] || [ $worker_id -lt 0 ]; then',
        '    echo "Invalid call: requires 0 <= worker_id < num_workers."',
        '    exit 1',
        'fi'
        '',
        'num_exps=%d' % num_exps,
        'i=0',
        'while [ $i -lt $num_exps ]; do',
        '    if [ $(($i % $num_workers)) -eq $worker_id ]; then',
        '        if [ ! -f %s ] || [ $force_rerun -eq 1 ]; then' % join_paths(
                    [exp_folderpath, "cfg$i", 'results.json']),
        '            echo cfg$i',
        '            %s' % join_paths([exp_folderpath, "cfg$i", 'run.sh']),
        '        fi',
        '    fi',
        '    i=$(( $i + 1 ))',
        'done'
    ]
    # creating the run all script.
    out_filepath = join_paths([exp_folderpath, 'run.sh'])
    # print out_filepath
    write_textfile(out_filepath, sc_lines, with_newline=True)
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)

# NOTE: not the perfect way of doing things, but it is a reasonable way for now.
# main_relfilepath is relative to the project folder path.
# entry_folderpath is the place it changes to before executing.
# if code and data folderpaths are provided, they are copied to the exp folder.
# all paths are relative I think that that is what makes most sense.
def create_experiment_folder(main_filepath,
        argnames, argvals_list, out_folderpath_argname, 
        exps_folderpath, readme, expname=None, 
        # entry_folderpath=None, 
        code_folderpath=None, 
        # data_folderpath=None,
        capture_output=False, profile_run=False):
    
    assert folder_exists(exps_folderpath)
    assert expname is None or (not path_exists(
        join_paths([exps_folderpath, expname])))
    # assert folder_exists(project_folderpath) and file_exists(join_paths([
    #     project_folderpath, main_relfilepath]))
    
    # TODO: add asserts for the part of the model that matters.

    # create the main folder where things for the experiment will be.
    if expname is None:    
        expname = get_valid_name(exps_folderpath, "exp")
    ex_folderpath = join_paths([exps_folderpath, expname])
    create_folder(ex_folderpath)

    # copy the code to the experiment folder.
    if code_folderpath is not None:
        code_foldername = path_last_element(code_folderpath)
        dst_code_fo = join_paths([ex_folderpath, code_foldername])

        copy_folder(code_folderpath, dst_code_fo, 
            ignore_hidden_files=True, ignore_hidden_folders=True, 
            ignore_file_exts=['.pyc'])

        # change main_filepath to use that new code.
        main_filepath = join_paths([ex_folderpath, main_filepath])

    # NOTE: no data copying for now, because it does not make much 
    # sense in most cases.
    data_folderpath = None ### TODO: remove later.
    # # copy the code to the experiment folder.    
    # if data_folderpath is not None:
    #     data_foldername = path_last_element(data_folderpath)
    #     dst_data_fo = join_paths([ex_folderpath, data_foldername])

    #     copy_folder(data_folderpath, dst_data_fo, 
    #         ignore_hidden_files=True, ignore_hidden_folders=True)

    # write the config for the experiment.
    write_jsonfile( 
        retrieve_values(locals(), [
        'main_filepath',
        'argnames', 'argvals_list', 'out_folderpath_argname', 
        'exps_folderpath', 'readme', 'expname',
        'code_folderpath', 'data_folderpath',
        'capture_output', 'profile_run']),
        join_paths([ex_folderpath, 'config.json']))

    # generate the executables for each configuration.
    argnames = list(argnames)
    argnames.append(out_folderpath_argname)
    for (i, vs) in enumerate(argvals_list): 
        cfg_folderpath = join_paths([ex_folderpath, "cfg%d" % i])
        create_folder(cfg_folderpath)

        # create the script
        argvals = list(vs)
        argvals.append(cfg_folderpath)
        call_args = retrieve_values(locals(), 
            ['argnames', 'argvals', 'main_filepath'])

        call_args['script_filepath'] = join_paths([cfg_folderpath, 'run.sh'])
        if capture_output:
            call_args['output_filepath'] = join_paths(
                [cfg_folderpath, 'output.txt'])
        if profile_run:
            call_args['profile_filepath'] = join_paths(
                [cfg_folderpath, 'profile.txt'])
        create_run_script(**call_args)

        # write a config file for each configuration
        write_jsonfile(
            create_dict(argnames, argvals), 
            join_paths([cfg_folderpath, 'config.json']))
    # create_runall_script(ex_folderpath)
    create_runall_script_with_parallelization(ex_folderpath)

    return ex_folderpath

def map_experiment_folder(exp_folderpath, fn):
    fo_paths = list_folders(exp_folderpath, recursive=False, use_relative_paths=False)
    num_exps = len([p for p in fo_paths if path_last_element(p).startswith('cfg') ])

    ps = []
    rs = []
    for i in xrange(num_exps):
        p = join_paths([exp_folderpath, 'cfg%d' % i])
        rs.append( fn(p) )
        ps.append( p )
    return (ps, rs)

def load_experiment_folder(exp_folderpath, json_filenames, 
    abort_if_notexists=True, only_load_if_all_exist=False):
    def _fn(cfg_path):
        ds = []
        for name in json_filenames:
            p = join_paths([cfg_path, name])

            if (not abort_if_notexists) and (not file_exists(p)):
                d = None
            # if abort if it does not exist, it is going to fail reading the file.
            else:
                d = read_jsonfile( p )
            ds.append( d )
        return ds
    
    (ps, rs) = map_experiment_folder(exp_folderpath, _fn)
    
    # filter only the ones that loaded successfully all files.
    if only_load_if_all_exist:
        proc_ps = []
        proc_rs = []
        for i in xrange( len(ps) ):
            if all( [x is not None for x in rs[i] ] ):
                proc_ps.append( ps[i] )
                proc_rs.append( rs[i] )
        (ps, rs) = (proc_ps, proc_rs)

    return (ps, rs)

### useful for generating configurations to run experiments.
import itertools 

def generate_config_args(d, ortho=False):
    ks = d.keys()
    if not ortho:
        vs_list = iter_product( [d[k] for k in ks] )
    else:
        vs_list = iter_ortho_all( [d[k] for k in ks], [0] * len(ks))
            
    argvals_list = []
    for vs in vs_list:
        proc_v = []
        for k, v in itertools.izip(ks, vs):
            if isinstance(k, tuple):

                # if it is iterable, unpack v 
                if isinstance(v, list) or isinstance(v, tuple):
                    assert len(k) == len(v)

                    proc_v.extend( v )

                # if it is not iterable, repeat a number of times equal 
                # to the size of the key.
                else:
                    proc_v.extend( [v] * len(k) )

            else:
                proc_v.append( v )

        argvals_list.append( proc_v )

    # unpacking if there are multiple tied argnames
    argnames = []
    for k in ks:
        if isinstance(k, tuple):
            argnames.extend( k )
        else: 
            argnames.append( k )
    
    # guarantee no repeats.
    assert len( set(argnames) ) == len( argnames ) 
    
    # resorting the tuples according to sorting permutation.
    idxs = argsort(argnames, [ lambda x: x ] )
    argnames = apply_permutation(argnames, idxs)
    argvals_list = [apply_permutation(vs, idxs) for vs in argvals_list]

    return (argnames, argvals_list)

# NOTE: this has been made a bit restrictive, but captures the main functionality
# that it is required to generate the experiments.
def copy_regroup_config_generator(d_gen, d_update):
    
    # all keys in the regrouping dictionary have to be in the original dict
    flat_ks = []     
    for k in d_update:
        if isinstance(k, tuple):
            assert all( [ ki in d_gen for ki in k ] )
            flat_ks.extend( k )
        else:
            assert k in d_gen
            flat_ks.append( k )

    # no tuple keys. NOTE: this can be relaxed by flattening, and reassigning,
    # but this is more work. 
    assert all( [ not isinstance(k, tuple) for k in d_gen] )
    # no keys that belong to multiple groups.
    assert len(flat_ks) == len( set( flat_ks ) ) 

    # regrouping of the dictionary.
    proc_d = dict( d_gen )
    
    for (k, v) in d_update.iteritems():
        
        # check that the dimensions are consistent.
        assert all( [ 
            ( ( not isinstance(vi, tuple) ) and ( not isinstance(vi, tuple) ) ) or 
            len( vi ) == len( k )
                for vi in v ] )

        if isinstance(k, tuple):

            # remove the original keys
            map(proc_d.pop, k)
            proc_d[ k ] = v

        else:

            proc_d[ k ] = v

    return proc_d

### parallelizing on a single machine
import psutil
import multiprocessing
import time

def mbs_process(pid):
    psutil_p = psutil.Process(pid)
    mem_p = psutil_p.memory_info()[0]
    
    return mem_p

def run_guarded_experiment(maxmemory_mbs, maxtime_secs, experiment_fn, **kwargs):
        start = time.time()
        
        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        p.start()
        while p.is_alive():
            p.join(1.0)     
            try:
                mbs_p = mbs_process(p.pid)
                if mbs_p > maxmemory_mbs:
                    print "Limit of %0.2f MB exceeded. Terminating." % maxmemory_mbs
                    p.terminate()

                secs_p = time.time() - start
                if secs_p > maxtime_secs:
                    print "Limit of %0.2f secs exceeded. Terminating." % maxtime_secs
                    p.terminate()
            except psutil.NoSuchProcess:
                pass

# TODO: probably this works better with keyword args.
# NOTE: also, probably the way this is done is not the best way because how things 
# work.
def run_parallel_experiment(experiment_fn, iter_args):            
    ps = []                                 
    for args in iter_args:
        p = multiprocessing.Process(target=experiment_fn, args=args)
        p.start()
        ps.append(p)
    
    for p in ps:
        p.join()

# for keeping track of configurations and results for programs
class ArgsDict:
    def __init__(self, fpath=None):
        self.d = {}
        if fpath != None:
            self._read(fpath)
    
    def set_arg(self, key, val, abort_if_exists=True):
        assert (not self.abort_if_exists) or key not in self.d 
        assert (type(key) == str and len(key) > 0)
        self.d[key] = val
    
    def write(self, fpath):
        write_jsonfile(self.d, fpath)
    
    def _read(self, fpath):
        return tb_io.read_jsonfile(fpath)

    def get_dict(self):
        return dict(self.d)

class ConfigDict(ArgsDict):
    pass

class ResultsDict(ArgsDict):
    pass

class SummaryDict:
    def __init__(self, fpath=None, abort_if_different_lengths=False):
        self.abort_if_different_lengths = abort_if_different_lengths
        if fpath != None:
            self.d = self._read(fpath)
        else:
            self.d = {}
        self._check_consistency()
    
    def append(self, d):
        for k, v in d.iteritems():
            assert type(k) == str and len(k) > 0
            if k not in self.d:
                self.d[k] = []
            self.d[k].append(v)
        
        self._check_consistency()
    
    ### NOTE: I don't think that read makes sense anymore because you 
    # can keep think as a json file or something else.
    def write(self, fpath):
        write_jsonfile(self.d, fpath)
    
    def _read(self, fpath):
        return tb_io.read_jsonfile(fpath)
    
    def _check_consistency(self):
        assert (not self.abort_if_different_lengths) or (len(
            set([len(v) for v in self.d.itervalues()])) <= 1)

    def get_dict(self):
        return dict(self.d)
