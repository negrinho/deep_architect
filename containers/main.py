import argparse
import os

### Auxiliary functions.
def write_textfile(filepath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'

    with open(filepath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")

def file_exists(path):
    return os.path.isfile(path)

def folder_exists(path):
    return os.path.isdir(path)

def path_prefix(path):
    return os.path.split(path)[0]

def create_folder(folderpath,
        abort_if_exists=True, create_parent_folders=False):
    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))
    assert not (abort_if_exists and folder_exists(folderpath))

    if not folder_exists(folderpath):
        os.makedirs(folderpath)

def join_paths(paths):
    return os.path.join(*paths)

class CommandLineArgs:
    def __init__(self, argname_prefix=''):
        self.parser = argparse.ArgumentParser()
        self.argname_prefix = argname_prefix

    def add(self, argname, argtype, default_value=None, optional=False, help=None,
            valid_value_lst=None, list_valued=False):
        valid_types = {'int' : int, 'str' : str, 'float' : float}
        assert argtype in valid_types

        nargs = None if not list_valued else '*'
        argtype = valid_types[argtype]

        self.parser.add_argument('--' + self.argname_prefix + argname,
            required=not optional, default=default_value, nargs=nargs,
            type=argtype, choices=valid_value_lst, help=help)

    def parse(self):
        return vars(self.parser.parse_args())

    def get_parser(self):
        return self.parser

### Configurations for building the containers.
def get_key(is_py27, is_gpu):
    return  ('py27' if is_py27 else 'py36', 'gpu' if is_gpu else 'cpu')

def get_config(is_py27, is_gpu):
    """Gets the config dictionary associated to the type of container that
    """
    config_d = {
        ('py27', 'cpu') : {
            'tag' : 'all-py27-cpu',
            'extra_py_packages' : [],
            'extra_apt_packages' : [],
            },
        ('py27', 'gpu') : {
            'tag' : 'all-py27',
            'extra_py_packages' : [],
            'extra_apt_packages' : [],
            },
        ('py36', 'cpu') : {
            'tag' : 'all-py36-cpu',
            'extra_py_packages' : [],
            'extra_apt_packages' : [],
            },
        ('py36', 'gpu') : {
            'tag' : 'all-py36',
            'extra_py_packages' : [],
            'extra_apt_packages' : [],
            }
        }
    key = ('py27' if is_py27 else 'py36', 'gpu' if is_gpu else 'cpu')
    return key, config_d[key]

extra_py_packages = [
    # for documentation.
    'sphinx',
    'sphinx_rtd_theme',
    # for dash visualizations (update later)
    'dash==0.21.0',
    'dash-renderer==0.12.1',
    'dash-html-components==0.10.0',
    'dash-core-components==0.22.1',
    'plotly --upgrade',
]

extra_apt_packages = [
    'python-tk',
]

def create_singularity_container(config_d, recipe_filepath):
    header_lines = [
        'Bootstrap: docker',
        'From: docker://ufoym/deepo:%s' % config_d['tag'],
    ]

    help_lines = [
        '%help',
        'This container contains the development enviroment for darch.',
        'You should be able to run all the examples, generate documentation,',
        'and generate visualizations with it.',
    ]

    post_lines = [
        '%post',
        '    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"',
        '    APT_INSTALL="apt-get install -y --no-install-recommends"',
        '    echo > /bin/nvidia-smi', # necessary for the --nv flag in some cases.
        # cleanup lines
        '    ldconfig && \\',
        '        apt-get clean && \\',
        '        apt-get autoremove && \\',
        '        rm -rf /var/lib/apt/lists/* /tmp/* ~/*',
    ]
    for pkg in extra_py_packages + config_d['extra_py_packages']:
        post_lines.append('    $PIP_INSTALL %s' % pkg)
    for pkg in extra_apt_packages + config_d['extra_apt_packages']:
        post_lines.append('    $APT_INSTALL %s' % pkg)

    runscript_lines = [
        '%runscript',
        '    export PYTHONPATH=".:$PYTHONPATH" && exec python "$@"'
    ]

    lines = []
    for lst in [header_lines, help_lines, post_lines, runscript_lines]:
        lines.extend(lst)
        lines.append('')

    write_textfile(recipe_filepath, lines)

# TODO: create the equivalent docker containers.
def create_docker_container(config_d, recipe_filepath):
    raise NotImplementedError

def main():
    folderpath = join_paths(['containers', 'singularity'])
    create_folder(folderpath, False, True)
    for is_py27 in [False, True]:
        for is_gpu in [False, True]:
            key, config_d = get_config(is_py27, is_gpu)
            recipe_filepath = join_paths([folderpath, '-'.join(['darch'] + list(key))])
            create_singularity_container(config_d, recipe_filepath)

if __name__== '__main__':
    main()