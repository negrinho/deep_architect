import subprocess
import os


### Auxiliary functions.
def run_bash_command(cmd):
    str_output = subprocess.check_output(cmd, shell=True)
    return str_output


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


def create_folder(folderpath, abort_if_exists=True,
                  create_parent_folders=False):
    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))
    assert not (abort_if_exists and folder_exists(folderpath))

    if not folder_exists(folderpath):
        os.makedirs(folderpath)


def join_paths(paths):
    return os.path.join(*paths)


def create_bash_script(cmd_lines, filepath):
    script_lines = ['#!/bin/bash'] + cmd_lines
    write_textfile(filepath, script_lines)
    run_bash_command('chmod +x %s' % filepath)


### Configurations for building the containers.
def get_key(is_py27, is_gpu):
    return ('py27' if is_py27 else 'py36', 'gpu' if is_gpu else 'cpu')


def get_config(is_py27, is_gpu):
    """Gets the config dictionary associated to the type of container to be built.
    """
    config_d = {
        ('py27', 'cpu'): {
            'key': ('py27', 'cpu'),
            'tag': 'all-py27-cpu',
            'extra_py_packages': [],
            'extra_apt_packages': ['python-tk'],
            'extra_bash_commands': [],
        },
        ('py27', 'gpu'): {
            'key': ('py27', 'gpu'),
            'tag': 'all-py27',
            'extra_py_packages': [],
            'extra_apt_packages': ['python-tk'],
            'extra_bash_commands': [],
        },
        ('py36', 'cpu'): {
            'key': ('py36', 'cpu'),
            'tag': 'all-py36-cpu',
            'extra_py_packages': [],
            'extra_apt_packages': ['python3-tk', 'libopenmpi-dev'],
            'extra_bash_commands': [],
        },
        ('py36', 'gpu'): {
            'key': ('py36', 'gpu'),
            'tag': 'all-py36',
            'extra_py_packages': [],
            'extra_apt_packages': ['python3-tk', 'libopenmpi-dev'],
            'extra_bash_commands': [],
        }
    }
    key = ('py27' if is_py27 else 'py36', 'gpu' if is_gpu else 'cpu')
    return key, config_d[key]


extra_py_packages = [
    # for documentation.
    'sphinx',
    'sphinx_rtd_theme',
    # # for dash visualizations (update later).
    # 'dash==0.21.0',
    # 'dash-renderer==0.12.1',
    # 'dash-html-components==0.10.0',
    # 'dash-core-components==0.22.1',
    # 'plotly --upgrade',
    # for multi-machine support.
    'mpi4py',
]

### NOTE: this is not fully tested
extra_bash_commands = [
    # 'apt-get update'
    # # for one of max's examples ; this is not fully tested.
    # '$PIP_INSTALL pathlib tqdm tables',
    # '$APT_INSTALL software-properties-common',
    # 'add-apt-repository ppa:ubuntugis/ppa',
    # 'apt-get update',
    # '$APT_INSTALL gdal-bin libgdal-dev',
    # '$PIP_INSTALL shapely[vectorized] rasterio',
]

extra_apt_packages = [
    'pandoc'
    # 'software-properties-common'
]


def create_singularity_container(config_d, out_folderpath):
    header_lines = [
        'Bootstrap: docker',
        'From: docker://ufoym/deepo:%s' % config_d['tag'],
    ]

    help_lines = [
        '%help',
        'This container contains the development environment for deep_architect.',
        'You should be able to run all the examples, generate documentation,',
        'and generate visualizations with it.',
    ]

    post_lines = [
        '%post',
        '    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"',
        '    APT_INSTALL="apt-get install -y --no-install-recommends"',
        '    apt-get update'
    ]
    if config_d["key"][1] == 'gpu':
        # necessary for the --nv flag in some cases.
        post_lines.append('    echo > /bin/nvidia-smi')

    for cmd in extra_bash_commands + config_d['extra_bash_commands']:
        post_lines.append('    %s' % cmd)
    for pkg in extra_apt_packages + config_d['extra_apt_packages']:
        post_lines.append('    $APT_INSTALL %s' % pkg)
    for pkg in extra_py_packages + config_d['extra_py_packages']:
        post_lines.append('    $PIP_INSTALL %s' % pkg)

    post_lines.extend([
        # cleanup lines
        '    ldconfig && \\',
        '        apt-get clean && \\',
        '        apt-get autoremove && \\',
        '        rm -rf /var/lib/apt/lists/* /tmp/* ~/*',
    ])

    runscript_lines = [
        '%runscript',
        '    export PYTHONPATH=".:$PYTHONPATH" && exec python "$@"'
    ]

    lines = []
    for lst in [header_lines, help_lines, post_lines, runscript_lines]:
        lines.extend(lst)
        lines.append('')

    recipe_filepath = join_paths([out_folderpath, 'Singularity'])
    write_textfile(recipe_filepath, lines)

    # script for creating the container.
    container_filepath = join_paths([out_folderpath, 'deep_architect.img'])
    create_bash_script([
        'sudo singularity build --sandbox %s %s' %
        (container_filepath, recipe_filepath)
    ], join_paths([out_folderpath, 'build.sh']))


# TODO: create the equivalent docker containers.
def create_docker_container(config_d, recipe_filepath):
    raise NotImplementedError


# TODO: change to a make file later.
def create_build_all_script(out_folderpath, folderpath_lst):
    create_bash_script(['./%s/build.sh' % path for path in folderpath_lst],
                       join_paths([out_folderpath, 'build.sh']))


def create_makefile(out_folderpath, container_config_lst):
    fn = lambda rule_name: {
        'rule_name': rule_name,
        'target_lst': [],
        'command_lst': []
    }
    rule_name_to_config = {
        'py27': fn('py27'),
        'py36': fn('py36'),
        'cpu': fn('cpu'),
        'gpu': fn('gpu'),
        'all': fn('all')
    }

    def add_to_lists(cfg):
        lst = [rule_name_to_config['all']]
        lst.append(rule_name_to_config['py27' if cfg['is_py27'] else 'py36'])
        lst.append(rule_name_to_config['gpu' if cfg['is_gpu'] else 'cpu'])
        for x in lst:
            x['target_lst'].append(cfg['target'])
            x['command_lst'].append(cfg['command'])

    for cfg in container_config_lst:
        assert cfg['is_singularity']  # only for singularity for now.
        cfg['target'] = join_paths([cfg['folderpath'], 'deep_architect.img'])
        cfg['command'] = join_paths(
            ['./%s' % join_paths([cfg['folderpath'], 'build.sh'])])
        add_to_lists(cfg)

    # create scripts for all commands.
    for k, d in rule_name_to_config.items():
        create_bash_script(d['command_lst'],
                           join_paths([out_folderpath,
                                       'build_%s.sh' % k]))

    create_bash_script(
        ['rm %s' % t for t in rule_name_to_config['all']['target_lst']],
        join_paths([out_folderpath, 'clean.sh']))


def main():
    container_folderpath_lst = []
    container_config_lst = []
    for is_py27 in [False, True]:
        for is_gpu in [False, True]:
            key, config_d = get_config(is_py27, is_gpu)
            out_folderpath = join_paths([
                'containers', 'singularity',
                '-'.join(['deep_architect'] + list(key))
            ])
            create_folder(
                out_folderpath,
                abort_if_exists=False,
                create_parent_folders=True)
            create_singularity_container(config_d, out_folderpath)
            container_folderpath_lst.append(out_folderpath)
            container_config_lst.append({
                'folderpath': out_folderpath,
                'is_singularity': True,
                'is_py27': is_py27,
                'is_gpu': is_gpu
            })
    create_build_all_script('containers', container_folderpath_lst)
    create_makefile('containers', container_config_lst)


if __name__ == '__main__':
    main()

# TODO: do the generation of the Docker containers. Should be similar to the singularity ones.
# TODO: add a file to create all the containers at once.
# NOTE: the makefile is not fully correct.
