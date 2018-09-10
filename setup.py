from setuptools import setup, find_packages
from codecs import open
from os import path
from pip.req import parse_requirements
import pip.download


install_reqs = parse_requirements("requirements.txt", session=pip.download.PipSession())
install_requires = [str(ir.req) for ir in install_reqs]


here = path.abspath(path.dirname(__file__))


# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='darch',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',

    description='Deep Architect',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/negrinho/darch',

    # Author details
    author='Renato Negrinho',
    author_email='negrinho@cs.cmu.edu',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='deep architect',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['darch'],

    install_requires=install_requires
)
