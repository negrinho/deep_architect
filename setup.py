from setuptools import setup
from setuptools import find_packages

long_description = """
DeepArchitect is an architecture search framework with a focus on modularity,
extensibility, composability, and ease of use.
DeepArchitect uses composable and modular operators to express search
spaces over computational graphs that are then passed to search algorithms that
sample architectures from them with the goal of maximizing a desired performance
metric.

We aim to impact the workflows of researchers and practitioners with DeepArchitect.
For researchers, DeepArchitect aims to make architecture search research more
reusable and reproducible by providing them with a modular framework that they
can use to implement new search algorithms and new search spaces while reusing
a large amount of existing code.
For practicioners, DeepArchitect aims to augment their workflow by providing them
with a tool that allows them to easily write a search space encoding the large
number of choices involved in designing an architecture and use a search
algorithm automatically find an architecture in the search space.

DeepArchitect has the following **main components**:

* a language for writing composable and expressive search spaces over computational
graphs in arbitrary domains (e.g., Tensorflow, Keras, Pytorch, and even
non deep learning frameworks such as scikit-learn and preprocessing pipelines);
* search algorithms that can be used for arbitrary search spaces;
* logging functionality to easily keep track of the results of a search;
* visualization functionality to explore and inspect logging information resulting
from a search experiment.
"""

setup(
    name='deep_architect',
    version='0.1.0',
    description=
    "DeepArchitect: Architecture search so easy that you'll think it's magic!",
    long_description=long_description,
    url='https://github.com/negrinho/darch',
    license='MIT',
    author='Renato Negrinho',
    author_email='negrinho@cs.cmu.edu',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=['deep_architect'],
    install_requires=[],
    extras_require={})
