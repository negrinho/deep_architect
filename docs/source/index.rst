DeepArchitect: A Framework for Architecture Search
==================================================

DeepArchitect is a framework for architecture search in arbitrary domains.
DeepArchitect
was designed with a focus on **modularity**, **ease of use**,
**reusability**, and **extensibility**.
With DeepArchitect, we aim to impact the workflows of
both researchers and practitioners by reducing the burden resulting from
the large number of arbitrary choices that have to be made to design
deep learning models.
We recommend the reader to start with the :doc:`overview <readme>`,
tutorials
(e.g., :doc:`here <tutorials/search_space_constructs>`)
and simple examples
(e.g., `here <https://github.com/negrinho/darch/blob/master/examples/mnist_with_logging/main.py>`_)
to get a gist of the framework.

Currently, we support Tensorflow, Keras, and PyTorch.
See `here <https://github.com/negrinho/darch/tree/master/examples/framework_starters>`_
for minimal complete examples for each of these frameworks.
It should be straightforward to take these examples and adapt them for your
custom problem.
See :doc:`here <tutorials/new_frameworks>`
to learn how to support new frameworks, which should require minimal adaptation
of the existing framework helpers found
`here <https://github.com/negrinho/darch/tree/master/examples/framework_starters>`_.

Questions and bug reports should be done through Github issues.
See :doc:`here <contributing>` for details on how to contribute.

Overview
===============

.. toctree::

    readme


Tutorials
=========

.. toctree::

    tutorials


API documentation
=================

.. toctree::

    deep_architect_api


Contributing
============

.. toctree::

    contributing


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`