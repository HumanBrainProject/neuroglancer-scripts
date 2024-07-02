neuroglancer-scripts
====================

Tools for converting volumetric images and surface meshes to the pre-computed format of Neuroglancer_.


.. image:: https://img.shields.io/pypi/v/neuroglancer-scripts.svg
   :target: https://pypi.python.org/pypi/neuroglancer-scripts
   :alt: PyPI Version

.. image:: https://github.com/HumanBrainProject/neuroglancer-scripts/actions/workflows/tox.yaml/badge.svg
   :target: https://github.com/HumanBrainProject/neuroglancer-scripts/actions/workflows/tox.yaml
   :alt: Build Status

.. image:: https://codecov.io/gh/HumanBrainProject/neuroglancer-scripts/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/HumanBrainProject/neuroglancer-scripts
   :alt: Coverage Status

.. image:: https://readthedocs.org/projects/neuroglancer-scripts/badge/?version=latest
   :target: http://neuroglancer-scripts.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


Installation
------------

The easiest way to install the latest stable version of neuroglancer-scripts is
through ``pip``. Using a virtual environment is recommended:

.. code-block:: shell

   python3 -m venv venv/
   . venv/bin/activate
   pip install neuroglancer-scripts


Usage
-----

See the `documentation <http://neuroglancer-scripts.readthedocs.io/>`_.


Development
-----------

The code is hosted on https://github.com/HumanBrainProject/neuroglancer-scripts.

Useful commands for development:

.. code-block:: shell

  git clone https://github.com/HumanBrainProject/neuroglancer-scripts.git

  # Install in a virtual environment
  cd neuroglancer-scripts
  python3 -m venv venv/
  . venv/bin/activate
  pip install -e .[dev]

  # Tests
  pytest  # run tests
  pytest --cov=neuroglancer_scripts --cov-report=html  # detailed test coverage report
  tox  # run tests under all supported Python versions

  # Please install pre-commit if you intend to contribute
  pre-commit install  # install the pre-commit hook


Contributing
============

This repository uses `pre-commit`_ to ensure that all committed code follows minimal quality standards. Please install it and configure it to run as a pre-commit hook in your local repository (see above). Also, please note that the code quality checks may need a more recent version of Python than that required by neuroglancer_scripts itself (> 3.8 at the time of this writing).


.. _Neuroglancer: https://github.com/google/neuroglancer
.. _pre-commit: https://pre-commit.com/


Acknowledgments
===============

`cloud-volume <https://github.com/seung-lab/cloud-volume>`_ (BSD 3-Clause licensed) for compressed morton code and shard/minishard mask implementation.
