neuroglancer-scripts
====================

Tools for converting 3D images to the Neuroglancer pre-computed format.


.. image:: https://img.shields.io/pypi/v/neuroglancer-scripts.svg
   :target: https://pypi.python.org/pypi/neuroglancer-scripts
   :alt: PyPI Version

.. image:: https://travis-ci.org/HumanBrainProject/neuroglancer-scripts.svg?branch=master
   :target: https://travis-ci.org/HumanBrainProject/neuroglancer-scripts
   :alt: Build Status

.. image:: https://codecov.io/gh/HumanBrainProject/neuroglancer-scripts/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/HumanBrainProject/neuroglancer-scripts
   :alt: Coverage Status

.. image:: https://readthedocs.org/projects/neuroglancer-scripts/badge/?version=latest
   :target: http://neuroglancer-scripts.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


Installation
------------

The easiest way to install neuroglancer-scripts is in a virtual environment:

.. code-block:: shell

   python3 -m venv venv/
   . venv/bin/activate
   pip install neuroglancer-scripts


Usage
-----

See the `documentation <http://neuroglancer-scripts.readthedocs.io/>`_.


Development
-----------

The code is hosted `on GitHub
<https://github.com/HumanBrainProject/neuroglancer-scripts>`_.

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

This repository uses `pre-commit`_ to ensure that all committed code follows minimal quality standards. Please install it and configure it to run as a pre-commit hook in your local repository (see above).


.. _pre-commit: https://pre-commit.com/
