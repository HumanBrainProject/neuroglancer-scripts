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

This project uses ``pytest`` and ``tox`` for testing. You will need to install
the project (a virtual environment is recommended) before you can use it or run
the tests. ``pip install -e`` will install in editable mode, so you do not need
to re-install when changing existing files. The ``[dev]`` suffix pulls extra
dependencies that are useful for development.

.. code-block:: shell

   git clone https://github.com/HumanBrainProject/neuroglancer-scripts.git
   cd neuroglancer-scripts
   python3 -m venv venv/
   . venv/bin/activate
   pip install -e .[dev]

   pytest --cov=neuroglancer_scripts --cov-report=html --cov-report=term


It is advised to use ``tox`` for testing the code against all supported Python
versions. This will run the same tests as the Travis continuous integration:

.. code-block:: shell

   tox
