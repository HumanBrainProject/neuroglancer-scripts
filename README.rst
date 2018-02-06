.. image:: https://travis-ci.org/HumanBrainProject/neuroglancer-scripts.svg?branch=master
   :target: https://travis-ci.org/HumanBrainProject/neuroglancer-scripts
   :alt: Build Status
.. image:: https://codecov.io/gh/HumanBrainProject/neuroglancer-scripts/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/HumanBrainProject/neuroglancer-scripts
   :alt: Coverage Status
.. image:: https://readthedocs.org/projects/neuroglancer-scripts/badge/?version=latest
   :target: http://neuroglancer-scripts.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


neuroglancer-scripts
====================

Conversion of images to the Neuroglancer pre-computed format.

How to use
----------

See the user documentation at http://neuroglancer-scripts.readthedocs.io/


Development
-----------

This project uses ``pytest`` and ``tox`` for testing. You will need to install
the project (a virtual environment is recommended) before you can use it or run
the tests. ``pip install -e`` will install in editable mode, so you do not need
to re-install anything when changing existing files:

.. code-block:: shell

   python3 -m venv venv/
   . venv/bin/activate
   pip install -e .[dev]
   pytest --cov=neuroglancer_scripts --cov-report=html --cov-report=term


It is advisde to use ``tox`` for testing changes against all supported Python
versions. This is will run the same tests as the Travis continuous integration:

.. code-block:: shell

   tox
