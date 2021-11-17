.. neuroglancer-scripts documentation master file, created by
   sphinx-quickstart on Fri Feb  2 15:05:24 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation of neuroglancer-scripts
=====================================

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

The ``neuroglancer-scripts`` package provides :ref:`command-line tools
<command-line>`, and a :ref:`Python API <python-api>`, for converting
volumetric images and surface meshes to formats used by Neuroglancer_.


Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   script-usage
   serving-data
   neuroglancer-info
   examples
   api/neuroglancer_scripts
   release-notes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Neuroglancer: https://github.com/google/neuroglancer
