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

Documentation
-------------

http://neuroglancer-scripts.readthedocs.io/

Development
-----------

This project uses ``pytest`` and ``tox`` for testing. Useful commands:

.. code-block:: shell

   pytest
   pytest <pytest options>
   pytest --cov=neuroglancer_scripts
   tox
   tox -e docs
