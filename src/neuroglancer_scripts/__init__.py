# Copyright (c) 2018 Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

"""Conversion of images to the Neuroglancer pre-computed format.

.. todo:: introduction to the high-level APIs
"""

# Version used by setup.py and docs/conf.py (parsed with a regular expression).
#
# Release checklist (based on https://packaging.python.org/):
# 1.  Ensure that tests pass for all supported Python version (Travis CI),
#     ensure that the API documentation is complete (sphinx-apidoc -o docs/api/
#     src/neuroglancer_scripts);
# 2.  Update the release notes;
# 3.  Run check-manifest;
# 4.  Bump the version number in this file;
# 5.  pip install -U setuptools wheel twine
# 6.  python setup.py sdist bdist_wheel
# 7.  twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# 8.  Commit the updated version number
# 9.  Tag the commit (git tag -a vX.Y.Z)
# 10. Bump the version number to something that ends with .dev0 and commit
# 11. Push the master branch and the new tag to Github
# 12. twine upload dist/*
__version__ = "0.2.0"
