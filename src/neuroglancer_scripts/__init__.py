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
# 3.  Bump the version number in this file;
# 4.  pip install -U build twine
# 5.  python3 -m build
# 6.  twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# 7.  Commit the updated version number
# 8.  Tag the commit (git tag -a vX.Y.Z). The release notes for the last
#     version should be converted to plain text and included in the tag
#     message:
#     pandoc -t plain docs/release-notes.rst
# 9.  Bump the version number in this file to something that ends with .dev0
#     and commit
# 10. Push the master branch and the new tag to Github
# 11. twine upload dist/*
__version__ = "0.4.0.dev0"
