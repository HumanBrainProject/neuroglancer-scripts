Release notes
=============

0.3.0.dev0 (in development)
---------------------------

Bug fixes
~~~~~~~~~

- Fix the swarm of warning messages that appear when downscaling images of
  integer type by the averaging method.


Other improvements
~~~~~~~~~~~~~~~~~~

- The default downscaling method is now chosen automatically based on the
  ``type`` of dataset: ``image`` uses local averaging, ``segmentation`` uses
  striding.

- The command-line interface of ``mesh-to-precomputed`` was changed to work
  with Accessors, instead of being restricted to files. The command also now
  reads and writes the ``info`` file to make sure that Neuroglancer knows where
  to find the mesh fragments.


0.2.0 (16 October 2018)
-----------------------

Changes that affect converted data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use correct rounding when downscaling by local average (`commit 8f77b486 <https://github.com/HumanBrainProject/neuroglancer-scripts/commit/8f77b486122190dddf70aff2d321bd7664d3a0df>`_).


Bug fixes
~~~~~~~~~

- Fixed passing of options between functions (`issue #7 <https://github.com/HumanBrainProject/neuroglancer-scripts/issues/7>`_,
  `commit 67430f13 <https://github.com/HumanBrainProject/neuroglancer-scripts/commit/67430f1341352edeed6b63bc2177e052dd284993>`_,
  `commit f0e1e79d <https://github.com/HumanBrainProject/neuroglancer-scripts/commit/f0e1e79ddd1b3ef772b6920399f732e9cd487df3>`_).
  Thanks to Ben Falk for reporting and fixing the issue.

- Fixed the conversion of stacks of multi-channel images (`commit ff8d4dcc <https://github.com/HumanBrainProject/neuroglancer-scripts/commit/ff8d4dcc70ef25ba34798e2474bd37183aa289b7>`_).

- Fixed a crash when flippping mesh triangles for negative-determinant
  transformations (`commit 97545914 <https://github.com/HumanBrainProject/neuroglancer-scripts/commit/975459147174465b897d1bce8364e7bf434ce08c>`_,
  `issue #5 <https://github.com/HumanBrainProject/neuroglancer-scripts/issues/5>`_).

- Fixed a bug where the chunk size of extremely anisotropic volumes was set to
  zero (`commit 92264c91 <https://github.com/HumanBrainProject/neuroglancer-scripts/commit/92264c9189a8eec40a45622dbc30f785dd60a4d5>`_).

- Fixed loading of ``uint64`` TIFF files (`issue #8 <https://github.com/HumanBrainProject/neuroglancer-scripts/issues/8>`_).
  Thanks to Ben Falk for reporting and fixing the issue.


Other improvements
~~~~~~~~~~~~~~~~~~

- Introduced a new command ``volume-to-precomputed-pyramid`` for all-in-one
  conversion of volume in simple cases.

- Introduced a new command ``convert-chunks`` which can be used to convert
  existing data in Neuroglancer prec-computed format to a different encoding.

- ``slice-to-precomputed`` now uses proper rounding and clamping for performing
  data-type conversion.

- Improved the conversion of meshes to Neuroglancer-compatible VTK format, and
  got rid of the dependency on `PyVTK <https://github.com/pearu/pyvtk>`_.

- Improved the performance of common use-cases by loading the full volume in
  memory by default. The old behaviour can be restored with ``--mmap``.


0.1.0 (7 February 2018)
-----------------------

Initial PyPI release.
