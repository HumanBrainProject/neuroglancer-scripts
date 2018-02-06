Using the converted data in Neuroglancer
========================================

.. _info:

The Neuroglancer *info* file
----------------------------

See the `info JSON file specification
<https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/README.md#info-json-file-specification>`_
from Neuroglancer.


.. _half_voxel_shift:

Different conventions for coordinate transformations
----------------------------------------------------

Beware that Neuroglancer departs from the NIfTI convention in associating
physical coordinates to voxels: Neuroglancer associates physical coordinates to
the *corner* of a voxel, whereas NIfTI specifies that they refer to the
*centre* of a voxel. Therefore, **images will be offset by half a voxel
relative to meshes** if you do not compensate for this offset.

For standalone meshes, this offset can be compensated for by using the
``transform`` URL parameter. For pre-computed segmentation meshes however, there
is no way of specifying a different ``transform`` for the image and the
associated meshes: the offset must be applied to the vertex coordinates. This
can be achieved by using the ``--coord-transform`` option.

Please note that if you want to display images correctly with respect to
physical coordinates (e.g. stereotaxic coordinates), you have to use the
``transform`` parameter as well. The ``transform.json`` which is output by
``volume_to_raw_chunks.py`` *does* take the half-voxel shift into account.
