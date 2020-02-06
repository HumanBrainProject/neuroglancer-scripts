.. _command-line:

Command-line usage
==================

This page will teach you how to use the scripts in common cases.


Converting a single-file (NIfTI) volume
---------------------------------------

If your input dataset is stored in a single file that can be read by Nibabel_
(such as a NIfTI_ file), then you are in luck. The
``volume-to-precomputed-pyramid`` script will do all the work for you. It will
convert your dataset to raw pre-computed chunks.

Usage: ``volume-to-precomputed-pyramid [-h] [--ignore-scaling] [--mmap]
[--input-min INPUT_MIN] [--input-max INPUT_MAX] [--no-gzip] [--flat]
[--downscaling-method {auto,average,majority,stride}] [--outside-value
OUTSIDE_VALUE] volume_filename dest_url``.

You may want to use :ref:`convert-chunks <convert-chunks>` in a second step, to
further compres your dataset with JPEG or ``compressed_segmentation``
encoding).


Converting image volumes
------------------------

The instructions below are applicable to the two accepted input data layouts:

- Volumes in NIfTI format (or any other format readable by Nibabel). See
  :ref:`JuBrain` for an example.

- Series of 2D slices. See :ref:`BigBrain` for an example.


1. Write the metadata for the full-resolution image `in JSON format
   <https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md>`_.
   If your input data is readable by Nibabel, you can use
   ``volume-to-precomputed --generate-info`` do do the job. Here is an example
   with minimal metadata (note that the resolution is expressed in
   *nanometres*):

   .. code-block:: json

     {
       "type": "image",
       "data_type": "uint8",
       "num_channels": 1,
       "scales": [
         {
           "size": [151, 188, 154],
           "resolution": [1000000, 1000000, 1000000],
           "voxel_offset": [0, 0, 0]
         }
       ]
     }

2. Create metadata for all scales using ``generate-scales-info`` on the
   previously created JSON file. This step writes a file named ``info`` in the
   current directory, which is needed by Neuroglancer as well as by all the
   subsequent steps. You are advised to create a fresh directory for each
   dataset.

   You can use any lossless encoding for the following steps (i.e. ``raw`` or
   ``compressed_segmentation``).

   At this stage you may want to run ``scale-stats``, which displays the
   number of chunks that will be created, and their uncompressed size. Thus,
   you can make sure that you have enough disk space before proceeding.

3. Convert your data to raw full-resolution chunks by using one of these
   scripts:

   - ``slices-to-precomputed``
   - ``volume-to-precomputed``

4. Compute downscaled pyramid levels using ``compute-scales``. Make sure to
   use the correct downscaling method (``average`` for greyscale images,
   ``majority`` for segmentations, or ``stride`` for a fast low-quality
   downsampling).

   At this point the raw-format data is ready to be displayed in Neuroglancer.

.. _convert-chunks:

5. Optionally, you can convert the raw chunks to a compressed format using
   ``convert-chunks``. You will need to generate these compressed chunks in
   a separate directory from the raw chunks, and generate a suitable *info*
   file by using the ``--encoding`` parameter to ``generate-scales-info``.
   Two compressed encodings are available:

   - ``compressed_segmentation``: lossless compression, recommended for images
     containing discrete labels;
   - ``jpeg``: lossy JPEG compression, see ``--jpeg-quality`` and
     ``--jpeg-plane``.


Converting surface meshes
-------------------------

A surface mesh can be displayed in two ways in Neuroglancer: associated with a
segmentation label as part of a ``segmentation`` type layer, or as a standalone
mesh layer.

A mesh associated with a segmentation label needs to be in a
Neuroglancer-specific binary precomputed format. ``mesh-to-precomputed`` can be
used to convert meshes to this format. The ``link-mesh-fragments`` command must
then be used so that Neuroglancer knows what meshes are associated to each
label of the segmentation. See the last step of :ref:`BigBrain` for an example.


.. _Nibabel: https://nipy.org/nibabel/
.. _NIfTI: https://nifti.nimh.nih.gov/
