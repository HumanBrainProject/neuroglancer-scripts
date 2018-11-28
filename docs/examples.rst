.. _Examples:

Examples
========

.. _JuBrain:

Conversion of JuBrain
---------------------

In the ``examples/JuBrain`` directory of the source distribution, you will find
two Nifti files based on the JuBrain human brain atlas, as published in version
2.2c of the `SPM Anatomy Toolbox
<http://www.fz-juelich.de/inm/inm-1/EN/Forschung/_docs/SPMAnatomyToolbox/SPMAnatomyToolbox_node.html>`_.
Note that you need to use `git-lfs <https://git-lfs.github.com/>`_ in order to
see the contents of the NIfTI files (otherwise you can download them `from the
repository on Github
<https://github.com/HumanBrainProject/neuroglancer-scripts/tree/master/JuBrain>`_.

Conversion of the grey-level template image (MNI Colin27 T1 MRI)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: sh

  volume-to-precomputed \
      --generate-info \
      colin27T1_seg.nii.gz \
      colin27T1_seg/

At this point, you need to edit ``colin27T1_seg/info_fullres.json`` to set
``"data_type": "uint8"``. This is needed because ``colin27T1_seg.nii.gz`` uses
a peculiar encoding, with slope and intercept set in the NIfTI header, even
though only integers between 0 and 255 are encoded.

.. code-block:: sh

  generate-scales-info colin27T1_seg/info_fullres.json colin27T1_seg/
  volume-to-precomputed colin27T1_seg.nii.gz colin27T1_seg/
  compute-scales colin27T1_seg/


Conversion of the Maximum Probability Map
+++++++++++++++++++++++++++++++++++++++++

.. code-block:: sh

   volume-to-precomputed --generate-info MPM.nii.gz MPM/
   generate-scales-info \
       --type=segmentation \
       --encoding=compressed_segmentation \
       MPM/info_fullres.json \
       MPM/
   volume-to-precomputed MPM.nii.gz MPM/
   compute-scales --downscaling-method=majority MPM/


.. _BigBrain:

Conversion of BigBrain
----------------------

BigBrain is a very large image (6572 × 7404 × 5711 voxels) reconstructed from
7404 serial coronal section of a human brain, with a resolution of about
20 microns.

1. Download slices from ftp://bigbrain.loris.ca/BigBrainRelease.2015/2D_Final_Sections/Coronal/Png/Full_Resolution/

2. Create ``info_fullres.json`` with the appropriate metadata:

.. code-block:: json

   {
     "type": "image",
     "data_type": "uint8",
     "num_channels": 1,
     "scales": [
       {
         "chunk_sizes": [],
         "encoding": "raw",
         "key": "full",
         "resolution": [21166.6666666666666, 20000, 21166.6666666666666],
         "size": [6572, 7404, 5711],
         "voxel_offset": [0, 0, 0]
       }
     ]
   }

3. Create raw chunks

.. code-block:: sh

   generate-scales-info info_fullres.json 8bit/
   slices-to-precomputed --input-orientation RIA <path/to/slices> 8bit/
   compute-scales --outside-value=255 8bit/

4. Optionally, convert raw chunks to JPEG:

.. code-block:: sh

   generate-scales-info --encoding=jpeg 8bit/info jpeg/
   convert-chunks --jpeg-plane=xz 8bit/ jpeg/

5. Convert the segmentation volume
   (``examples/BigBrainRelease.2015/classif.nii.gz`` in the source
   distribution, this is a voxelized version of the meshes below).

.. code-block:: sh

   volume-to-precomputed --generate-info classif.nii.gz classif/
   generate-scales-info \
       --encoding=compressed_segmentation \
       classif/info_fullres.json \
       classif/
   volume-to-precomputed --load-full-volume classif.nii.gz classif/
   compute-scales --downscaling-method=majority classif/

6. Add the cortical meshes to the segmentation (downloaded from
   ftp://bigbrain.loris.ca/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/gii/).
   The meshes are displayed in the 3D view.

   Edit ``classif/info`` to add a top-level ``mesh`` key pointing to a ``mesh``
   sub-directory: ``"mesh": "mesh"``.

   Finally, convert the Gifti meshes to mesh fragments in pre-computed format,
   and create the JSON files that Neuroglancer needs in order to find the mesh
   fragments. The coordinate transformation is needed for two reasons:

   - the translation is the inverted transform of the classification volume (as
     output by ``volume-to-precomputed``, it is needed to bring the mesh into
     alignment with the volume;

   - the -1 coefficients on the diagonal are needed because the X and Y axes
     are inverted in these Gifti files.

.. code-block:: sh

   mesh-to-precomputed \
       --coord-transform=-1,0,0,70.7666,0,-1,0,73.01,0,0,1,58.8777 \
       --mesh-name=grey-left \
       gray_left_327680.gii \
       classif/
   mesh-to-precomputed \
       --coord-transform=-1,0,0,70.7666,0,-1,0,73.01,0,0,1,58.8777 \
       --mesh-name=grey-right \
       gray_right_327680.gii \
       classif/
   mesh-to-precomputed \
       --coord-transform=-1,0,0,70.7666,0,-1,0,73.01,0,0,1,58.8777 \
       --mesh-name=white-left \
       white_left_327680.gii \
       classif/
   mesh-to-precomputed \
       --coord-transform=-1,0,0,70.7666,0,-1,0,73.01,0,0,1,58.8777 \
       --mesh-name=white-right \
       white_right_327680.gii \
       classif/
   link-mesh-fragments --no-colon-suffix mesh_labels.csv classif/
