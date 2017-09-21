Dependencies
============

These scripts should work with Python 3.4 or later. In order to install the
required dependencies in a virtual environment, run:
```
python3 -m venv conversion-venv
. conversion-venv/bin/activate
pip install -r requirements.txt
```


Converting data
===============

Two main types of data can be converted with these scripts: image volumes, and
surface meshes.

Converting image volumes
------------------------

Here is a summary of the steps needed for converting volumetric images to the
Neuroglancer precomputed chunk format.

Two types of data layout are accepted as input:
- Volumes in NIfTI format (or any other format readable by Nibabel). See
  `JuBrain` for an example.
- Series of 2D slices. See `BigBrainRelease.2015` for an example.

1. Write the metadata for the full-resolution image [in JSON
   format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/README.md).
   If your input data is readable by Nibabel, you can use
   `volume_to_raw_chunks.py --generate-info` do do the job. Here is an example
   with minimal metadata (note that the resolution is expressed in
   *nanometres*):

   ```JSON
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
   ```

2. Create metadata for all scales using `generate_scales_info.py` on the
   previously created JSON file. This step writes a file named `info` in the
   current directory, which is needed by Neuroglancer as well as by all the
   subsequent steps. You are advised to create a fresh directory for each
   dataset.

   If your image contains discrete labels, you will need to pass
   `--max-scales=1`, because scale pyramids are not (yet) supported for
   labelled data.

   At this stage you may want to run `scale_stats.py`, which displays the
   number of chunks that will be created, and their uncompressed size. Thus,
   you can make sure that you have enough disk space before proceeding.

3. Convert your data to raw full-resolution chunks by using one of these
   scripts:
   - `slices_to_raw_chunks.py`
   - `volume_to_raw_chunks.py`

4. Compute downscaled pyramid levels using `compute_scales.py`. The scales are
   computed by local averaging, so this step is not suitable for images
   containing discrete labels.

   At this point the raw-format data is ready to be displayed in Neuroglancer.

5. Finally, you can convert the raw chunks to a compressed format using one of:
   - `convert_chunks_to_jpeg.py` (lossy JPEG compression, see `--jpeg-quality`)
   - `compress_segmentation_chunks.py` (lossless compression, recommended for
      discrete labels).

   You will need to generate these compressed chunks in a separate directory
   from the raw chunks, and generate a suitable `info` file by using the
   `--encoding` parameter to `generate_scales_info.py`.


Converting surface meshes
-------------------------

A surface mesh can be displayed in two ways in Neuroglancer: associated with a
segmentation label as part of a `segmentation` type layer, or as a standalone
mesh layer.

A standalone mesh layer needs a mesh in the VTK ASCII format. `mesh_to_vtk.py`
can be used to convert a mesh from GIfTI to that format.

A mesh associated with a segmentation label needs to be in a
Neuroglancer-specific binary precomputed format. `mesh_to_precomputed.py` or
`stl_to_precomputed.py` can be used to convert meshes to this format.
Additionally, you need to add a `mesh` key to the `info` file of the
segmentation volume, and provide one JSON file per segment, as described in
[the Neuroglancer documentation of the precomputed
format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/README.md).
At the moment this must be done manually.


Serving the converted data
==========================

nginx
-----

A Docker image running a specially-configured *nginx* web-server is available
for serving the converted data:
[neuroglancer-docker](https://github.com/HumanBrainProject/neuroglancer-docker)
(directly
[available on Docker Hub](https://hub.docker.com/r/ylep/neuroglancer/)).


Apache
------

Alternatively, you serve the pre-computed images using Apache, with the
following Apache configuration (e.g. put it in a ``.htaccess`` file):

```ApacheConf
<IfModule headers_module>
    # Needed to use the data from a Neuroglancer instance served from a
    # different server (see http://enable-cors.org/server_apache.html).
    Header set Access-Control-Allow-Origin "*"
</IfModule>

# Data chunks are stored in sub-directories, in order to avoid having
# directories with millions of entries. Therefore we need to rewrite URLs
# because Neuroglancer expects a flat layout.
Options FollowSymLinks
RewriteEngine On
RewriteRule "^(.*)/([0-9]+-[0-9]+)_([0-9]+-[0-9]+)_([0-9]+-[0-9]+)$" "$1/$2/$3/$4"

<IfModule mime_module>
    # Allow serving pre-compressed files, which can save a lot of space for raw
    # chunks, compressed segmentation chunks, and mesh chunks.
    #
    # The AddType directive should in theory be replaced by a "RemoveType .gz"
    # directive, but with that configuration Apache fails to serve the
    # pre-compressed chunks (confirmed with Debian version 2.2.22-13+deb7u6).
    # Fixes welcome.
    Options Multiviews
    AddEncoding x-gzip .gz
    AddType application/octet-stream .gz
</IfModule>
```
