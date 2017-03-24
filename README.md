Data conversion
===============

**Converting volumetric images to the Neuroglancer precomputed tile format.**
For practical examples, see the `BigBrainRelease.2015` or `Jubrain`
sub-directories.

1. Write the metadata for the full-resolution image (example:
   `BigBrainRelease.2015/info_fullres.json`).

2. Create metadata for all scales using `generate_scales_info.py`.

3. Convert your data to raw full-resolution tiles by using one of these
   scripts:
   - `slices_to_raw_chunks.py`
   - `volume_to_raw_chunks.py`

4. Compute downscaled pyramid levels using `compute_scales.py`.

5. If needed, convert chunks to compressed format using one of:
   - `convert_chunks_to_jpeg.py`
   - `compress_segmentation_chunks.py`


Serving the converted data
==========================

A Docker image running a specially-configured *nginx* web-server is available
for serving the converted data:
[neuroglancer-docker](https://github.com/HumanBrainProject/neuroglancer-docker)
(directly
[available on Docker Hub](https://hub.docker.com/r/ylep/neuroglancer/)).

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
    AddEncoding x-gzip .gz
    AddType application/octet-stream .gz
</IfModule>
```
