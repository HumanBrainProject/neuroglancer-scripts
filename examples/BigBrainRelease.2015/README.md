1. Download slices from ftp://bigbrain.loris.ca/BigBrainRelease.2015/2D_Final_Sections/Coronal/Png/Full_Resolution/

2. Create `info_fullres.json`

3. Create raw chunks

   ```Shell
   mkdir raw
   cd raw
   ../../generate_scales_info.py ../info_fullres.json
   ../../slices_to_raw_chunks.py <path/to/slices> RIA
   ../../compute_scales.py --outside-value=255
   ```

4. Convert raw chunks to JPEG: copy the `info` file from the `raw`
   sub-directory, and replace `encoding` by `jpeg`. Then, convert the chunks:

   ```Shell
   mkdir jpeg
   cd jpeg
   ../../convert_chunks_to_jpeg.py --slicing-plane=xz ../raw/
   ```

5. Convert the cortical meshes (downloaded from ftp://bigbrain.loris.ca/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/gii/ and segmentation volume (`classif.nii.gz` in this directory).

   ```Shell
   mkdir classif
   cd classif
   ../../../neuroglancer-scripts/volume_to_raw_chunks.py --generate-info ../classif.nii.gz
   ```
   Edit `info_fullres.json` which was just created by the previous command:
       - fix the voxel size to exactly 200000 nm isotropic (the extra digits are because of the limited precision of the Nifti header);
       - Add a top-level key `"mesh": "mesh"`.
   ```Shell
   ../../../generate_scales_info.py --encoding=compressed_segmentation info_fullres.json

   mkdir raw
   cd raw
   ../../../../generate_scales_info.py --encoding=raw ../info_fullres.json
   ../../../../volume_to_raw_chunks.py --load-full-volume ../../classif.nii.gz
   ../../../../compute_scales.py --downscaling-method=mode

   cd ..
   ../../../compress_segmentation_chunks.py raw
   rm -r raw
   ```

    Finally, convert the Gifti meshes to mesh fragments in pre-computed format, and create the JSON files that Neuroglancer needs in order to find the mesh fragments. The coordinate transformation is needed because the X and Y axes are inverted in these Gifti files.
   ```Shell
   mkdir mesh
   ../../../mesh_to_precomputed.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 gray_left_327680.gii mesh/grey-left
   ../../../mesh_to_precomputed.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 gray_right_327680.gii mesh/grey-right
   ../../../mesh_to_precomputed.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 white_left_327680.gii mesh/white-left
   ../../../mesh_to_precomputed.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 white_right_327680.gii mesh/white-right
   echo '{"fragments":[]}' > mesh/0
   echo '{"fragments":["grey-left","grey-right"]}' > mesh/100
   echo '{"fragments":["white-left","white-right"]}' > mesh/200
   ```
