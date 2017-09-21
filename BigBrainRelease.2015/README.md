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

5. Convert the cortical meshes (downloaded from ftp://bigbrain.loris.ca/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/gii/). The coordinate transformation is needed because the X and Y axes are inverted in these Gifti files.

   ```Shell
   mkdir Apr7_2016
   cd Apr7_2016
   ../../mesh_to_vtk.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 gray_left_327680.gii grey-left.vtk
   ../../mesh_to_vtk.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 gray_right_327680.gii grey-right.vtk
   ../../mesh_to_vtk.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 white_left_327680.gii white-left.vtk
   ../../mesh_to_vtk.py --coord-transform=-1,0,0,0,0,-1,0,0,0,0,1,0 white_right_327680.gii white-right.vtk
   gzip -9 *.vtk
   ```
