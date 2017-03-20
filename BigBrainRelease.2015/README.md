1. Download slices from ftp://bigbrain.loris.ca/BigBrainRelease.2015/2D_Final_Sections/Coronal/Png/Full_Resolution

2. Create `info_fullres.json`

3. Create raw chunks

   ```
   mkdir raw
   cd raw
   ../../generate_scales_info.py ../info_fullres.json
   ../../slices_to_raw_chunks.py <path/to/slices> RIA
   ../../compute_scales.py --outside-value=255
   ```

4. Convert raw chunks to JPEG: copy the `info` file from the `raw`
   sub-directory, and replace `encoding` by `jpeg`. Then, convert the chunks:

   ```
   mkdir jpeg
   cd jpeg
   ../../convert_chunks_to_jpeg.py --slicing-plane=xz ../raw/
   ```
