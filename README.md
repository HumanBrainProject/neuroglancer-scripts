Convert volumetric images to the Neuroglancer precomputed tile format

1. Write the metadata for the full-resolution image (example:
   `BigBrainRelease.2015/info_fullres.json`).

2. Create metadata for all scales using `generate_scales_info.py`.

3. Convert your data to raw full-resolution tiles by adapting one of these
   scripts to your needs:
   - `slices_to_raw_chunks.py`

4. Compute downscaled pyramid levels using `compute_scales.py`.

5. If needed, convert chunks to compressed format using one of:
   - `convert_chunks_to_jpeg.py`
