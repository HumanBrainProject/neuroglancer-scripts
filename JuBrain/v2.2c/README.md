* Conversion of the grey-level template image (MNI Colin27 T1 MRI)

  ```
  mkdir colin27T1_seg
  cd colin27T1_seg
  ../../../volume_to_raw_chunks.py --generate-info ../colin27T1_seg.nii.gz
  # Edit info_fullres.json to set "data_type": "uint8"
  ../../../generate_scales_info.py info_fullres.json
  ../../../volume_to_raw_chunks.py ../colin27T1_seg.nii.gz
  ../../../compute_scales.py
  ```

* Conversion of the Maximum Probability Map

** Conversion to raw chunks

   These raw-encoded tiles are an intermediate step to the compressed tiles
   below. Note that the info file specifies `encoding:
   "compressed_segmentation"`, if you want to serve these raw-encoded tiles to
   Neuroglancer you have to change the encoding to `"raw"`.

   We generate only one scale, because we do not (yet) have a tool to downscale
   a parcellation (`compute_scales.py` works for greyscale volumes, not for
   labelled volumes).

   ```
   mkdir MPM
   cd MPM
   ../../../volume_to_raw_chunks.py --generate-info ../MPM.nii.gz
   mkdir raw
   cd raw
   ../../../../generate_scales_info.py --max-scales=1 ../info_fullres.json
   ../../../../volume_to_raw_chunks.py ../../MPM.nii.gz
   ```

** Conversion to the compressed segmentation format

   ```
   cd MPM
   ../../../generate_scales_info.py  --encoding=compressed_segmentation --max-scales=1 info_fullres.json
   ../../../compress_segmentation_chunks.py raw/
   ```
