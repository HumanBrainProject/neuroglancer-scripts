* Conversion of the grey-level template image (MNI Colin27 T1 MRI)

  ```
  mkdir colin
  cd colin
  ../../generate_scales_info.py ../info_colin_fullres.json
  ../../volume_to_raw_chunks.py --ignore-scaling ../colin.nii.gz
  ../../compute_scales.py
  ```

* Conversion of the Maximum Probability Map to raw chunks

  ```
  mkdir atlas-raw
  cd atlas-raw
  ../../generate_scales_info.py ../info_atlas_fullres.json
  ../../volume_to_raw_chunks.py ../atlas.nii.gz
  ../../compute_scales.py
  ```

* Conversion to the compressed segmentation format
  ```
  mkdir atlas
  cd atlas
  ../../generate_scales_info.py ../info_atlas_fullres.json
  ../../compress_segmentation_chunks.py ../atlas-raw
  ```
