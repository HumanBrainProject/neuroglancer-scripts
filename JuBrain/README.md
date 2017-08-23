* Conversion of the grey-level template image (MNI Colin27 T1 MRI)

  ```
  mkdir colin
  cd colin
  ../../generate_scales_info.py ../info_colin_fullres.json
  ../../volume_to_raw_chunks.py ../colin.nii.gz
  ../../compute_scales.py
  ```

* Conversion of the Maximum Probability Map to raw chunks

  These raw-encoded tiles are an intermediate step to the compressed tiles
  below. Note that the info file specifies `encoding:
  "compressed_segmentation"`, if you want to serve these raw-encoded tiles to
  Neuroglancer you have to change the encoding to `"raw"`.

  We generate only one scale, because we do not (yet) have a tool to downscale
  a parcellation (`compute_scales.py` works for greyscale volumes, not for
  labelled volumes).

  ```
  mkdir atlas-raw
  cd atlas-raw
  ../../generate_scales_info.py --max-scales=1 ../info_atlas_fullres.json
  ../../volume_to_raw_chunks.py ../atlas.nii.gz
  ```

* Conversion to the compressed segmentation format
  ```
  mkdir atlas
  cd atlas
  ../../generate_scales_info.py --max-scales=1 ../info_atlas_fullres.json
  ../../compress_segmentation_chunks.py ../atlas-raw
  ```
