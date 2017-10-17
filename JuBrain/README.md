The data in this directory is based on the JuBrain human brain atlas, as
published in version 2.2c of the [SPM Anatomy Toolbox](http://www.fz-juelich.de/inm/inm-1/EN/Forschung/_docs/SPMAnatomyToolbox/SPMAnatomyToolbox_node.html).
Note that you need to use [git-lfs](https://git-lfs.github.com/) in order to
see the contents of the NIfTI files (otherwise you can download them [from the
repository on Github](https://github.com/HumanBrainProject/neuroglancer-scripts/tree/master/JuBrain)).

Conversion of the grey-level template image (MNI Colin27 T1 MRI)
================================================================

  ```Shell
  mkdir colin27T1_seg
  cd colin27T1_seg
  ../../volume_to_raw_chunks.py --generate-info ../colin27T1_seg.nii.gz
  ```

  At this point, you need to edit `info_fullres.json` to set `"data_type":
  "uint8"`. This is needed because `colin27T1_seg.nii.gz` uses a peculiar
  encoding, with slope and intercept set in the NIfTI header, even though only
  integers between 0 and 255 are encoded.

  ```Shell
  ../../generate_scales_info.py info_fullres.json
  ../../volume_to_raw_chunks.py ../colin27T1_seg.nii.gz
  ../../compute_scales.py
  ```

Conversion of the Maximum Probability Map
=========================================

1. Conversion to raw chunks

   These raw-encoded chunks are an intermediate step to the compressed chunks
   below. We generate only one scale, because we do not (yet) have a tool to
   downscale a parcellation (`compute_scales.py` works for greyscale volumes,
   not for labelled volumes).

   ```Shell
   mkdir MPM-raw
   cd MPM-raw
   ../../volume_to_raw_chunks.py --generate-info ../MPM.nii.gz
   ../../generate_scales_info.py --type=segmentation --max-scales=1 info_fullres.json
   ../../volume_to_raw_chunks.py ../MPM.nii.gz
   ```

2. Conversion to the compressed segmentation format

   ```Shell
   mkdir MPM
   cd MPM
   ../../generate_scales_info.py --type=segmentation --encoding=compressed_segmentation --max-scales=1 ../MPM-raw/info_fullres.json
   ../../compress_segmentation_chunks.py ../MPM-raw/
   ```
