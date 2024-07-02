import json
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
from neuroglancer_scripts.volume_reader import (
    nibabel_image_to_info,
    volume_file_to_precomputed,
)


def prepare_nifti_images():

    random_rgb_val = np.random.rand(81).reshape((3, 3, 3, 3)) * 255
    random_rgb_val = random_rgb_val.astype(np.uint8)
    right_type = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    new_data = random_rgb_val.copy().view(dtype=right_type).reshape((3, 3, 3))
    rgb_img = nib.Nifti1Image(new_data, np.eye(4))

    random_uint8_val = np.random.rand(27).reshape((3, 3, 3)) * 255
    random_uint8_val = random_uint8_val.astype(np.uint8)
    uint8_img = nib.Nifti1Image(random_uint8_val, np.eye(4))

    return [(rgb_img, 3), (uint8_img, 1)]


@pytest.mark.parametrize("nifti_img,expected_num_channel",
                         prepare_nifti_images())
def test_nibabel_image_to_info(nifti_img, expected_num_channel):

    formatted_info, _, _, _ = nibabel_image_to_info(nifti_img)
    info = json.loads(formatted_info)
    assert info.get("num_channels") == expected_num_channel


@pytest.mark.parametrize("nifti_img,expected_num_channel",
                         prepare_nifti_images())
@patch('neuroglancer_scripts.precomputed_io.get_IO_for_existing_dataset',
       return_value=None)
@patch('neuroglancer_scripts.volume_reader.nibabel_image_to_precomputed')
@patch("nibabel.load")
def test_volume_file_to_precomputed(m_nib_load, m_nib_img_precomp, _,
                                    nifti_img, expected_num_channel):
    m_nib_load.return_value = nifti_img
    m_nib_img_precomp.return_value = "hoho"
    volume_file_to_precomputed("mock_file_name", "./bla")

    assert m_nib_load.called
    assert m_nib_img_precomp.called

    nibabel_image = m_nib_img_precomp.call_args[0][0]

    if expected_num_channel == 1:
        assert nibabel_image is nifti_img
    else:
        assert nibabel_image is not nifti_img
        assert len(nibabel_image.dataobj.shape) == 4
