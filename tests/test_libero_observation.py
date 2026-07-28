import numpy as np
import pytest

from smolvla_analysis.libero_observation import orient_archived_camera_for_policy


def test_orient_archived_camera_matches_libero_double_axis_flip():
    camera = np.arange(3 * 2 * 4, dtype=np.uint8).reshape(3, 2, 4)

    oriented = orient_archived_camera_for_policy(camera)

    np.testing.assert_array_equal(oriented, camera[:, ::-1, ::-1])
    assert oriented.flags.c_contiguous


def test_orient_archived_camera_rejects_non_chw_input():
    with pytest.raises(ValueError, match="CHW"):
        orient_archived_camera_for_policy(np.zeros((2, 4), dtype=np.uint8))
