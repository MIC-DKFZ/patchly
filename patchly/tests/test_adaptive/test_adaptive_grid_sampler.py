import unittest
import numpy as np
import zarr
from patchly import GridSampler, slicer, utils, SamplingMode
import torch


class TestAdaptiveGridSampler(unittest.TestCase):
    def test_without_offset_without_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size)

    def test_without_offset_with_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size)

    def test_with_offset_without_remainder_2d(self):
        patch_offset = (5, 5)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_2d(self):
        patch_offset = (5, 5)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_2d_v2(self):
        patch_offset = (3, 3)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_without_offset_without_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size)

    def test_without_offset_with_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size)

    def test_with_offset_without_remainder_3d(self):
        patch_offset = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_3d(self):
        patch_offset = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_without_offset_without_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size)

    def test_without_offset_with_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size)

    def test_with_offset_without_remainder_Nd(self):
        patch_offset = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_Nd(self):
        patch_offset = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        image = np.random.random(spatial_size)

        self._test_sampler(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_channel_first(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((3, *spatial_size))
        spatial_first = False

        self._test_sampler(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_channel_last(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((*spatial_size, 5))
        spatial_first = True

        self._test_sampler(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_batch_and_channel_dim(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((4, 3, *spatial_size))
        spatial_first = False

        self._test_sampler(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_multiple_non_spatial_dims(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((5, 4, 3, *spatial_size))
        spatial_first = False

        self._test_sampler(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_zarr(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)
        image = zarr.array(image)

        self._test_sampler(image, spatial_size, patch_size)

    def test_tensor(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)
        image = torch.tensor(image)

        self._test_sampler(image, spatial_size, patch_size)

    def test_patch_size_larger_than_spatial_size(self):
        patch_size = (101, 100)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)
        image = zarr.array(image)

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, mode=SamplingMode.SAMPLE_ADAPTIVE)

    def test_offset_size_larger_than_patch_size(self):
        patch_offset = (11, 10)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, patch_offset=patch_offset, mode=SamplingMode.SAMPLE_ADAPTIVE)

    def test_spatial_size_unequal_to_spatial_image_size(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((200, 200))

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, mode=SamplingMode.SAMPLE_ADAPTIVE)

    def _test_sampler(self, image, spatial_size, patch_size, patch_offset=None, spatial_first=True):
        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_offset=patch_offset, spatial_first=spatial_first, mode=SamplingMode.SAMPLE_ADAPTIVE)

        for patch, patch_bbox in sampler:
            result[slicer(result, patch_bbox)] = 1
            patch_bbox = utils.bbox_s_to_bbox_h(patch_bbox, image, spatial_first)
            np.testing.assert_array_equal(patch, image[slicer(image, patch_bbox)], err_msg="image shape: {}, patch shape: {}, patch bbox: {}".format(image.shape, patch.shape, patch_bbox))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_offset: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_offset=patch_offset, spatial_first=spatial_first, mode=SamplingMode.SAMPLE_ADAPTIVE)

        for patch_bbox in sampler:
            result[slicer(result, patch_bbox)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_offset: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))


if __name__ == '__main__':
    unittest.main()