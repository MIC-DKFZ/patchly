import unittest
import numpy as np
import zarr
from samplify.sampler import GridSampler
from samplify.slicer import slicer
from samplify import utils


class TestBasicGridSampler(unittest.TestCase):
    def test_without_overlap_without_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        result_size = spatial_size
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size)

    def test_without_overlap_with_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (103, 107)
        result_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size)

    def test_with_overlap_without_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        result_size = spatial_size
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        result_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_2d_v2(self):
        patch_overlap = (3, 3)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        result_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_without_overlap_without_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        result_size = spatial_size
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size)

    def test_without_overlap_with_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        result_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size)

    def test_with_overlap_without_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        result_size = spatial_size
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        result_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_without_overlap_without_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        result_size = spatial_size
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size)

    def test_without_overlap_with_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        result_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size)

    def test_with_overlap_without_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        result_size = spatial_size
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        result_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_sampler(image, result_size, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_channel_first(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        result_size = spatial_size
        image = np.random.random((3, *spatial_size))        
        spatial_first = False

        self._test_sampler(image, result_size, spatial_size, patch_size, spatial_first=spatial_first)

    def test_channel_last(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        result_size = spatial_size
        image = np.random.random((*spatial_size, 5))
        spatial_first = True

        self._test_sampler(image, result_size, spatial_size, patch_size, spatial_first=spatial_first)

    def test_batch_and_channel_dim(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        result_size = spatial_size
        image = np.random.random((4, 3, *spatial_size))
        spatial_first = False

        self._test_sampler(image, result_size, spatial_size, patch_size, spatial_first=spatial_first)

    def test_multiple_non_spatial_dims(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        result_size = spatial_size
        image = np.random.random((5, 4, 3, *spatial_size))
        spatial_first = False

        self._test_sampler(image, result_size, spatial_size, patch_size, spatial_first=spatial_first)

    def test_zarr(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        result_size = spatial_size
        image = np.random.random(spatial_size)
        image = zarr.array(image)

        self._test_sampler(image, result_size, spatial_size, patch_size)

    def test_patch_size_larger_than_spatial_size(self):
        patch_size = (101, 100)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, mode="sample_crop")

    def test_overlap_size_larger_than_patch_size(self):
        patch_overlap = (11, 10)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, mode="sample_crop")

    def test_spatial_size_unequal_to_spatial_image_size(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((200, 200))

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, mode="sample_crop")

    def _test_sampler(self, image, result_size, spatial_size, patch_size, patch_overlap=None, spatial_first=True):
        # Test with image
        result = np.zeros(result_size)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, spatial_first=spatial_first, mode="sample_crop")

        for patch, patch_bbox in sampler:
            if not spatial_first:
                _patch_size = patch.shape[-len(patch_size):]
            else:
                _patch_size = patch.shape[:len(patch_size)]
            self.assertEqual(_patch_size, patch_size, "patch.shape: {}, patch_size: {}, patch bbox: {}".format(patch.shape, patch_size, patch_bbox))
            result[slicer(result, patch_bbox)] = 1
            patch_bbox = utils.add_non_spatial_bbox_dims(patch_bbox, image, spatial_first)
            np.testing.assert_array_equal(patch, image[slicer(image, patch_bbox)], err_msg="image shape: {}, patch shape: {}, patch bbox: {}".format(image.shape, patch.shape, patch_bbox))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros(result_size)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, spatial_first=spatial_first, mode="sample_crop")

        for patch_bbox in sampler:
            result[slicer(result, patch_bbox)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))


if __name__ == '__main__':
    unittest.main()