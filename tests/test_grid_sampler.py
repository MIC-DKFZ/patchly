import unittest
import numpy as np
import zarr
from samplify.sampler import _GridSampler
from samplify.slicer import slicer


class TestGridSampler(unittest.TestCase):
    def test_without_overlap_without_remainder_2d(self):
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_without_overlap_with_remainder_2d(self):
        patch_size = (10, 10)
        image_size = (103, 107)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_with_overlap_without_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_with_overlap_with_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        image_size = (103, 107)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_without_overlap_without_remainder_3d(self):
        patch_size = (10, 10, 5)
        image_size = (100, 100, 50)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_without_overlap_with_remainder_3d(self):
        patch_size = (10, 10, 5)
        image_size = (103, 107, 51)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_with_overlap_without_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        image_size = (100, 100, 50)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_with_overlap_with_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        image_size = (103, 107, 51)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_without_overlap_without_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        image_size = (4, 16, 8, 8, 4)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_without_overlap_with_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        image_size = (5, 18, 9, 10, 6)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_with_overlap_without_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        image_size = (4, 16, 8, 8, 4)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_with_overlap_with_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        image_size = (5, 18, 9, 10, 6)
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_channel_first(self):
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random((3, *image_size))
        channel_first = True

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_channel_last(self):
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random((*image_size, 5))
        channel_first = False

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_batch_and_channel_dim(self):
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random((4, 3, *image_size))
        channel_first = True

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_multiple_non_spatial_dims(self):
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random((5, 4, 3, *image_size))
        channel_first = True

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_zarr(self):
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random(image_size)
        image = zarr.array(image)

        # Test with image
        result = np.zeros_like(image)
        sampler = _GridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = _GridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_without_patch_size(self):
        image_size = (100, 100)
        image = np.random.random(image_size)
        image = zarr.array(image)

        self.assertRaises(RuntimeError, _GridSampler, image=image, image_size=image_size)

    def test_patch_size_larger_than_image_size(self):
        patch_size = (101, 100)
        image_size = (100, 100)
        image = np.random.random(image_size)
        image = zarr.array(image)

        self.assertRaises(RuntimeError, _GridSampler, image=image, image_size=image_size, patch_size=patch_size)

    def test_overlap_size_larger_than_patch_size(self):
        patch_overlap = (11, 10)
        patch_size = (10, 10)
        image_size = (100, 100)
        image = np.random.random(image_size)

        self.assertRaises(RuntimeError, _GridSampler, image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

    def get_slices(self, image, image_size, patch_indices, channel_first):
        non_image_dims = len(image.shape) - len(image_size)
        if channel_first:
            slices = [None] * non_image_dims
            slices.extend([index_pair.tolist() for index_pair in patch_indices])
        else:
            slices = [index_pair.tolist() for index_pair in patch_indices]
            slices.extend([None] * non_image_dims)
        return slices


if __name__ == '__main__':
    unittest.main()