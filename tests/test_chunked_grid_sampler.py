import unittest
import numpy as np
import zarr
from samplify.sampler import GridSampler
from samplify.slicer import slicer


class TestChunkedGridSampler(unittest.TestCase):
    def test_without_overlap_without_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_without_overlap_with_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (103, 107)
        chunk_size = (40, 40)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_with_overlap_without_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_with_overlap_with_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        chunk_size = (40, 40)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_without_overlap_without_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        chunk_size = (40, 40, 20)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_without_overlap_with_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        chunk_size = (40, 40, 20)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_with_overlap_without_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        chunk_size = (40, 40, 20)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_with_overlap_with_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        chunk_size = (40, 40, 20)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_without_overlap_without_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 8)
        chunk_size = (4, 16, 8, 8, 8)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_without_overlap_with_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 9)
        chunk_size = (4, 16, 8, 8, 8)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_with_overlap_without_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 8)
        chunk_size = (4, 16, 8, 8, 8)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_with_overlap_with_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 9)
        chunk_size = (4, 16, 8, 8, 8)
        image = np.random.random(spatial_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, patch_size, "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_overlap
        ))

    def test_channel_first(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random((3, *spatial_size))
        spatial_first = False

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, spatial_first=spatial_first, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, (3, *patch_size), "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_channel_last(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random((*spatial_size, 5))
        spatial_first = True

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, spatial_first=spatial_first, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, (*patch_size, 5), "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_batch_and_channel_dim(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random((4, 3, *spatial_size))
        spatial_first = False

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, spatial_first=spatial_first, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, (4, 3, *patch_size), "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_multiple_non_spatial_dims(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random((5, 4, 3, *spatial_size))
        spatial_first = False

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, spatial_first=spatial_first, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            self.assertEqual(patch.shape, (5, 4, 3, *patch_size), "patch.shape: {}, patch_size: {}, patch indices: {}".format(patch.shape, patch_size, patch_indices))
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, slices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}, slices: {}".format(image.shape, patch.shape, patch_indices, slices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            slices = self.add_non_spatial_indices(result, spatial_size, patch_indices, spatial_first)
            result[slicer(result, slices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_zarr(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random(spatial_size)
        image = zarr.array(image)

        # Test with image
        result = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch, patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1
            np.testing.assert_array_equal(patch, image[slicer(image, patch_indices)], err_msg="image shape: {}, patch shape: {}, patch indices: {}".format(image.shape, patch.shape, patch_indices))

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

        # Test without image
        result = np.zeros_like(image)
        sampler = GridSampler(spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

        for patch_indices, chunk_id in sampler:
            result[slicer(result, patch_indices)] = 1

        self.assertEqual(np.sum(result), result.size, "result sum: {}, result size: {}, result shape: {}, image shape: {}, patch shape: {}, patch_overlap: {}".format(
            np.sum(result), result.size, result.shape, image.shape, patch_size, patch_size
        ))

    def test_patch_size_larger_than_spatial_size(self):
        patch_size = (101, 100)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random(spatial_size)
        image = zarr.array(image)

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, chunk_size=chunk_size, mode="sample_edge")

    def test_overlap_size_larger_than_patch_size(self):
        patch_overlap = (11, 10)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (40, 40)
        image = np.random.random(spatial_size)

        self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, mode="sample_edge")

    def add_non_spatial_indices(self, image, spatial_size, patch_indices, spatial_first):
        non_image_dims = len(image.shape) - len(spatial_size)
        if spatial_first:
            slices = [index_pair.tolist() for index_pair in patch_indices]
            slices.extend([[None]] * non_image_dims)
        else:
            slices = [[None]] * non_image_dims
            slices.extend([index_pair.tolist() for index_pair in patch_indices])
        return slices


if __name__ == '__main__':
    unittest.main()