import unittest
import random
import numpy as np
from samplify.sampler import BasicGridSampler
from samplify.slicer import slicer


class TestBasicGridSampler(unittest.TestCase):
    def test_without_overlap_without_remainder_2d(self):
        self._test_without_overlap_without_remainder(spatial_dims=2, patch_size_range=(50, 500), image_size_multiple=(1, 50))

    def test_without_overlap_with_remainder_2d(self):
        self._test_without_overlap_with_remainder(spatial_dims=2, patch_size_range=(50, 500), image_size_max=5000)

    def test_with_overlap_without_remainder_2d(self):
        self._test_with_overlap_without_remainder(spatial_dims=2, patch_overlap_size_range=(50, 500), patch_size_range=(1, 10), image_size_multiple=(1, 50))

    def test_with_overlap_with_remainder_2d(self):
        self._test_with_overlap_with_remainder(spatial_dims=2, patch_overlap_size_range=(50, 500), patch_size_range=(1, 10), image_size_max=5000)

    def test_without_overlap_without_remainder_3d(self):
        self._test_without_overlap_without_remainder(spatial_dims=3, patch_size_range=(50, 500), image_size_multiple=(1, 50))

    def test_without_overlap_with_remainder_3d(self):
        self._test_without_overlap_with_remainder(spatial_dims=3, patch_size_range=(50, 500), image_size_max=5000)

    def test_with_overlap_without_remainder_3d(self):
        self._test_with_overlap_without_remainder(spatial_dims=3, patch_overlap_size_range=(50, 500), patch_size_range=(1, 10), image_size_multiple=(1, 50))

    def test_with_overlap_with_remainder_3d(self):
        self._test_with_overlap_with_remainder(spatial_dims=3, patch_overlap_size_range=(50, 500), patch_size_range=(1, 10), image_size_max=5000)

    def test_without_overlap_without_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_without_overlap_without_remainder(spatial_dims=spatial_dims, patch_size_range=(50, 500), image_size_multiple=(1, 50))

    def test_without_overlap_with_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_without_overlap_with_remainder(spatial_dims=spatial_dims, patch_size_range=(50, 500), image_size_max=5000)

    def test_with_overlap_without_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_with_overlap_without_remainder(spatial_dims=spatial_dims, patch_overlap_size_range=(50, 500), patch_size_range=(1, 10), image_size_multiple=(1, 50))

    def test_with_overlap_with_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_with_overlap_with_remainder(spatial_dims=spatial_dims, patch_overlap_size_range=(50, 500), patch_size_range=(1, 10), image_size_max=5000)

    def test_without_channel(self):
        pass

    def test_channel_first(self):
        pass

    def test_channel_last(self):
        pass

    def test_batch_and_channel_dim(self):
        pass

    def test_multiple_non_spatial_dims(self):
        pass

    def test_numpy(self):
        pass

    def test_zarr(self):
        pass

    def test_without_patch_size(self):
        pass

    def test_patch_size_larger_than_image_size(self):
        pass

    def test_overlap_size_larger_than_patch_size(self):
        pass

    def _test_without_overlap_without_remainder(self, spatial_dims, patch_size_range, image_size_multiple):
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1

        assert np.sum(result) == result.size

    def _test_without_overlap_with_remainder(self, spatial_dims, patch_size_range, image_size_max):
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_max = self._adjust_to_dim(image_size_max, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [random.randint(patch_size[i], image_size_max) for i in range(spatial_dims)]
        image = np.random.random(image_size)
        quotient_size = np.floor_divide(image_size, patch_size) * patch_size

        # Test with image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size)

        for patch, patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1

        assert np.sum(result) == result.size

    def _test_with_overlap_without_remainder(self, spatial_dims, patch_overlap_size_range, patch_size_range, image_size_multiple):
        patch_overlap_size_range = tuple(self._adjust_to_dim(patch_overlap_size_range, spatial_dims))
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_overlap = [random.randint(*patch_overlap_size_range) for i in range(spatial_dims)]
        patch_size = [patch_overlap[i] * random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1

        assert np.sum(result) == result.size

    def _test_with_overlap_with_remainder(self, spatial_dims, patch_overlap_size_range, patch_size_range, image_size_max):
        patch_overlap_size_range = tuple(self._adjust_to_dim(patch_overlap_size_range, spatial_dims))
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_max = self._adjust_to_dim(image_size_max, spatial_dims)

        patch_overlap = [random.randint(*patch_overlap_size_range) for i in range(spatial_dims)]
        patch_size = [patch_overlap[i] * random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [random.randint(patch_size[i], image_size_max) for i in range(spatial_dims)]
        image = np.random.random(image_size)
        quotient_size = np.floor_divide(image_size, patch_size) * patch_size

        # Test with image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch, patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(image, patch_indices)] = 1

        assert np.sum(result) == result.size

    def _adjust_to_dim(self, arr, spatial_dims):
        return np.uint32(np.rint(np.power(arr, 1/spatial_dims)))
