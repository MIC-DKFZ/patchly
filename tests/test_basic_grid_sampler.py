import unittest
import random
import numpy as np
import zarr
from samplify.sampler import BasicGridSampler
from samplify.slicer import slicer


class TestBasicGridSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.max_value = 100000
        self.image_size_multiple = 10
        self.enable_prints = False

    def test_without_overlap_without_remainder_2d(self):
        self._test_without_overlap_without_remainder(spatial_dims=2, patch_size_range=(self.max_value, self.max_value), image_size_multiple=(1, self.image_size_multiple))

    def test_without_overlap_with_remainder_2d(self):
        self._test_without_overlap_with_remainder(spatial_dims=2, patch_size_range=(self.max_value, self.max_value*10), image_size_multiple=(1, self.image_size_multiple))

    def test_with_overlap_without_remainder_2d(self):
        self._test_with_overlap_without_remainder(spatial_dims=2, patch_overlap_size_range=(self.max_value, self.max_value*10), patch_size_range=(1, 100), image_size_multiple=(1, self.image_size_multiple))

    def test_with_overlap_with_remainder_2d(self):
        self._test_with_overlap_with_remainder(spatial_dims=2, patch_overlap_size_range=(self.max_value, self.max_value*10), patch_size_range=(1, 100), image_size_multiple=(1, self.image_size_multiple))

    def test_without_overlap_without_remainder_3d(self):
        self._test_without_overlap_without_remainder(spatial_dims=3, patch_size_range=(self.max_value, self.max_value*10), image_size_multiple=(1, self.image_size_multiple))

    def test_without_overlap_with_remainder_3d(self):
        self._test_without_overlap_with_remainder(spatial_dims=3, patch_size_range=(self.max_value, self.max_value*10), image_size_multiple=(1, self.image_size_multiple))

    def test_with_overlap_without_remainder_3d(self):
        self._test_with_overlap_without_remainder(spatial_dims=3, patch_overlap_size_range=(self.max_value, self.max_value*10), patch_size_range=(1, 100), image_size_multiple=(1, self.image_size_multiple))

    def test_with_overlap_with_remainder_3d(self):
        self._test_with_overlap_with_remainder(spatial_dims=3, patch_overlap_size_range=(self.max_value, self.max_value*10), patch_size_range=(1, 100), image_size_multiple=(1, self.image_size_multiple))

    def test_without_overlap_without_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_without_overlap_without_remainder(spatial_dims=spatial_dims, patch_size_range=(self.max_value, self.max_value*10), image_size_multiple=(1, self.image_size_multiple))

    def test_without_overlap_with_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_without_overlap_with_remainder(spatial_dims=spatial_dims, patch_size_range=(self.max_value, self.max_value*10), image_size_multiple=(1, self.image_size_multiple))

    def test_with_overlap_without_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_with_overlap_without_remainder(spatial_dims=spatial_dims, patch_overlap_size_range=(self.max_value, self.max_value), patch_size_range=(1, 100), image_size_multiple=(1, self.image_size_multiple))

    def test_with_overlap_with_remainder_Nd(self):
        spatial_dims = random.randint(1, 5)
        self._test_with_overlap_with_remainder(spatial_dims=spatial_dims, patch_overlap_size_range=(self.max_value, self.max_value), patch_size_range=(1, 100), image_size_multiple=(1, self.image_size_multiple))

    def test_without_channel(self):
        self._test_without_overlap_without_remainder(spatial_dims=2, patch_size_range=(self.max_value, self.max_value), image_size_multiple=(1, self.image_size_multiple))

    def test_channel_first(self):
        spatial_dims = 2
        patch_size_range = (self.max_value, self.max_value)
        image_size_multiple = (1, self.image_size_multiple)
        max_channels = 5
        channel_first = True

        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]

        num_channels = random.randint(1, max_channels)
        if channel_first:
            image = np.random.random((num_channels, *image_size))
        else:
            image = np.random.random((*image_size, num_channels))

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}".format(spatial_dims, len(sampler), image_size, patch_size))

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            assert np.array_equal(patch, image[slicer(image, slices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        assert np.sum(result) == result.size

    def test_channel_last(self):
        spatial_dims = 2
        patch_size_range = (self.max_value, self.max_value)
        image_size_multiple = (1, self.image_size_multiple)
        max_channels = 5
        channel_first = False

        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]

        num_channels = random.randint(1, max_channels)
        if channel_first:
            image = np.random.random((num_channels, *image_size))
        else:
            image = np.random.random((*image_size, num_channels))

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}".format(spatial_dims, len(sampler), image_size, patch_size))

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            assert np.array_equal(patch, image[slicer(image, slices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        assert np.sum(result) == result.size

    def test_batch_and_channel_dim(self):
        spatial_dims = 2
        patch_size_range = (self.max_value, self.max_value)
        image_size_multiple = (1, self.image_size_multiple)
        max_channels = 5
        max_batches = 5
        channel_first = random.randint(0, 1)

        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]

        num_channels = random.randint(1, max_channels)
        num_batches = random.randint(1, max_batches)
        if channel_first:
            image = np.random.random((num_batches, num_channels, *image_size))
        else:
            image = np.random.random((*image_size, num_channels, num_batches))

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}".format(spatial_dims, len(sampler), image_size, patch_size))

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            assert np.array_equal(patch, image[slicer(image, slices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        assert np.sum(result) == result.size

    def test_multiple_non_spatial_dims(self):
        spatial_dims = 2
        patch_size_range = (self.max_value, self.max_value)
        image_size_multiple = (1, self.image_size_multiple)
        max_non_spatial_dims = 5
        max_non_spatial_dims_size = 5
        channel_first = random.randint(0, 1)

        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]

        num_non_spatials = [random.randint(1, max_non_spatial_dims_size) for _ in range(max_non_spatial_dims)]
        if channel_first:
            image = np.random.random((*num_non_spatials, *image_size))
        else:
            image = np.random.random((*image_size, *num_non_spatials))

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size, channel_first=channel_first)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}".format(spatial_dims, len(sampler), image_size, patch_size))

        for patch, patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1
            assert np.array_equal(patch, image[slicer(image, slices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            slices = self.get_slices(result, image_size, patch_indices, channel_first)
            result[slicer(result, slices)] = 1

        assert np.sum(result) == result.size

    def test_numpy(self):
        self._test_without_overlap_without_remainder(spatial_dims=2, patch_size_range=(self.max_value, self.max_value), image_size_multiple=(1, self.image_size_multiple))

    def test_zarr(self):
        spatial_dims = 2
        patch_size_range = (self.max_value, self.max_value)
        image_size_multiple = (1, self.image_size_multiple)

        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]
        image = np.random.random(image_size)
        image = zarr.array(image)

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}".format(spatial_dims, len(sampler), image_size, patch_size))

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        assert np.sum(result) == result.size

    def test_without_patch_size(self):
        spatial_dims = 2
        patch_size_range = (self.max_value, self.max_value)
        image_size_multiple = (1, self.image_size_multiple)

        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]
        image = np.random.random(image_size)
        image = zarr.array(image)

        # Test with image
        result = np.zeros_like(image)
        self.assertRaises(RuntimeError, BasicGridSampler, image=image, image_size=image_size)

    def test_patch_size_larger_than_image_size(self):
        spatial_dims = 2
        patch_size_range = (self.max_value, self.max_value)
        image_size_multiple = (2, self.image_size_multiple)

        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]
        image = np.random.random(image_size)
        image = zarr.array(image)

        # Test with image
        result = np.zeros_like(image)
        self.assertRaises(RuntimeError, BasicGridSampler, image=image, image_size=patch_size, patch_size=image_size)

    def test_overlap_size_larger_than_patch_size(self):
        spatial_dims = 2
        patch_overlap_size_range = (self.max_value, self.max_value * 10)
        patch_size_range = (2, 100)
        image_size_multiple = (1, self.image_size_multiple)

        patch_overlap_size_range = tuple(self._adjust_to_dim(patch_overlap_size_range, spatial_dims))
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_overlap = [random.randint(*patch_overlap_size_range) for i in range(spatial_dims)]
        patch_size = [patch_overlap[i] * random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        self.assertRaises(RuntimeError, BasicGridSampler, image=image, image_size=image_size, patch_size=patch_overlap, patch_overlap=patch_size)

    def _test_without_overlap_without_remainder(self, spatial_dims, patch_size_range, image_size_multiple):
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [patch_size[i] * random.randint(*image_size_multiple) for i in range(spatial_dims)]
        image = np.random.random(image_size)

        # Test with image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}".format(spatial_dims, len(sampler), image_size, patch_size))

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        assert np.sum(result) == result.size

    def _test_without_overlap_with_remainder(self, spatial_dims, patch_size_range, image_size_multiple):
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_size = [random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [random.randint(patch_size[i], patch_size[i]*image_size_multiple[1]) for i in range(spatial_dims)]
        image = np.random.random(image_size)
        quotient_size = np.floor_divide(image_size, patch_size) * patch_size

        # Test with image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}".format(spatial_dims, len(sampler), image_size, patch_size))

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

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
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}, patch_overlap: {}".format(spatial_dims, len(sampler), image_size, patch_size, patch_overlap))

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros_like(image)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        assert np.sum(result) == result.size

    def _test_with_overlap_with_remainder(self, spatial_dims, patch_overlap_size_range, patch_size_range, image_size_multiple):
        patch_overlap_size_range = tuple(self._adjust_to_dim(patch_overlap_size_range, spatial_dims))
        patch_size_range = tuple(self._adjust_to_dim(patch_size_range, spatial_dims))
        image_size_multiple = self._adjust_to_dim(image_size_multiple, spatial_dims)

        patch_overlap = [random.randint(*patch_overlap_size_range) for i in range(spatial_dims)]
        patch_size = [patch_overlap[i] * random.randint(*patch_size_range) for i in range(spatial_dims)]
        image_size = [random.randint(patch_size[i], patch_size[i] * image_size_multiple[1]) for i in range(spatial_dims)]
        image = np.random.random(image_size)
        quotient_size = np.floor_divide(image_size, patch_size) * patch_size

        # Test with image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image=image, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)
        if self.enable_prints:
            print("Dim: {}, sampler len: {}, image_size: {}, patch_size: {}, patch_overlap: {}".format(spatial_dims, len(sampler), image_size, patch_size, patch_overlap))

        for patch, patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1
            assert np.array_equal(patch, image[slicer(image, patch_indices)])

        assert np.sum(result) == result.size

        # Test without image
        result = np.zeros(quotient_size)
        sampler = BasicGridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

        for patch_indices in sampler:
            result[slicer(result, patch_indices)] = 1

        assert np.sum(result) == result.size

    def _adjust_to_dim(self, arr, spatial_dims):
        return np.int64(np.rint(np.power(arr, 1/spatial_dims)))

    def get_slices(self, image, image_size, patch_indices, channel_first):
        non_image_dims = len(image.shape) - len(image_size)
        if channel_first:
            slices = [None] * non_image_dims
            slices.extend([index_pair.tolist() for index_pair in patch_indices])
        else:
            slices = [index_pair.tolist() for index_pair in patch_indices]
            slices.extend([None] * non_image_dims)
        return slices
