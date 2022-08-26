import unittest
import random
import numpy as np
from samplify.sampler import BasicGridSampler
from samplify.slicer import slicer


class TestBasicGridSampler(unittest.TestCase):
    def test_without_overlap_without_remainder_2d(self):
        patch_size = (random.randint(50, 500), random.randint(50, 500))
        image_size = (patch_size[0] * random.randint(1, 50), patch_size[1] * random.randint(1, 50))
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

    def test_without_overlap_with_remainder_2d(self):
        patch_size = (random.randint(50, 500), random.randint(50, 500))
        image_size = (random.randint(patch_size[0], 5000), random.randint(patch_size[1], 5000))
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

    def test_with_overlap_without_remainder_2d(self):
        patch_overlap = (random.randint(50, 500), random.randint(50, 500))
        patch_size = (patch_overlap[0] * random.randint(1, 10), patch_overlap[1] * random.randint(1, 10))
        image_size = (patch_size[0] * random.randint(1, 50), patch_size[1] * random.randint(1, 50))
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

    def test_with_overlap_with_remainder_2d(self):
        patch_overlap = (random.randint(50, 500), random.randint(50, 500))
        patch_size = (patch_overlap[0] * random.randint(1, 10), patch_overlap[1] * random.randint(1, 10))
        image_size = (random.randint(patch_size[0], 5000), random.randint(patch_size[1], 5000))
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

    def test_without_overlap_without_remainder_3d(self):
        pass

    def test_without_overlap_with_remainder_3d(self):
        pass

    def test_with_overlap_without_remainder_3d(self):
        pass

    def test_with_overlap_with_remainder_3d(self):
        pass

    def test_without_overlap_without_remainder_Nd(self):
        pass

    def test_without_overlap_with_remainder_Nd(self):
        pass

    def test_with_overlap_without_remainder_Nd(self):
        pass

    def test_with_overlap_with_remainder_Nd(self):
        pass

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