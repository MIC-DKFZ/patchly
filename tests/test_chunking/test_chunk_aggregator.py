import unittest
import numpy as np
import zarr
from samplify.sampler import GridSampler
from samplify.aggregator import Aggregator
from samplify.slicer import slicer
from scipy.ndimage.filters import gaussian_filter
import copy


class TestChunkAggregator(unittest.TestCase):
    def test_without_overlap_without_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        chunk_size = (50, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size)

    def test_without_overlap_with_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (103, 107)
        chunk_size = (50, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size)

    def test_with_overlap_without_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, patch_overlap=patch_overlap)

    def test_without_overlap_without_remainder_3d(self):
        patch_size = (10, 10, 5)
        chunk_size = (50, 50, 25)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size)

    def test_without_overlap_with_remainder_3d(self):
        patch_size = (10, 10, 5)
        chunk_size = (50, 50, 25)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size)

    def test_with_overlap_without_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        chunk_size = (50, 50, 25)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        chunk_size = (50, 50, 25)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, patch_overlap=patch_overlap)

    def test_without_overlap_without_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        chunk_size = (4, 16, 8, 8, 8)
        spatial_size = (8, 32, 16, 16, 8)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size)

    def test_without_overlap_with_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        chunk_size = (4, 16, 8, 8, 8)
        spatial_size = (10, 33, 18, 19, 9)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size)

    def test_with_overlap_without_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        chunk_size = (4, 16, 8, 8, 8)
        spatial_size = (8, 32, 16, 16, 8)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        chunk_size = (4, 16, 8, 8, 8)
        spatial_size = (10, 33, 18, 19, 9)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, patch_overlap=patch_overlap)

    def test_channel_first(self):
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (100, 100)
        image = np.random.random((3, *spatial_size))
        spatial_first_sampler = False
        spatial_first_aggregator = False

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_channel_last(self):
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (100, 100)
        image = np.random.random((*spatial_size, 5))
        spatial_first_sampler = True
        spatial_first_aggregator = True

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_batch_and_channel_dim(self):
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (100, 100)
        image = np.random.random((4, 3, *spatial_size))
        spatial_first_sampler = False
        spatial_first_aggregator = False

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_multiple_non_spatial_dims(self):
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (100, 100)
        image = np.random.random((5, 4, 3, *spatial_size))
        spatial_first_sampler = False
        spatial_first_aggregator = False

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_zarr(self):
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)
        image = zarr.array(image)
        output = np.zeros_like(image)
        output = zarr.array(output)

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, output=output)

    def test_gaussian_weights(self):
        patch_size = (10,)
        patch_overlap = (5,)
        chunk_size = (20,)
        spatial_size = (40,)
        image = np.random.random(spatial_size).astype(np.float32)

        expected_output = np.zeros_like(image, dtype=np.float32)
        weights = self.create_gaussian_weights(np.asarray(patch_size)).astype(np.float32)
        weight_map = np.zeros_like(image, dtype=np.float32)

        for i, index in enumerate(range(0, spatial_size[0]-patch_overlap[0], patch_overlap[0])):
            expected_output[index:index+patch_size[0]] += image[index:index+patch_size[0]] * weights * 2
            weight_map[index:index+patch_size[0]] += weights
        expected_output /= weight_map
        expected_output = np.nan_to_num(expected_output)

        output1, output2 = self._test_aggregator(image, spatial_size, patch_size, chunk_size, patch_overlap=patch_overlap, weights='gaussian', multiply_elements_by_two=True)

        np.testing.assert_almost_equal(output1, expected_output, decimal=6)
        np.testing.assert_almost_equal(output2, expected_output, decimal=6)

    def test_softmax(self):
        patch_size = (10, 10)
        chunk_size = (50, 50)
        spatial_size = (100, 100)
        image = np.random.random((3, *spatial_size))
        output = np.zeros(spatial_size, dtype=np.uint16)
        spatial_first_sampler = False
        spatial_first_aggregator = False
        softmax_dim = 0

        self._test_aggregator(image, spatial_size, patch_size, chunk_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator, output=output, softmax_dim=softmax_dim)

    def _test_aggregator(self, image, spatial_size, patch_size, chunk_size, patch_overlap=None, spatial_first_sampler=True, spatial_first_aggregator=True, output=None, weights='avg', multiply_elements_by_two=False, softmax_dim=None):
        if softmax_dim is None:
            output_size = image.shape
        else:
            output_size = np.moveaxis(image.shape, softmax_dim, 0)[1:]
        
        # Test with output size
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, spatial_first=spatial_first_sampler, mode="sample_edge")
        aggregator = Aggregator(sampler=sampler, output_size=output_size, chunk_size=chunk_size, weights=weights, spatial_first=spatial_first_aggregator, softmax_dim=softmax_dim)

        for i, (patch, patch_indices) in enumerate(sampler):
            if multiply_elements_by_two:
                _patch = patch * 2
            else:
                _patch = patch
            aggregator.append(_patch, patch_indices)

        output1 = aggregator.get_output()

        if not multiply_elements_by_two:
            if softmax_dim is None:
                np.testing.assert_almost_equal(np.array(image), np.array(output1), decimal=6)
            else:
                np.testing.assert_almost_equal(np.array(image).argmax(axis=softmax_dim).astype(np.uint16), np.array(output1), decimal=6)

        # Test without output array
        if output is None:
            output = np.zeros_like(image)
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, spatial_first=spatial_first_sampler, mode="sample_edge")
        aggregator = Aggregator(sampler=sampler, output=output, chunk_size=chunk_size, weights=weights, spatial_first=spatial_first_aggregator, softmax_dim=softmax_dim)

        for i, (patch, patch_indices) in enumerate(sampler):
            if multiply_elements_by_two:
                _patch = patch * 2
            else:
                _patch = patch
            aggregator.append(_patch, patch_indices)

        output2 = aggregator.get_output()

        if not multiply_elements_by_two:
            if softmax_dim is None:
                np.testing.assert_almost_equal(np.array(image), np.array(output2), decimal=6)
            else:
                np.testing.assert_almost_equal(np.array(image).argmax(axis=softmax_dim).astype(np.uint16), np.array(output2), decimal=6)

        return output1, output2

    def create_gaussian_weights(self, size):
        sigma_scale = 1. / 8
        sigmas = size * sigma_scale
        center_coords = size // 2
        tmp = np.zeros(size)
        tmp[tuple(center_coords)] = 1
        gaussian_weights = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_weights[gaussian_weights == 0] = np.min(gaussian_weights[gaussian_weights != 0])
        return gaussian_weights


if __name__ == '__main__':
    unittest.main()