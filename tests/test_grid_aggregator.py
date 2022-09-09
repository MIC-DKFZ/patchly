import unittest
import numpy as np
import zarr
from samplify.sampler import GridSampler
from samplify.aggregator import Aggregator
from samplify.slicer import slicer
from scipy.ndimage.filters import gaussian_filter


class TestGridSampler(unittest.TestCase):
    def test_without_overlap_without_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_without_overlap_with_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_with_overlap_without_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_2d(self):
        patch_overlap = (5, 5)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_without_overlap_without_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_without_overlap_with_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_with_overlap_without_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_3d(self):
        patch_overlap = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_without_overlap_without_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_without_overlap_with_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_with_overlap_without_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_with_overlap_with_remainder_Nd(self):
        patch_overlap = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_overlap=patch_overlap)

    def test_channel_first(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((3, *spatial_size))
        spatial_first = False

        self._test_aggregator(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_channel_last(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((*spatial_size, 5))
        spatial_first = True

        self._test_aggregator(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_batch_and_channel_dim(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((4, 3, *spatial_size))
        spatial_first = False

        self._test_aggregator(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_multiple_non_spatial_dims(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((5, 4, 3, *spatial_size))
        spatial_first = False

        self._test_aggregator(image, spatial_size, patch_size, spatial_first=spatial_first)

    def test_zarr(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)
        image = zarr.array(image)
        output = np.zeros_like(image)
        output = zarr.array(output)

        self._test_aggregator(image, spatial_size, patch_size, output=output)

    def test_gaussian_weights(self):
        patch_size = (50, 100)
        patch_overlap = (25, 100)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        expected_output = np.zeros_like(image)
        weights = self.create_gaussian_weights(np.asarray(patch_size))
        weight_map = np.zeros_like(image)

        expected_output[:50, :] += image[:50, :] * weights * 0
        expected_output[25:75, :] += image[25:75, :] * weights * 1
        expected_output[50:, :] += image[50:, :] * weights * 2
        weight_map[:50, :] += weights
        weight_map[25:75, :] += weights
        weight_map[50:, :] += weights
        expected_output /= weight_map
        expected_output = np.nan_to_num(expected_output)

        output1, output2 = self._test_aggregator(image, spatial_size, patch_size, patch_overlap=patch_overlap, weights='gaussian', multiply_patch_by_index=True)

        np.testing.assert_almost_equal(output1, expected_output, decimal=4)
        np.testing.assert_almost_equal(output2, expected_output, decimal=4)

    # def test_patch_size_larger_than_spatial_size(self):
    #     patch_size = (101, 100)
    #     spatial_size = (100, 100)
    #     image = np.random.random(spatial_size)
    #     image = zarr.array(image)
    #
    #     self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, mode="sample_edge")
    #
    # def test_overlap_size_larger_than_patch_size(self):
    #     patch_overlap = (11, 10)
    #     patch_size = (10, 10)
    #     spatial_size = (100, 100)
    #     image = np.random.random(spatial_size)
    #
    #     self.assertRaises(RuntimeError, GridSampler, image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, mode="sample_edge")

    def _test_aggregator(self, image, spatial_size, patch_size, patch_overlap=None, spatial_first=True, output=None, weights='avg', multiply_patch_by_index=False):
        # Test with output size
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, spatial_first=spatial_first, mode="sample_edge")
        aggregator = Aggregator(sampler=sampler, output_size=image.shape, weights=weights)

        for i, (patch, patch_indices) in enumerate(sampler):
            if multiply_patch_by_index:
                _patch = patch * i
            else:
                _patch = patch
            aggregator.append(_patch, patch_indices)

        output1 = aggregator.get_output()

        if not multiply_patch_by_index:
            np.testing.assert_almost_equal(image, output1, decimal=6)

        # Test without output array
        if output is None:
            output = np.zeros_like(image)
        sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, spatial_first=spatial_first, mode="sample_edge")
        aggregator = Aggregator(sampler=sampler, output=output)

        for i, (patch, patch_indices) in enumerate(sampler):
            if multiply_patch_by_index:
                _patch = patch * i
            else:
                _patch = patch
            aggregator.append(_patch, patch_indices)

        output2 = aggregator.get_output()

        if not multiply_patch_by_index:
            np.testing.assert_almost_equal(image, output2, decimal=6)

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