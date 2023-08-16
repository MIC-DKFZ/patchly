import unittest
import numpy as np
import zarr
from samplify.sampler import GridSampler
from samplify.aggregator import Aggregator
from samplify.slicer import slicer
from scipy.ndimage.filters import gaussian_filter
import copy


class TestEdgeAggregator(unittest.TestCase):
    def test_without_offset_without_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_without_offset_with_remainder_2d(self):
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_with_offset_without_remainder_2d(self):
        patch_offset = (5, 5)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_2d(self):
        patch_offset = (5, 5)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_2d_v2(self):
        patch_offset = (3, 3)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_without_offset_without_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_without_offset_with_remainder_3d(self):
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_with_offset_without_remainder_3d(self):
        patch_offset = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_3d(self):
        patch_offset = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_without_offset_without_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_without_offset_with_remainder_Nd(self):
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size)

    def test_with_offset_without_remainder_Nd(self):
        patch_offset = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_with_offset_with_remainder_Nd(self):
        patch_offset = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)

    def test_channel_first(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((3, *spatial_size))
        spatial_first_sampler = False
        spatial_first_aggregator = False

        self._test_aggregator(image, spatial_size, patch_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_channel_last(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((*spatial_size, 5))
        spatial_first_sampler = True
        spatial_first_aggregator = True

        self._test_aggregator(image, spatial_size, patch_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_batch_and_channel_dim(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((4, 3, *spatial_size))
        spatial_first_sampler = False
        spatial_first_aggregator = False

        self._test_aggregator(image, spatial_size, patch_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_multiple_non_spatial_dims(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((5, 4, 3, *spatial_size))
        spatial_first_sampler = False
        spatial_first_aggregator = False

        self._test_aggregator(image, spatial_size, patch_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator)

    def test_zarr(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)
        image = zarr.array(image)
        output = np.zeros_like(image)
        output = zarr.array(output)

        self._test_aggregator(image, spatial_size, patch_size, output=output)

    def test_gaussian_weights(self):
        patch_size = (10, 10)
        patch_offset = (5, 10)
        spatial_size = (20, 10)
        image = np.random.random(spatial_size)

        expected_output = np.zeros_like(image)
        weights = self.create_gaussian_weights(np.asarray(patch_size))
        weight_map = np.zeros_like(image)

        expected_output[:10, :] += image[:10, :] * weights * 0
        expected_output[5:15, :] += image[5:15, :] * weights * 1
        expected_output[10:, :] += image[10:, :] * weights * 2
        weight_map[:10, :] += weights
        weight_map[5:15, :] += weights
        weight_map[10:, :] += weights
        expected_output /= weight_map
        expected_output = np.nan_to_num(expected_output)

        output1, output2 = self._test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset, weights='gaussian', multiply_patch_by_index=True)

        np.testing.assert_almost_equal(output1, expected_output, decimal=6)
        np.testing.assert_almost_equal(output2, expected_output, decimal=6)

    def test_softmax(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random((3, *spatial_size))
        output = np.zeros((3, *spatial_size))
        spatial_first_sampler = False
        spatial_first_aggregator = False

        self._test_aggregator(image, spatial_size, patch_size, spatial_first_sampler=spatial_first_sampler, spatial_first_aggregator=spatial_first_aggregator, output=output, softmax_dim=0)

    def _test_aggregator(self, image, spatial_size, patch_size, patch_offset=None, spatial_first_sampler=True, spatial_first_aggregator=True, output=None, weights='avg', multiply_patch_by_index=False, softmax_dim=None):        
        # Test with output size
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, patch_offset=patch_offset, spatial_first=spatial_first_sampler, mode="sample_edge")
        aggregator = Aggregator(sampler=sampler, output_size=image.shape, weights=weights, spatial_first=spatial_first_aggregator, softmax_dim=softmax_dim)

        for i, (patch, patch_bbox) in enumerate(sampler):
            if multiply_patch_by_index:
                _patch = patch * i
            else:
                _patch = patch
            aggregator.append(_patch, patch_bbox)

        output1 = aggregator.get_output()

        if not multiply_patch_by_index:
            if softmax_dim is None:
                np.testing.assert_almost_equal(image, output1, decimal=6)
            else:
                np.testing.assert_almost_equal(image.argmax(axis=softmax_dim).astype(np.uint16), output1, decimal=6)

        # Test without output array
        if output is None:
            output = np.zeros_like(image)
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, patch_offset=patch_offset, spatial_first=spatial_first_sampler, mode="sample_edge")
        aggregator = Aggregator(sampler=sampler, output=output, weights=weights, spatial_first=spatial_first_aggregator, softmax_dim=softmax_dim)

        for i, (patch, patch_bbox) in enumerate(sampler):
            if multiply_patch_by_index:
                _patch = patch * i
            else:
                _patch = patch
            aggregator.append(_patch, patch_bbox)

        output2 = aggregator.get_output()

        if not multiply_patch_by_index:
            if softmax_dim is None:
                np.testing.assert_almost_equal(image, output2, decimal=6)
            else:
                np.testing.assert_almost_equal(image.argmax(axis=softmax_dim).astype(np.uint16), output2, decimal=6)

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