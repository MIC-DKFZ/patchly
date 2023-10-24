import unittest
import numpy as np
import zarr
from patchly import GridSampler, Aggregator, SamplingMode, utils
import copy
import torch


class TestSqueezeAggregator(unittest.TestCase):
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
        step_size = (5, 5)
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, step_size=step_size)

    def test_with_offset_with_remainder_2d(self):
        step_size = (5, 5)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, step_size=step_size)

    def test_with_offset_with_remainder_2d_v2(self):
        step_size = (3, 3)
        patch_size = (10, 10)
        spatial_size = (103, 107)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, step_size=step_size)

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
        step_size = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (100, 100, 50)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, step_size=step_size)

    def test_with_offset_with_remainder_3d(self):
        step_size = (5, 5, 5)
        patch_size = (10, 10, 5)
        spatial_size = (103, 107, 51)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, step_size=step_size)

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
        step_size = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (4, 16, 8, 8, 4)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, step_size=step_size)

    def test_with_offset_with_remainder_Nd(self):
        step_size = (1, 8, 2, 2, 2)
        patch_size = (2, 8, 4, 4, 4)
        spatial_size = (5, 18, 9, 10, 6)
        image = np.random.random(spatial_size)

        self._test_aggregator(image, spatial_size, patch_size, step_size=step_size)

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

    def test_tensor(self):
        patch_size = (10, 10)
        spatial_size = (100, 100)
        image = np.random.random(spatial_size)
        image = torch.tensor(image)
        output = np.zeros_like(image)
        output = torch.tensor(output)

        self._test_aggregator(image, spatial_size, patch_size, output=output)

    def test_gaussian_weights(self):
        patch_size = (10, 10)
        step_size = (5, 10)
        spatial_size = (20, 10)
        image = np.random.random(spatial_size)

        expected_output = np.zeros_like(image)
        weights = utils.gaussian_kernel_numpy(np.asarray(patch_size), dtype=np.float64)
        weight_map = np.zeros_like(image)

        expected_output[:10, :] += image[:10, :] * weights * 0
        expected_output[5:15, :] += image[5:15, :] * weights * 1
        expected_output[10:, :] += image[10:, :] * weights * 2
        weight_map[:10, :] += weights
        weight_map[5:15, :] += weights
        weight_map[10:, :] += weights
        expected_output /= weight_map
        expected_output = np.nan_to_num(expected_output)

        output1, output2 = self._test_aggregator(image, spatial_size, patch_size, step_size=step_size, weights='gaussian', multiply_patch_by_index=True)

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

    def _test_aggregator(self, image, spatial_size, patch_size, step_size=None, spatial_first_sampler=True, spatial_first_aggregator=True, output=None, weights='avg', multiply_patch_by_index=False, softmax_dim=None):        
        # Test with output size
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, step_size=step_size, spatial_first=spatial_first_sampler, mode=SamplingMode.SAMPLE_SQUEEZE)
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
                if not isinstance(output1, torch.Tensor):
                    np.testing.assert_almost_equal(image, output1, decimal=6)
                else:
                    torch.testing.assert_close(image.to(dtype=torch.float32), output1.to(dtype=torch.float32))
            else:
                if not isinstance(output1, torch.Tensor):
                    np.testing.assert_almost_equal(image.argmax(axis=softmax_dim).astype(np.uint16), output1, decimal=6)
                else:
                    torch.testing.assert_close(image.argmax(axis=softmax_dim).to(dtype=torch.int32), output1.to(dtype=torch.int32))

        # Test without output array
        if output is None:
            output = np.zeros_like(image)
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, step_size=step_size, spatial_first=spatial_first_sampler, mode=SamplingMode.SAMPLE_SQUEEZE)
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
                if not isinstance(output2, torch.Tensor):
                    np.testing.assert_almost_equal(image, output2, decimal=6)
                else:
                    torch.testing.assert_close(image.to(dtype=torch.float32), output2.to(dtype=torch.float32))
            else:
                if not isinstance(output2, torch.Tensor):
                    np.testing.assert_almost_equal(image.argmax(axis=softmax_dim).astype(np.uint16), output2, decimal=6)
                else:
                    torch.testing.assert_close(image.argmax(axis=softmax_dim).to(dtype=torch.int32), output2.to(dtype=torch.int32))

        return output1, output2


if __name__ == '__main__':
    unittest.main()