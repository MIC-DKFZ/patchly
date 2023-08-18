import unittest
import numpy as np
import zarr
from samplify import GridSampler, Aggregator, SamplingMode
import copy


def _test_aggregator(image, spatial_size, patch_size, patch_offset=None, spatial_first_sampler=True, spatial_first_aggregator=True, output=None, weights='avg', multiply_patch_by_index=False, softmax_dim=None):        
        # Test with output size
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, patch_offset=patch_offset, spatial_first=spatial_first_sampler, mode=SamplingMode.SAMPLE_SQUEEZE)
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
        sampler = GridSampler(image=copy.deepcopy(image), spatial_size=spatial_size, patch_size=patch_size, patch_offset=patch_offset, spatial_first=spatial_first_sampler, mode=SamplingMode.SAMPLE_SQUEEZE)
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

patch_offset = (3, 3)
patch_size = (10, 10)
spatial_size = (103, 107)
image = np.random.random(spatial_size)

_test_aggregator(image, spatial_size, patch_size, patch_offset=patch_offset)