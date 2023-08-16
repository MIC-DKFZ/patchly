import numpy as np
from samplify.sampler import GridSampler, _AdaptiveGridSampler
from samplify.slicer import slicer
from samplify import utils
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict
import copy
import concurrent.futures
from typing import Union, Optional, Tuple, Callable
import numpy.typing as npt
from enum import Enum
import zarr
import warnings


class PatchStatus(Enum):
    EMPTY = 1
    FILLED = 2
    COMPLETED = 3


class Aggregator:
    def __init__(self, sampler: GridSampler, output_size: Optional[Union[Tuple, npt.ArrayLike]] = None, output: Optional[npt.ArrayLike] = None, chunk_size: Optional[Union[Tuple, npt.ArrayLike]] = None, 
                 weights: Union[str, Callable] = 'avg', softmax_dim: Optional[int] = None, spatial_first: bool = True):
        self.sampler = sampler
        self.spatial_size_s = sampler.spatial_size_s
        self.patch_size_s = sampler.patch_size_s
        self.patch_overlap_s = sampler.patch_overlap_s
        self.chunk_size_s = chunk_size
        self.spatial_first = spatial_first
        self.mode = sampler.mode
        self.softmax_dim = softmax_dim
        self.output_h = self.set_output(output, output_size)
        self.weight_patch_s, self.weight_map_s = self.set_weights(weights)
        self.check_sanity()
        self.aggregator = self.set_aggregator(self.sampler, self.output_h, self.softmax_dim)

    def set_output(self, output_h, output_size_h):
        if output_h is None and output_size_h is not None:
            output_h = np.zeros(output_size_h, dtype=np.float32)
            return output_h
        elif output_h is None and output_size_h is None:
            raise RuntimeError("Either the output array-like data or the output size must be given.")
        elif output_h is not None and output_size_h is not None and output_h.shape != output_size_h:
            raise RuntimeError("The variable output_size must be equal to the output shape if both are given. Only one of the two must be given.")
        return output_h

    def set_weights(self, weights_s):
        if weights_s == 'avg':
            weight_patch_s = np.ones(self.patch_size_s, dtype=np.uint8)
        elif weights_s == 'gaussian':
            weight_patch_s = self.create_gaussian_weights(self.patch_size_s)
        elif callable(weights_s):
            weight_patch_s = weights_s
        else:
            raise RuntimeError("The given type of weights is not supported.")

        if self.softmax_dim is None:
            if self.chunk_size_s is None:
                weight_map_size_s = self.spatial_size_s
            else:
                weight_map_size_s = self.chunk_size_s
            if weights_s == 'avg':
                weight_map_s = np.zeros(weight_map_size_s, dtype=np.uint16)
            else:
                weight_map_s = np.zeros(weight_map_size_s, dtype=np.float32)
        else:
            weight_map_s = None
        return weight_patch_s, weight_map_s
    
    def create_gaussian_weights(self, size_s):
        sigma_scale = 1. / 8
        sigmas = size_s * sigma_scale
        center_coords = size_s // 2
        tmp = np.zeros(size_s)
        tmp[tuple(center_coords)] = 1
        gaussian_weights_s = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_weights_s[gaussian_weights_s == 0] = np.min(gaussian_weights_s[gaussian_weights_s != 0])
        return gaussian_weights_s

    def check_sanity(self):
        if not hasattr(self.output_h, '__getitem__'):
            raise RuntimeError("The given output is not ArrayLike.")
        if self.spatial_first and (self.output_h.shape[:len(self.spatial_size_s)] != tuple(self.spatial_size_s)):
            raise RuntimeError("The spatial size of the given output {} is unequal to the given spatial size {}.".format(self.output_h.shape[:len(self.spatial_size_s)], self.spatial_size_s))
        if (not self.spatial_first) and (self.output_h.shape[-len(self.spatial_size_s):] != tuple(self.spatial_size_s)):
            raise RuntimeError("The spatial size of the given output {} is unequal to the given spatial size {}.".format(self.output_h.shape[-len(self.spatial_size_s):], self.spatial_size_s))
        if self.chunk_size_s is not None and np.any(self.chunk_size_s > self.spatial_size_s):
            raise RuntimeError("The chunk size ({}) cannot be greater than the spatial size ({}) in one or more dimensions.".format(self.chunk_size_s, self.spatial_size_s))
        if self.chunk_size_s is not None and np.any(self.patch_size_s >= self.chunk_size_s):
            raise RuntimeError("The patch size ({}) cannot be greater or equal to the chunk size ({}) in one or more dimensions.".format(self.patch_size_s, self.chunk_size_s))

    def set_aggregator(self, sampler, output_h, softmax_dim):
        if self.mode.startswith('sample_') and self.chunk_size_s is None:
            aggregator = _Aggregator(sampler=sampler, spatial_size_s=self.spatial_size_s, patch_size_s=self.patch_size_s,
                                  output_h=output_h, spatial_first=self.spatial_first, softmax_dim=softmax_dim, weight_patch_s=self.weight_patch_s, weight_map_s=self.weight_map_s)
        elif self.mode.startswith('sample_') and self.chunk_size_s is not None:
            aggregator = _ChunkAggregator(sampler=sampler, spatial_size_s=self.spatial_size_s, patch_size_s=self.patch_size_s, patch_overlap_s=self.patch_overlap_s, chunk_size_s=self.chunk_size_s,
                                       output_h=output_h, spatial_first=self.spatial_first, softmax_dim=softmax_dim, weight_patch_s=self.weight_patch_s, mode=self.mode)
        elif self.mode.startswith('pad_') and self.chunk_size_s is None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        elif self.mode.startswith('pad_') and self.chunk_size_s is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        else:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        if self.chunk_size_s is not None and len(self.spatial_size_s) != len(self.chunk_size_s):
            raise RuntimeError("The dimensionality of the chunk size ({}) is required to be the same as the spatial size ({})."
                               .format(self.chunk_size_s, self.spatial_size_s))
        if self.mode.startswith('pad_') and self.chunk_size_s is not None:
            raise RuntimeError("The given sampling mode ({}) is not compatible with chunk sampling.".format(self.mode))
        return aggregator

    def append(self, patch, patch_bbox):
        self.aggregator.append(patch, patch_bbox)

    def get_output(self, inplace: bool = False):
        output_h = self.aggregator.get_output(inplace)
        if self.sampler.pad_width is not None:
            output_h = self.unpad_output(output_h, self.sampler.pad_width)
        return output_h

    def unpad_output(self, output_h, pad_width_h):
        pad_width_h[:, 1] *= -1
        crop_bbox_h = slicer(output_h, pad_width_h)
        output_h = output_h[crop_bbox_h]
        return output_h


class _Aggregator:
    def __init__(self, sampler: GridSampler, spatial_size_s: Union[Tuple, npt.ArrayLike], patch_size_s: Union[Tuple, npt.ArrayLike],
                 output_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True, softmax_dim: Optional[int] = None,
                 weight_patch_s: npt.ArrayLike = None, weight_map_s: npt.ArrayLike = None):
        """
        Aggregator to assemble an image with continuous content from patches. The content of overlapping patches is averaged.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param output: A numpy-style zero-initialized output (Numpy, Samplify.Subject, Tensor, Zarr, Dask, ...) of a continuous data type.
        If none then a zero-initialized Numpy output array of data type np.float32 is created internally.
        :param spatial_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        """
        self.sampler = sampler
        self.spatial_size_s = np.asarray(spatial_size_s)
        self.patch_size_s = np.asarray(patch_size_s)
        self.output_h = output_h
        self.spatial_first = spatial_first
        self.softmax_dim = softmax_dim
        self.weight_patch_s = weight_patch_s
        self.weight_patch_h = None
        self.weight_map_s = weight_map_s
        self.computed_inplace = False

    def append(self, patch, patch_bbox):
        """
        Appends a patch to the output.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_bbox: The patch bbox in the format of (w_start, w_end, h_start, h_end, d_start, d_end, ...).
        """
        patch_h = patch
        patch_bbox_s = patch_bbox

        if self.computed_inplace:
            raise RuntimeError("get_output() has already been called with inplace=True. Therefore, no further patches can be appended.")
        if self.weight_patch_h is None:
            self.weight_patch_h = utils.broadcast_to(self.weight_patch_s, utils.add_non_spatial_dims(self.weight_patch_s.shape, patch_h.shape, self.spatial_first), self.spatial_first)
        patch_bbox_h = utils.add_non_spatial_bbox_dims(patch_bbox_s, self.output_h, self.spatial_first)
        self.output_h[slicer(self.output_h, patch_bbox_h)] += patch_h.astype(self.output_h.dtype) * self.weight_patch_h.astype(self.output_h.dtype)
        if self.weight_map_s is not None:
            self.weight_map_s[slicer(self.weight_map_s, patch_bbox_s)][...] += self.weight_patch_s

    def get_output(self, inplace: bool = False):
        """
        Computes and returns the final aggregated output based on all provided patches. The content of overlapping patches is averaged.
        :param inplace: Computes the output inplace without allocating new memory. Afterwards, no further patches can be appended.
        :return: The final aggregated output.
        """
        if not inplace:
            output_h = np.copy(self.output_h)
        else:
            output_h = self.output_h

        if not inplace or (inplace and not self.computed_inplace):
            if self.weight_map_s is not None:
                weight_map_h = utils.broadcast_to(self.weight_map_s, utils.add_non_spatial_dims(self.weight_map_s.shape, output_h.shape, self.spatial_first), self.spatial_first)
                output_h[...] = output_h / weight_map_h.astype(output_h.dtype)
                output_h[...] = np.nan_to_num(output_h)
            if self.softmax_dim is not None:
                # Cannot be done inplace -> No [...]
                output_h = output_h.argmax(axis=self.softmax_dim)
            if inplace:
                self.computed_inplace = True
        return output_h


class _ChunkAggregator(_Aggregator):
    def __init__(self, sampler: GridSampler, spatial_size_s: Union[Tuple, npt.ArrayLike], patch_size_s: Union[Tuple, npt.ArrayLike], patch_overlap_s: Union[Tuple, npt.ArrayLike], chunk_size_s: Union[Tuple, npt.ArrayLike],
                 output_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True,
                 softmax_dim: Optional[int] = None, weight_patch_s: npt.ArrayLike = None, mode: str = 'sample_edge'):
        """
        Weighted aggregator to assemble an image with continuous content from patches. Returns the maximum class at each position of the image. The content of overlapping patches is gaussian-weighted by default.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param output: A numpy-style zero-initialized image (Numpy, Zarr, Dask, ...) of a continuous data type. If none then a zero-initialized Numpy image of data type np.float32 is created internally.
        :param spatial_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param weights: A weight map of size patch_size that should be used for weighting the importance of each value in a patch. Default is a gaussian weight map.
        :param low_memory_mode: Reduces memory consumption by more than 50% in comparison to the normal WeightedAggregator and Aggregator. However, the prediction quality is slightly reduced.
        """
        super().__init__(sampler=sampler, spatial_size_s=spatial_size_s, patch_size_s=patch_size_s, output_h=output_h, spatial_first=spatial_first, softmax_dim=softmax_dim, weight_patch_s=weight_patch_s, weight_map_s=None)
        self.patch_overlap_s = patch_overlap_s
        self.chunk_size_s = chunk_size_s
        self.chunk_dtype = self.set_chunk_dtype()
        self.mode = mode
        self.chunk_sampler, self.chunk_patch_dict, self.patch_chunk_dict = self.compute_patches()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def set_chunk_dtype(self):
        if self.softmax_dim is None:
            return self.output_h.dtype
        else:
            return np.float32

    def compute_patches(self):
        patch_sampler = self.sampler
        chunk_sampler = _AdaptiveGridSampler(spatial_size_s=self.spatial_size_s, patch_size_s=self.chunk_size_s, patch_overlap_s=self.chunk_size_s)
        chunk_patch_dict = defaultdict(dict)
        patch_chunk_dict = defaultdict(dict)
        
        for idx in range(len(patch_sampler)):
            patch_bbox_s = patch_sampler._get_bbox(idx)
            patch_h = utils.LazyArray()
            patch_chunk_dict[str(patch_bbox_s)]["patch"] = patch_h
            patch_chunk_dict[str(patch_bbox_s)]["chunks"] = []
            for chunk_id, chunk_bbox_s in enumerate(chunk_sampler):
                if utils.is_overlapping(chunk_bbox_s, patch_bbox_s):
                    # Shift to chunk coordinate system
                    valid_patch_bbox_s = patch_bbox_s - np.array([chunk_bbox_s[:, 0], chunk_bbox_s[:, 0]]).T
                    # Crop patch bbox to chunk bounds
                    valid_patch_bbox_s = np.array([[max(valid_patch_bbox_s[i][0], 0), min(valid_patch_bbox_s[i][1], chunk_bbox_s[i][1] - chunk_bbox_s[i][0])] for i in range(len(chunk_bbox_s))])
                    crop_patch_bbox_s = valid_patch_bbox_s + np.array([chunk_bbox_s[:, 0], chunk_bbox_s[:, 0]]).T - np.array([patch_bbox_s[:, 0], patch_bbox_s[:, 0]]).T
                    chunk_patch_dict[chunk_id][str(patch_bbox_s)] = {"valid_patch_bbox": valid_patch_bbox_s, "crop_patch_bbox": crop_patch_bbox_s, "patch": patch_h, "status": PatchStatus.EMPTY}
                    patch_chunk_dict[str(patch_bbox_s)]["chunks"].append(chunk_id)

        return chunk_sampler, chunk_patch_dict, patch_chunk_dict

    def append(self, patch, patch_bbox):
        """
        Appends a patch to the output.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_bbox: The patch bbox in the format of (w_start, w_end, h_start, h_end, d_start, d_end, ...).
        """
        patch_h = patch
        patch_bbox_s = patch_bbox

        if self.weight_patch_h is None:
            self.weight_patch_h = utils.broadcast_to(self.weight_patch_s, utils.add_non_spatial_dims(self.weight_patch_s.shape, patch_h.shape, self.spatial_first), self.spatial_first)

        self.patch_chunk_dict[str(patch_bbox_s)]["patch"].create(patch_h)

        for chunk_id in self.patch_chunk_dict[str(patch_bbox_s)]["chunks"]:
            self.chunk_patch_dict[chunk_id][str(patch_bbox_s)]["status"] = PatchStatus.FILLED
            if self.is_chunk_complete(chunk_id):
                if isinstance(self.output_h, zarr.core.Array):
                    warnings.warn("Ouput is a Zarr array. Switching to single threading for chunk processing. See issue #39 for more information.") # If issue is solved remove zarr and warnings import statements
                    self.process_chunk(chunk_id)
                else:
                    self.executor.submit(self.process_chunk, chunk_id)

    def is_chunk_complete(self, chunk_id):
        for patch_data in self.chunk_patch_dict[chunk_id].values():
            if patch_data["status"] != PatchStatus.FILLED:
                return False
        return True

    def process_chunk(self, chunk_id):
        patch_size_h = list(self.chunk_patch_dict[chunk_id].values())[0]["patch"].shape
        chunk_size_s = self.chunk_sampler.patch_sizes_s[chunk_id]
        chunk_size_h = utils.add_non_spatial_dims(chunk_size_s, patch_size_h, self.spatial_first)
        chunk_h = np.zeros(chunk_size_h, dtype=self.chunk_dtype)
        if self.softmax_dim is None:
            weight_map_h = np.zeros(chunk_size_h)
        for patch_data in self.chunk_patch_dict[chunk_id].values():
            valid_patch_bbox_s = patch_data["valid_patch_bbox"]
            crop_patch_bbox_s = patch_data["crop_patch_bbox"]
            patch_h = patch_data["patch"].data
            patch_data["patch"] = None
            patch_data["status"] = PatchStatus.COMPLETED
            crop_patch_bbox_h = utils.add_non_spatial_bbox_dims(crop_patch_bbox_s, chunk_h, self.spatial_first)
            valid_patch_bbox_h = utils.add_non_spatial_bbox_dims(valid_patch_bbox_s, chunk_h, self.spatial_first)
            valid_patch_h = patch_h[slicer(patch_h, crop_patch_bbox_h)]
            valid_weight_patch_h = self.weight_patch_h[slicer(self.weight_patch_h, crop_patch_bbox_h)]
            chunk_h[slicer(chunk_h, valid_patch_bbox_h)] += valid_patch_h.astype(chunk_h.dtype) * valid_weight_patch_h.astype(chunk_h.dtype)
            if self.softmax_dim is None:
                weight_map_h[slicer(weight_map_h, valid_patch_bbox_h)] += valid_weight_patch_h
        if self.softmax_dim is None:
            chunk_h = chunk_h / weight_map_h.astype(chunk_h.dtype)
            chunk_h = np.nan_to_num(chunk_h)
        else:
            # Argmax the softmax chunk
            chunk_h = chunk_h.argmax(axis=self.softmax_dim).astype(np.uint16)
        chunk_bbox_h = self.chunk_sampler.__getitem__(chunk_id)
        chunk_bbox_h = utils.add_non_spatial_bbox_dims(chunk_bbox_h, self.output_h, self.spatial_first)
        self.output_h[slicer(self.output_h, chunk_bbox_h)] = chunk_h.astype(self.output_h.dtype)

    def get_output(self, inplace: bool = False):
        self.executor.shutdown(wait=True)
        return self.output_h
