import numpy as np
from patchly.sampler import GridSampler, _AdaptiveGridSampler
from patchly.slicer import slicer
from patchly import utils
from patchly.sampler import SamplingMode
from patchly.array_like import create_array_like, ArrayLike
from collections import defaultdict
import concurrent.futures
from typing import Union, Optional, Tuple, Callable
import numpy.typing as npt
from enum import Enum
import warnings

try:
    import zarr
except:
    zarr = None


class PatchStatus(Enum):
    EMPTY = 1
    FILLED = 2
    COMPLETED = 3


class Aggregator:
    def __init__(self, sampler: GridSampler, output_size: Optional[Union[Tuple, npt.ArrayLike]] = None, output: Optional[npt.ArrayLike] = None, chunk_size: Optional[Union[Tuple, npt.ArrayLike]] = None, 
                 weights: Union[str, Callable] = 'avg', softmax_dim: Optional[int] = None, has_batch_dim: bool = False, spatial_first: bool = True, device = 'cpu'):
        self.sampler = sampler
        self.image_size_s = sampler.image_size_s
        self.patch_size_s = sampler.patch_size_s
        self.step_size_s = sampler.step_size_s
        self.chunk_size_s = chunk_size
        self.spatial_first = spatial_first
        self.mode = sampler.mode
        self.softmax_dim = softmax_dim
        self.has_batch_dim = has_batch_dim
        self.device = device
        self.array_type = self.set_array_type(output)
        self.output_h = self.set_output(output, output_size)
        self.weight_patch_s, self.weight_map_s = self.set_weights(weights)
        self.check_sanity()
        self.aggregator = self.set_aggregator(self.sampler, self.output_h, self.softmax_dim)

    def set_array_type(self, output):
        if output is not None:
            array_type = type(output)
        else:
            array_type = None
        return array_type

    def set_output(self, output_h, output_size_h):
        if output_h is None and output_size_h is not None:
            output_h = create_array_like(self.array_type, None, self.device).create_zeros(output_size_h, "float32")
        elif output_h is not None and output_size_h is None:
            output_h = create_array_like(self.array_type, output_h, self.device)
        elif output_h is None and output_size_h is None:
            raise RuntimeError("Either the output array-like data or the output size must be given.")
        elif output_h is not None and output_size_h is not None and output_h.shape != output_size_h:
            raise RuntimeError("The variable output_size must be equal to the output shape if both are given. Only one of the two must be given.")
        return output_h

    def set_weights(self, weights_s):
        if weights_s == 'avg':
            weight_patch_s = create_array_like(self.array_type, None, self.device).create_ones(self.patch_size_s, "uint8")
        elif self.mode == SamplingMode.SAMPLE_ADAPTIVE and weights_s == 'gaussian':
            # The gaussian kernel would be shifted if the weight_patch is cropped due to adaptive mode
            raise RuntimeError("Adaptive sampling cannot be used with gaussian weighting. Use a different sampler mode.")
        elif weights_s == 'gaussian':
            weight_patch_s = create_array_like(self.array_type, None, self.device).create_gaussian_kernel(self.patch_size_s, dtype="float32")
        elif hasattr(self.output_h, '__getitem__'):
            weight_patch_s = weights_s
        elif callable(weights_s):
            weight_patch_s = weights_s
        else:
            raise RuntimeError("The given type of weights is not supported.")

        if self.softmax_dim is None:
            if self.chunk_size_s is None:
                weight_map_size_s = self.image_size_s
            else:
                weight_map_size_s = self.chunk_size_s
            if weights_s == 'avg':
                # uint8 saves memory, but might be problematic when using a very small patch offset
                # Consider providing your own uint16 weight map when working with a very small patch offset
                weight_map_s = create_array_like(self.array_type, None, self.device).create_zeros(weight_map_size_s, "uint8")
            else:
                weight_map_s = create_array_like(self.array_type, None, self.device).create_zeros(weight_map_size_s, "float32")
        else:
            weight_map_s = None
        return weight_patch_s, weight_map_s

    def check_sanity(self):
        if not hasattr(self.output_h, '__getitem__'):
            raise RuntimeError("The given output is not ArrayLike.")
        if self.spatial_first and (self.output_h.shape[:len(self.image_size_s)] != tuple(self.image_size_s)):
            raise RuntimeError("The spatial size of the given output {} is unequal to the given spatial size {}.".format(self.output_h.shape[:len(self.image_size_s)], self.image_size_s))
        if (not self.spatial_first) and (self.output_h.shape[-len(self.image_size_s):] != tuple(self.image_size_s)):
            raise RuntimeError("The spatial size of the given output {} is unequal to the given spatial size {}.".format(self.output_h.shape[-len(self.image_size_s):], self.image_size_s))
        if self.chunk_size_s is not None and np.any(self.chunk_size_s > self.image_size_s):
            raise RuntimeError("The chunk size ({}) cannot be greater than the spatial size ({}) in one or more dimensions.".format(self.chunk_size_s, self.image_size_s))
        if self.chunk_size_s is not None and np.any(self.patch_size_s >= self.chunk_size_s):
            raise RuntimeError("The patch size ({}) cannot be greater or equal to the chunk size ({}) in one or more dimensions.".format(self.patch_size_s, self.chunk_size_s))
        if self.chunk_size_s is not None and len(self.image_size_s) != len(self.chunk_size_s):
            raise RuntimeError("The dimensionality of the chunk size ({}) is required to be the same as the spatial size ({}).".format(self.chunk_size_s, self.image_size_s))
        if self.has_batch_dim and self.spatial_first:
            raise RuntimeError("The arguments has_batch_dim and spatial_first cannot both be true at the same time.")
        if self.mode.name.startswith('PAD_') and self.chunk_size_s is not None:
            raise RuntimeError("The given sampling mode ({}) is not compatible with chunk sampling.".format(self.mode))

    def set_aggregator(self, sampler, output_h, softmax_dim):
        if self.mode.name.startswith('SAMPLE_') and self.chunk_size_s is None:
            aggregator = _Aggregator(sampler=sampler, image_size_s=self.image_size_s, patch_size_s=self.patch_size_s,
                                  output_h=output_h, spatial_first=self.spatial_first, softmax_dim=softmax_dim, has_batch_dim=self.has_batch_dim, 
                                  weight_patch_s=self.weight_patch_s, weight_map_s=self.weight_map_s, device=self.device, array_type=self.array_type)
        elif self.mode.name.startswith('SAMPLE_') and self.chunk_size_s is not None:
            aggregator = _ChunkAggregator(sampler=sampler, image_size_s=self.image_size_s, patch_size_s=self.patch_size_s, step_size_s=self.step_size_s,
                                       chunk_size_s=self.chunk_size_s, output_h=output_h, spatial_first=self.spatial_first, softmax_dim=softmax_dim, 
                                       has_batch_dim=self.has_batch_dim, weight_patch_s=self.weight_patch_s, device=self.device, array_type=self.array_type)
        elif self.mode.name.startswith('PAD_') and self.chunk_size_s is None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        elif self.mode.name.startswith('PAD_') and self.chunk_size_s is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        else:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
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
    def __init__(self, sampler: GridSampler, image_size_s: Union[Tuple, npt.ArrayLike], patch_size_s: Union[Tuple, npt.ArrayLike],
                 output_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True, softmax_dim: Optional[int] = None, has_batch_dim: bool = False,
                 weight_patch_s: npt.ArrayLike = None, weight_map_s: npt.ArrayLike = None, device = 'cpu', array_type = None):
        """
        Aggregator to assemble an image with continuous content from patches. The content of overlapping patches is averaged.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param output: A numpy-style zero-initialized output (Numpy, Tensor, Zarr, Dask, ...) of a continuous data type.
        If none then a zero-initialized Numpy output array of data type np.float32 is created internally.
        :param spatial_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        """
        self.sampler = sampler
        self.image_size_h = None
        self.image_size_n = None
        self.image_size_s = np.asarray(image_size_s)
        self.patch_size_s = np.asarray(patch_size_s)
        self.output_h = output_h
        self.spatial_first = spatial_first
        self.softmax_dim = softmax_dim
        self.has_batch_dim = has_batch_dim
        self.weight_patch_s = weight_patch_s
        self.weight_patch_h = None
        self.weight_map_s = weight_map_s
        self.computed_inplace = False
        self.device = device
        self.array_type = array_type

    def append(self, patch_h, patch_bbox_s):
        """
        Appends a patch to the output.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_bbox: The patch bbox in the format of (w_start, w_end, h_start, h_end, d_start, d_end, ...).
        """

        if not isinstance(patch_h, ArrayLike):
            patch_h = create_array_like(type(patch_h), patch_h, self.device)

        # Check if out put was already computed inplace once.
        if self.computed_inplace:
            raise RuntimeError("get_output() has already been called with inplace=True. Therefore, no further patches can be appended.")
        
        # Determine holistic and non-spatial image shape
        if self.image_size_h is None:
            self.determine_image_sizes(patch_h)

        # If the patch is a batch input then each patch in the batch must be processed individually
        if self.has_batch_dim and len(patch_h.shape) == len(self.image_size_h) + 1:
            for batch_idx in range(patch_h.shape[0]):
                self.append(patch_h[batch_idx], patch_bbox_s[batch_idx])

        # Add a patch
        else:
            # Create holistic weight patch based on spatial weight patch
            if self.weight_patch_h is None:
                self.weight_patch_h = utils.broadcast_to(self.weight_patch_s, utils.data_s_to_data_h(self.weight_patch_s.shape, patch_h.shape, self.spatial_first), self.spatial_first)

            # Verify and correct the array types of specific ArrayLikes as their true array type can only be determined based on an actual patch
            self.verify_array_types(type(patch_h.data))

            # Create holistic bboxes based on spatial bbox
            patch_bbox_h = utils.bbox_s_to_bbox_h(patch_bbox_s, self.output_h, self.spatial_first)
            weight_patch_bbox_s = np.asarray([[0, patch_bbox_s[axis][1] - patch_bbox_s[axis][0]]  for axis in range(len(self.image_size_s))])
            weight_patch_bbox_h = utils.bbox_s_to_bbox_h(weight_patch_bbox_s, self.output_h, self.spatial_first)
            
            # Add patch to output with weight patch
            # self.output_h[slicer(self.output_h, patch_bbox_h)] += patch_h.astype(self.output_h.dtype) * self.weight_patch_h.astype(self.output_h.dtype)
            self.output_h[slicer(self.output_h, patch_bbox_h)] += patch_h.astype(self.output_h.dtype) * self.weight_patch_h[slicer(self.weight_patch_h, weight_patch_bbox_h)].astype(self.output_h.dtype)

            # Add weight patch to weight map
            if self.weight_map_s is not None:
                self.weight_map_s[slicer(self.weight_map_s, patch_bbox_s)] += self.weight_patch_s[slicer(self.weight_patch_s, weight_patch_bbox_s)]

    def get_output(self, inplace: bool = False):
        """
        Computes and returns the final aggregated output based on all provided patches. The content of overlapping patches is averaged.
        :param inplace: Computes the output inplace without allocating new memory. Afterwards, no further patches can be appended.
        :return: The final aggregated output.
        """
        if not inplace:
            output_h = self.output_h.copy()
        else:
            output_h = self.output_h

        if not inplace or (inplace and not self.computed_inplace):
            if self.weight_map_s is not None:
                weight_map_h = utils.broadcast_to(self.weight_map_s, utils.data_s_to_data_h(self.weight_map_s.shape, output_h.shape, self.spatial_first), self.spatial_first)
                output_h = output_h / weight_map_h.astype(output_h.dtype)
                output_h = output_h.nan_to_num()
            if self.softmax_dim is not None:
                # Cannot be done inplace -> No [...]
                output_h = output_h.argmax(axis=self.softmax_dim)
            if inplace:
                self.computed_inplace = True
        return output_h.data
    
    def determine_image_sizes(self, patch_h):
        self.image_size_h = patch_h.shape
        if self.has_batch_dim:
            self.image_size_h = self.image_size_h[1:]
        self.image_size_h = np.concatenate((self.image_size_s, self.image_size_h[len(self.image_size_s):])) if self.spatial_first else np.concatenate((self.image_size_h[:-len(self.image_size_s)], self.image_size_s))
        self.image_size_n = self.image_size_h[len(self.image_size_s):] if self.spatial_first else self.image_size_h[:-len(self.image_size_s)]

    def verify_array_types(self, array_type):
        if self.array_type is None:
            # Only change output_h if only an output_size_h was given during intialization
            self.output_h = create_array_like(array_type, self.output_h.data, self.device)

        if self.weight_patch_s is not None and array_type != self.weight_patch_s:
            self.weight_patch_s = create_array_like(array_type, self.weight_patch_s.data, self.device)            
            
        if self.weight_patch_h is not None and array_type!= self.weight_patch_h:
            self.weight_patch_h = create_array_like(array_type, self.weight_patch_h.data, self.device)

        if self.weight_map_s is not None and array_type != self.weight_map_s:
            self.weight_map_s = create_array_like(array_type, self.weight_map_s.data, self.device)

        self.array_type = array_type


class _ChunkAggregator(_Aggregator):
    def __init__(self, sampler: GridSampler, image_size_s: Union[Tuple, npt.ArrayLike], patch_size_s: Union[Tuple, npt.ArrayLike], step_size_s: Union[Tuple, npt.ArrayLike], chunk_size_s: Union[Tuple, npt.ArrayLike],
                 output_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True,
                 softmax_dim: Optional[int] = None, has_batch_dim: bool = False, weight_patch_s: npt.ArrayLike = None, device = 'cpu', array_type = None):
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
        super().__init__(sampler=sampler, image_size_s=image_size_s, patch_size_s=patch_size_s, output_h=output_h, spatial_first=spatial_first, 
                         softmax_dim=softmax_dim, has_batch_dim=has_batch_dim, weight_patch_s=weight_patch_s, weight_map_s=None, device=device, array_type=array_type)
        self.step_size_s = step_size_s
        self.chunk_size_s = chunk_size_s
        self.chunk_dtype = self.set_chunk_dtype()
        self.chunk_sampler, self.chunk_patch_dict, self.patch_chunk_dict = self.compute_patches()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def set_chunk_dtype(self):
        if self.softmax_dim is None:
            return self.output_h.dtype
        else:
            return np.float32

    def compute_patches(self):
        patch_sampler = self.sampler
        chunk_sampler = _AdaptiveGridSampler(image_size_s=self.image_size_s, patch_size_s=self.chunk_size_s, step_size_s=self.chunk_size_s)
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

    def append(self, patch_h, patch_bbox_s):
        """
        Appends a patch to the output.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_bbox: The patch bbox in the format of (w_start, w_end, h_start, h_end, d_start, d_end, ...).
        """

        if not isinstance(patch_h, ArrayLike):
            patch_h = create_array_like(type(patch_h), patch_h, self.device)

        # Determine holistic and non-spatial image shape
        if self.image_size_h is None:
            self.determine_image_sizes(patch_h)

        # If the patch is a batch input then each patch in the batch must be processed individually
        if self.has_batch_dim and len(patch_h.shape) == len(self.image_size_h) + 1:
            for batch_idx in range(patch_h.shape[0]):
                self.append(patch_h[batch_idx], patch_bbox_s[batch_idx])

        # Add a patch
        else:
            # Create holistic weight patch based on spatial weight patch
            if self.weight_patch_h is None:
                self.weight_patch_h = utils.broadcast_to(self.weight_patch_s, utils.data_s_to_data_h(self.weight_patch_s.shape, patch_h.shape, self.spatial_first), self.spatial_first)

            # Verify and correct the array types of specific ArrayLikes as their true array type can only be determined based on an actual patch
            self.verify_array_types(type(patch_h.data))

            # Add patch
            self.patch_chunk_dict[str(np.asarray(patch_bbox_s))]["patch"].create(patch_h)

            # Check on process finished chunks
            for chunk_id in self.patch_chunk_dict[str(np.asarray(patch_bbox_s))]["chunks"]:
                self.chunk_patch_dict[chunk_id][str(np.asarray(patch_bbox_s))]["status"] = PatchStatus.FILLED
                if self.is_chunk_complete(chunk_id):
                    if zarr is not None and isinstance(self.output_h.data, zarr.core.Array):
                        warnings.warn("Ouput is a Zarr array. Switching to single threading for chunk processing. See issue #39 for more information.") # If issue is solved remove zarr and warnings import statements
                        self.process_chunk(chunk_id)
                    else:
                        self.executor.submit(self.process_chunk, chunk_id)
                        # self.process_chunk(chunk_id)

    def is_chunk_complete(self, chunk_id):
        for patch_data in self.chunk_patch_dict[chunk_id].values():
            if patch_data["status"] != PatchStatus.FILLED:
                return False
        return True

    def process_chunk(self, chunk_id):
        patch_size_h = list(self.chunk_patch_dict[chunk_id].values())[0]["patch"].shape
        chunk_size_s = self.chunk_sampler.patch_sizes_s[chunk_id]
        chunk_size_h = utils.data_s_to_data_h(chunk_size_s, patch_size_h, self.spatial_first)
        chunk_h = create_array_like(self.array_type, None, self.device).create_zeros(chunk_size_h, self.chunk_dtype)
        if self.softmax_dim is None:
            weight_map_h = create_array_like(self.array_type, None, self.device).create_zeros(chunk_size_h)
        for patch_data in self.chunk_patch_dict[chunk_id].values():
            valid_patch_bbox_s = patch_data["valid_patch_bbox"]
            crop_patch_bbox_s = patch_data["crop_patch_bbox"]
            patch_h = patch_data["patch"].data
            patch_data["patch"] = None
            patch_data["status"] = PatchStatus.COMPLETED
            crop_patch_bbox_h = utils.bbox_s_to_bbox_h(crop_patch_bbox_s, chunk_h, self.spatial_first)
            valid_patch_bbox_h = utils.bbox_s_to_bbox_h(valid_patch_bbox_s, chunk_h, self.spatial_first)
            valid_patch_h = patch_h[slicer(patch_h, crop_patch_bbox_h)]
            valid_weight_patch_h = self.weight_patch_h[slicer(self.weight_patch_h, crop_patch_bbox_h)]
            chunk_h[slicer(chunk_h, valid_patch_bbox_h)] += valid_patch_h.astype(chunk_h.dtype) * valid_weight_patch_h.astype(chunk_h.dtype)
            if self.softmax_dim is None:
                weight_map_h[slicer(weight_map_h, valid_patch_bbox_h)] += valid_weight_patch_h
        if self.softmax_dim is None:
            chunk_h = chunk_h / weight_map_h.astype(chunk_h.dtype)
            chunk_h = chunk_h.nan_to_num()
        else:
            # Argmax the softmax chunk
            chunk_h = chunk_h.argmax(axis=self.softmax_dim).astype("uint16")
        chunk_bbox_h = self.chunk_sampler.__getitem__(chunk_id)
        chunk_bbox_h = utils.bbox_s_to_bbox_h(chunk_bbox_h, self.output_h, self.spatial_first)
        self.output_h[slicer(self.output_h, chunk_bbox_h)] = chunk_h.astype(self.output_h.dtype)

    def get_output(self, inplace: bool = False):
        self.executor.shutdown(wait=True)
        return self.output_h.data
