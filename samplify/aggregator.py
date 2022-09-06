import numpy as np
from samplify.sampler import GridSampler, _EdgeGridSampler
from samplify.slicer import slicer
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict
import copy
import concurrent.futures
from typing import Union, Optional, Tuple, Callable
import numpy.typing as npt


class Aggregator:
    def __init__(self, sampler: GridSampler, output_size: Optional[Union[Tuple, npt.ArrayLike]] = None, output: Optional[npt.ArrayLike] = None, weights: Union[str, Callable] = 'avg', softmax_dim: Optional[int] = None):
        self.aggregator = self.set_aggregator(sampler, output_size, output, weights, softmax_dim)

    def set_aggregator(self, sampler, output_size, output, weights, softmax_dim):
        mode = sampler.mode
        chunk_size = sampler.chunk_size
        if mode.startswith('sample_') and chunk_size is None:
            sampler = _Aggregator(spatial_size=sampler.spatial_size, patch_size=sampler.patch_size, output_size=output_size,
                                  output=output, weights=weights, spatial_first=sampler.spatial_first, softmax_dim=softmax_dim)
        elif mode.startswith('sample_') and chunk_size is not None:
            sampler = _ChunkAggregator(spatial_size=sampler.spatial_size, patch_size=sampler.patch_size, patch_overlap=sampler.patch_overlap, chunk_size=sampler.chunk_size,
                                       output_size=output_size, output=output, weights=weights, spatial_first=sampler.spatial_first, softmax_dim=softmax_dim)
        elif mode.startswith('pad_') and chunk_size is None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(mode))
        elif mode.startswith('pad_') and chunk_size is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(mode))
        else:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(mode))
        return sampler

    def append(self, patch, patch_indices, chunk_id=None):
        self.aggregator.append(patch, patch_indices, chunk_id)

    def get_output(self, inplace: bool = False):
        return self.aggregator.get_output(inplace)


class _Aggregator:
    def __init__(self, spatial_size: Union[Tuple, npt.ArrayLike], patch_size: Union[Tuple, npt.ArrayLike], output_size: Optional[Union[Tuple, npt.ArrayLike]] = None,
                 output: Optional[npt.ArrayLike] = None, weights: Union[str, Callable] = 'avg', spatial_first: str = True, softmax_dim: Optional[int] = None):
        """
        Aggregator to assemble an image with continuous content from patches. The content of overlapping patches is averaged.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param output: A numpy-style zero-initialized output (Numpy, Samplify.Subject, Tensor, Zarr, Dask, ...) of a continuous data type.
        If none then a zero-initialized Numpy output array of data type np.float32 is created internally.
        :param spatial_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        """
        self.spatial_size = np.asarray(spatial_size)
        self.patch_size = np.asarray(patch_size)
        self.output = self.set_output(output, output_size)
        self.spatial_first = spatial_first
        self.softmax_dim = softmax_dim
        self.weight_patch, self.weight_map = self.set_weights(weights)
        self.computed_inplace = False
        self.check_sanity()

    def set_output(self, output, output_size):
        if output is None and output_size is not None:
            output = np.zeros(output_size, dtype=np.float32)
            return output
        elif output is None and output_size is None:
            raise RuntimeError("Either the output array-like data or the output size must be given.")
        elif output is not None and output_size is not None and output.shape != output_size:
            raise RuntimeError("The variable output_size must be equal to the output shape if both are given. Only one of the two must be given.")
        return output

    def check_sanity(self):
        if np.any(self.patch_size > self.spatial_size):
            raise RuntimeError("The patch size ({}) cannot be larger than the spatial size ({}).".format(self.patch_size, self.spatial_size))
        # TODO: Check spatial_size == output.shape. Consider non-spatial dims and spatial_first
        # TODO: Check spatial_size == output_size. Consider non-spatial dims and spatial_first

    def set_weights(self, weights):
        if weights == 'avg':
            weight_patch = np.ones(self.patch_size, dtype=np.uint8)
        elif weights == 'gaussian':
            weight_patch = self.create_gaussian_weights(self.patch_size)
        elif callable(weights):
            weight_patch = weights
        else:
            raise RuntimeError("The given type of weights is not supported.")
        if self.softmax_dim is None:
            weight_map = np.zeros(self.spatial_size, dtype=np.float32)
        else:
            weight_map = None
        return weight_patch, weight_map

    def append(self, patch, patch_indices, chunk_id):
        """
        Appends a patch to the output.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_indices: The patch indices in the format of (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...).
        """
        if self.computed_inplace:
            raise RuntimeError("get_output() has already been called with inplace=True. Therefore, no further patches can be appended.")
        slices = self.add_non_spatial_dims(self.output, patch_indices)
        self.output[slicer(self.output, slices)] += patch.astype(self.output.dtype) * self.weight_patch.astype(self.output.dtype)
        if self.weight_map is not None:
            weight_map_patch = self.weight_map[slicer(self.weight_map, patch_indices)]  # weight_map_patch is only a reference to a patch in self.weight_map
            weight_map_patch[...] += self.weight_patch  # Adds patch to the map. [...] enables the use of memory-mapped array-like data

    def get_output(self, inplace: bool = False):
        """
        Computes and returns the final aggregated output based on all provided patches. The content of overlapping patches is averaged.
        :param inplace: Computes the output inplace without allocating new memory. Afterwards, no further patches can be appended.
        :return: The final aggregated output.
        """
        if not inplace:
            output = np.copy(self.output)
        else:
            output = self.output

        if not inplace or (inplace and not self.computed_inplace):
            if self.weight_map is not None:
                output[...] = output / self.weight_map.astype(output.dtype)
                output[...] = np.nan_to_num(output)
            if self.softmax_dim is not None:
                output[...] = output.argmax(axis=self.softmax_dim)
            if inplace:
                self.computed_inplace = True
        return output

    def create_gaussian_weights(self, size):
        sigma_scale = 1. / 8
        sigmas = size * sigma_scale
        center_coords = size // 2
        tmp = np.zeros(size)
        tmp[tuple(center_coords)] = 1
        gaussian_weights = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_weights[gaussian_weights == 0] = np.min(gaussian_weights[gaussian_weights != 0])
        return gaussian_weights

    def add_non_spatial_dims(self, output, patch_indices):
        non_spatial_dims = len(output.shape) - len(self.spatial_size)
        if self.spatial_first:
            slices = [None] * non_spatial_dims
            slices.extend([index_pair.tolist() for index_pair in patch_indices])
        else:
            slices = [index_pair.tolist() for index_pair in patch_indices]
            slices.extend([None] * non_spatial_dims)
        return slices


class _ChunkAggregator(_Aggregator):
    def __init__(self, spatial_size: Union[Tuple, npt.ArrayLike], patch_size: Union[Tuple, npt.ArrayLike], patch_overlap: Union[Tuple, npt.ArrayLike], chunk_size: Union[Tuple, npt.ArrayLike],
                 output_size: Optional[Union[Tuple, npt.ArrayLike]] = None, output: Optional[npt.ArrayLike] = None, weights: Union[str, Callable] = 'avg', spatial_first: str = True,
                 softmax_dim: Optional[int] = None, mode: str = 'sample_edge'):
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
        super().__init__(spatial_size=spatial_size, patch_size=patch_size, output_size=output_size, output=output, weights=weights, spatial_first=spatial_first, softmax_dim=softmax_dim)
        self.patch_overlap = patch_overlap
        self.chunk_size = chunk_size
        self.chunk_dtype = self.set_chunk_dtype()
        self.mode = mode
        self.compute_indices()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def set_chunk_dtype(self):
        if self.softmax_dim is None:
            return self.output.dtype
        else:
            return np.float32

    def compute_indices(self):
        self.grid_sampler = GridSampler(spatial_size=self.spatial_size, patch_size=self.chunk_size, patch_overlap=self.chunk_size - self.patch_size, mode=self.mode)
        self.chunk_sampler = []
        self.chunk_sampler_offset = []
        self.chunk_indices = list(self.grid_sampler)
        self.chunk_patches_dicts = defaultdict(dict)

        for chunk_id, chunk_indices in enumerate(self.chunk_indices):
            chunk_size = copy.copy(chunk_indices[:, 1] - chunk_indices[:, 0])
            sampler = _EdgeGridSampler(spatial_size=chunk_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap)
            sampler_offset = copy.copy(chunk_indices[:, 0])
            self.chunk_sampler.append(sampler)
            self.chunk_sampler_offset.append(sampler_offset)

            # The edges of non-border chunks need to be cropped as they have no overlap patch within the chunk
            for axis in range(len(self.spatial_size)):
                if 0 < chunk_indices[axis][0]:
                    chunk_indices[axis][0] += int(self.patch_size[axis] // 2)
                if chunk_indices[axis][1] < self.spatial_size[axis]:
                    chunk_indices[axis][1] -= int(self.patch_size[axis] // 2)

            for patch_indices in sampler:
                patch_indices_key = patch_indices + sampler_offset.reshape(-1, 1)
                patch_indices_key = patch_indices_key.astype(np.int64).tobytes()
                self.chunk_patches_dicts[chunk_id][patch_indices_key] = None

    def append(self, patch, patch_indices, chunk_id):
        """
        Appends a patch to the output.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_indices: The patch indices in the format of (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...).
        """
        if chunk_id is None:
            raise RuntimeError("The chunk ID must be given when the chunk size is set.")
        patch_indices_key = patch_indices.astype(np.int64).tobytes()
        if patch_indices_key not in self.chunk_patches_dicts[chunk_id]:
            unhashed_keys = [np.array(np.frombuffer(key, dtype=np.int64), dtype=int).reshape(-1, 2) for key in self.chunk_patches_dicts[chunk_id].keys()]
            raise RuntimeError("patch_indices_key not in self.chunk_patches_dicts[chunk_id]! patch_indices: {}. Offset for chunk_id {} is{}. unhashed_keys: {}".format(
                patch_indices, chunk_id, self.chunk_sampler_offset[chunk_id], unhashed_keys))
        self.chunk_patches_dicts[chunk_id][patch_indices_key] = patch
        if self.is_chunk_complete(chunk_id):
            # print("chunk_id: ", chunk_id)
            # self.process_chunk(chunk_id)
            self.executor.submit(self.process_chunk, chunk_id)

    def is_chunk_complete(self, chunk_id):
        # Check if all self.chunk_patches_dicts[chunk_id] values are not None
        for value in self.chunk_patches_dicts[chunk_id].values():
            if value is None:
                return False
        return True

    def process_chunk(self, chunk_id):
        # If they are all not None, create a softmax array of size chunk_size with number classes as channels
        patch_shape = self.chunk_patches_dicts[chunk_id][list(self.chunk_patches_dicts[chunk_id].keys())[0]].shape
        chunk_size = self.add_non_spatial_chunk_dims(self.chunk_size, patch_shape)
        chunk = np.zeros(chunk_size, dtype=self.chunk_dtype)
        # Weight each patch during insertion
        sampler_offset = self.chunk_sampler_offset[chunk_id].reshape(-1, 1)
        for patch_indices_key, patch in self.chunk_patches_dicts[chunk_id].items():
            patch_indices = np.array(np.frombuffer(patch_indices_key, dtype=np.int64), dtype=int).reshape(-1, 2)
            patch_indices -= sampler_offset
            slices = self.add_non_spatial_dims(chunk, patch_indices)
            chunk[slicer(chunk, slices)] += patch.astype(chunk.dtype) * self.weight_patch.astype(chunk.dtype)
        if self.softmax_dim is not None:
            # Argmax the softmax chunk
            chunk = chunk.argmax(axis=self.softmax_dim).astype(np.uint16)
        # Crop the chunk
        crop_indices = self.chunk_indices[chunk_id] - sampler_offset
        crop_indices = self.add_non_spatial_dims(chunk, crop_indices)
        chunk = chunk[slicer(chunk, crop_indices)]
        # Write the chunk into the global output
        crop_indices = self.chunk_indices[chunk_id]
        crop_indices = self.add_non_spatial_dims(self.output, crop_indices)
        self.output[slicer(self.output, crop_indices)] = chunk
        # Set all self.chunk_patches_dicts[chunk_id] values to None
        for key in self.chunk_patches_dicts[chunk_id].keys():
            self.chunk_patches_dicts[chunk_id][key] = None
        # print("Finished saving chunk ", chunk_id, flush=True)

    def add_non_spatial_chunk_dims(self, spatial_chunk_size, patch_shape):
        non_spatial_dims = len(patch_shape) - len(self.spatial_size)
        if non_spatial_dims > 0 and self.spatial_first:
            non_spatial_dims = patch_shape[self.spatial_size:]
            chunk_size = (*spatial_chunk_size, *non_spatial_dims)
            return chunk_size
        elif non_spatial_dims > 0:
            non_spatial_dims = patch_shape[:self.spatial_size]
            chunk_size = (*non_spatial_dims, *spatial_chunk_size)
            return chunk_size
        else:
            return spatial_chunk_size

    def get_output(self, inplace: bool = False):
        return self.output


# class ResizeChunkedWeightedSoftmaxAggregator(_ChunkAggregator):
#     def __init__(self, output=None, spatial_size=None, patch_size=None, patch_overlap=None, chunk_size=None, weights='gaussian', spacing=None):
#         """
#         Weighted aggregator to assemble an image with continuous content from patches. Returns the maximum class at each position of the image. The content of overlapping patches is gaussian-weighted by default.
#         Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
#         Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
#         :param output: A numpy-style zero-initialized image (Numpy, Zarr, Dask, ...) of a continuous data type. If none then a zero-initialized Numpy image of data type np.float32 is created internally.
#         :param spatial_size: The image size that was used for patchification without batch and channel dimensions. Always required.
#         :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
#         :param weights: A weight map of size patch_size that should be used for weighting the importance of each value in a patch. Default is a gaussian weight map.
#         :param low_memory_mode: Reduces memory consumption by more than 50% in comparison to the normal WeightedAggregator and Aggregator. However, the prediction quality is slightly reduced.
#         """
#         super().__init__(output, spatial_size, patch_size, patch_overlap, chunk_size, weights)
#         self.patch_overlap = patch_overlap
#         self.chunk_size = np.asarray(chunk_size)
#         self.spacing = spacing
#         # self.source_size = np.asarray(source_size)
#         self.compute_indices()
#         # self.set_weights(self.source_size, weights)
#         # self.size_conversion_factor = size_conversion_factor
#
#     def process_chunk(self, chunk_id):
#         # print("chunk_id: ", chunk_id)
#         # If they are all not None, create a softmax array of size chunk_size with number classes as channels
#         patch_shape = self.chunk_patches_dicts[chunk_id][list(self.chunk_patches_dicts[chunk_id].keys())[0]].shape
#         num_channels = patch_shape[0]
#         # size_conversion_factor = (patch_shape[1:] / self.patch_size)[0]
#         # resized_chunk_size = np.rint(self.chunk_size * self.size_conversion_factor).astype(np.int32)
#         image_chunk_softmax = np.zeros((num_channels, *self.chunk_size), dtype=np.float32)
#         # Weight each patch during insertion
#         sampler_offset = self.chunk_sampler_offset[chunk_id].reshape(-1, 1)
#         for patch_indices_key, patch in self.chunk_patches_dicts[chunk_id].items():
#             patch_indices = np.array(np.frombuffer(patch_indices_key, dtype=np.int64), dtype=int).reshape(-1, 2)
#             patch_indices -= sampler_offset
#             # patch_indices = np.rint(patch_indices * size_conversion_factor).astype(np.int32)
#             slices = self.add_non_spatial_dims(image_chunk_softmax, patch_indices)
#             image_chunk_softmax[slicer(image_chunk_softmax, slices)] += patch.astype(image_chunk_softmax.dtype) * self.weight_patch.astype(image_chunk_softmax.dtype)
#         # Argmax the softmax chunk
#         image_chunk = image_chunk_softmax.argmax(axis=0).astype(np.uint16)
#         # image_chunk = self.border_core2instance(image_chunk)
#         # image_chunk = ski_transform.resize(image_chunk, output_shape=self.chunk_size, order=0)
#         # save_nifti("/home/k539i/Documents/datasets/preprocessed/my_framework/tmp/tmp_1_{}.nii.gz".format(chunk_id), image_chunk)  # TODO: Remove
#         # Crop the chunk
#         crop_indices = self.chunk_indices[chunk_id] - sampler_offset
#         image_chunk = image_chunk[slicer(image_chunk, crop_indices)]
#         # save_nifti("/home/k539i/Documents/datasets/preprocessed/my_framework/tmp/tmp_2_{}.nii.gz".format(chunk_id), image_chunk)  # TODO: Remove
#
#         # image_chunk[:, :, 0] = 10  # TODO: Remove
#         # image_chunk[:, :, -1] = 10  # TODO: Remove
#         # image_chunk[:, 0, :] = 10  # TODO: Remove
#         # image_chunk[:, -1, :] = 10  # TODO: Remove
#         # image_chunk[0, :, :] = 10  # TODO: Remove
#         # image_chunk[-1, :, :] = 10  # TODO: Remove
#
#         # Write the chunk into the global image
#         crop_indices = self.chunk_indices[chunk_id]
#         # print("crop_indices 2: ", crop_indices)
#         self.image[slicer(self.image, crop_indices)] = image_chunk
#         # Set all self.chunk_patches_dicts[chunk_id] values to None
#         for key in self.chunk_patches_dicts[chunk_id].keys():
#             self.chunk_patches_dicts[chunk_id][key] = None
#
#     # def border_core2instance(self, image_chunk):
#     #     image_chunk = border_semantic2instance_patchify(image_chunk, self.spacing)
#     #     return image_chunk



if __name__ == '__main__':
    from sampler import _ChunkGridSampler
    import zarr
    from tqdm import tqdm

    image_size = (500, 500, 500)
    patch_size = (128, 128, 128)
    patch_overlap = (64, 64, 64)
    chunk_size = (384, 384, 384)

    # image = np.random.uniform(size=(3, *image_size))
    # image = np.random.randint(low=0, high=255, size=(3, *image_size), dtype=np.uint8)
    # result = np.zeros(image_size, dtype=np.uint8)
    result = zarr.open("tmp.zarr", mode='w', shape=image_size, chunks=chunk_size, dtype=np.uint8)

    grid_sampler = _ChunkGridSampler(spatial_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size)
    aggregrator = _ChunkAggregator(output=result, spatial_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size)

    print(len(grid_sampler))

    counter = 0
    for i, indices in enumerate(tqdm(grid_sampler)):
        # print("Iteration: {}, indices: {}".format(i, indices))
        patch = np.zeros((8, *patch_size), dtype=np.float32)
        chunk_id = indices[1]
        patch[chunk_id, ...] = 1
        aggregrator.append(patch, indices)
    print("")