import numpy as np
from samplify.slicer import slicer
from torch.utils.data import Dataset
from skimage import transform as ski_transform
import copy
import random
import augmentify as aug
from typing import Union, Optional, Tuple
import numpy.typing as npt


class GridSampler:
    def __init__(self, spatial_size: Union[Tuple, npt.ArrayLike], patch_size: Union[Tuple, npt.ArrayLike], patch_overlap: Optional[Union[Tuple, npt.ArrayLike]] = None,
                 chunk_size: Optional[Union[Tuple, npt.ArrayLike]] = None, image: Optional[npt.ArrayLike] = None, spatial_first: bool = True, mode: str = "sample_edge"):
        """
        TODO description
        If no image is given then only patch indices (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...) are returned instead.

        :param image: The image in an array-like format (Numpy, Samplify.Subject, Zarr, Dask, ...) that can be memory-mapped.
        Samplify.subject is directly supported and will sample the same patch from all stored images.
        The image can have an arbitrary number of additional non-spatial dimensions.
        :param spatial_size: The spatial shape of the image. The spatial shape excludes the channel, batch and any other non-spatial dimensionality.
        :param patch_size: The spatial shape of the patch. The patch shape excludes the channel, batch and any other non-spatial dimensionality.
        :param patch_overlap: The spatial shape of the patch overlap. If None then the patch overlap is equal to the patch size.
        The patch overlap excludes the channel, batch and any other non-spatial dimensionality.
        :param chunk_size: The spatial shape of the chunk size. If given, the image is divided into chunks and patches are sampled from a single chunk until the chunk is depleted.
        This enables patch sampling of larger-than-RAM images in combination with memory-mapped arrays. The chunk size is required to be a multiple of the patch size.
        The chunk size excludes the channel, batch and any other non-spatial dimensionality. An in-depth explanation of chunked sampling can be found here: LINK
        :param spatial_first: Denotes that the spatial dimensions of the image are located before the non-spatial dimensions e.g. (Width, Height, Channel).
        Otherwise, the reverse is true e.g. (Channel, Width, Height).
        :param mode: TODO
        """
        self.image = image
        self.spatial_size = self.set_parameter(spatial_size)
        self.patch_size = self.set_parameter(patch_size)
        self.patch_overlap = self.set_patch_overlap(patch_overlap, patch_size)
        self.chunk_size = self.set_parameter(chunk_size)
        self.spatial_first = spatial_first
        self.mode = mode
        self.check_sanity()
        self.sampler = self.create_sampler()

    def set_parameter(self, parameter):
        if parameter is None:
            parameter = None
        else:
            parameter = np.asarray(parameter)
        return parameter

    def set_patch_overlap(self, patch_overlap, patch_size):
        if patch_overlap is None:
            patch_overlap = patch_size
        else:
            patch_overlap = np.asarray(patch_overlap)
        return patch_overlap

    def create_sampler(self):
        if self.mode == "sample_edge" and self.chunk_size is None:
            sampler = _EdgeGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, spatial_first=not self.spatial_first)
        elif self.mode == "sample_edge" and self.chunk_size is not None:
            sampler = _ChunkedGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, chunk_size=self.chunk_size,
                                          spatial_first=not self.spatial_first)
        elif self.mode == "sample_adaptive" and self.chunk_size is None:
            sampler = _AdaptiveGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, spatial_first=not self.spatial_first)
        elif self.mode == "sample_adaptive" and self.chunk_size is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        elif self.mode == "sample_crop" and self.chunk_size is None:
            sampler = _CropGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, spatial_first=not self.spatial_first)
        elif self.mode == "sample_crop" and self.chunk_size is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        elif self.mode.startswith('pad_') and self.chunk_size is None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        elif self.mode.startswith('pad_') and self.chunk_size is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        else:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        return sampler

    def check_sanity(self):
        if np.any(self.patch_size > self.spatial_size):
            raise RuntimeError("The patch size ({}) cannot be larger than the spatial size ({}).".format(self.patch_size, self.spatial_size))
        if self.patch_overlap is not None and np.any(self.patch_overlap > self.patch_size):
            raise RuntimeError("The patch overlap ({}) cannot be larger than the patch size ({}).".format(self.patch_overlap, self.patch_size))
        if self.chunk_size is not None and np.any(self.chunk_size > self.spatial_size):
            raise RuntimeError("The chunk size ({}) cannot be larger than the spatial size ({}).".format(self.chunk_size, self.spatial_size))
        if self.chunk_size is not None and np.any(self.patch_size > self.chunk_size):
            raise RuntimeError("The patch size ({}) cannot be larger than the chunk size ({}).".format(self.patch_size, self.chunk_size))
        if len(self.spatial_size) != len(self.patch_size):
            raise RuntimeError("The dimensionality of the patch size ({}) is required to be the same as the spatial size ({})."
                               .format(self.patch_size, self.spatial_size))
        if self.patch_overlap is not None and len(self.spatial_size) != len(self.patch_overlap):
            raise RuntimeError("The dimensionality of the patch overlap ({}) is required to be the same as the spatial size ({})."
                               .format(self.patch_overlap, self.spatial_size))
        if self.chunk_size is not None and len(self.spatial_size) != len(self.chunk_size):
            raise RuntimeError("The dimensionality of the chunk size ({}) is required to be the same as the spatial size ({})."
                               .format(self.chunk_size, self.spatial_size))
        if self.patch_overlap is not None and self.chunk_size is not None and (self.patch_size % self.patch_overlap != 0).any():
            raise RuntimeError("The patch size ({}) is required to be a multiple of the patch overlap ({}) when using chunked images.".format(self.patch_size, self.patch_overlap))
        if self.chunk_size is not None and (self.chunk_size % self.patch_size != 0).any():
            raise RuntimeError("The chunk size ({}) is required to be a multiple of the patch size ({}).".format(self.chunk_size, self.patch_size))

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self):
        return self.sampler.__len__()

    def __getitem__(self, idx):
        return self.sampler.__getitem__(idx)

    def __next__(self):
        return self.sampler.__next__()


class _CropGridSampler:
    def __init__(self, spatial_size, patch_size, patch_overlap=None, image=None, spatial_first=True):
        """
        TODO Redo doc

        An N-dimensional grid sampler that should mainly be used for inference. The image is divided into a grid with each grid cell having the size of patch_size. The grid can have overlap if patch_overlap is specified.
        If patch_size is not a multiple of image_size then the remainder part of the image is not sampled.
        The grid sampler only returns image patches if image is set.
        Otherwise, only the patch indices w_ini, w_fin, h_ini, h_fin, d_ini, d_fin are returned. They can be used to extract the patch from the image like this:
        img = img[w_ini:w_fin, h_ini:h_fin, d_ini:d_fin] (Example for a 3D image)
        Requiring only size parameters instead of the actual image makes the grid sampler file format independent if desired.

        :param image: The image data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions. Can also be a dict of multiple images.
        If None then patch indices (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...) are returned instead.
        :param spatial_size: The shape of the image without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param patch_overlap: The shape of the patch overlap without batch and channel dimensions. If None then the patch overlap is equal to patch_size.
        """
        self.image = image
        self.spatial_size = spatial_size
        self.spatial_first = spatial_first
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.indices = self.compute_indices()

    def compute_indices(self):
        n_axis = len(self.spatial_size)
        stop = [self.spatial_size[axis] - self.patch_size[axis] + 1 for axis in range(n_axis)]
        axis_indices = [np.arange(0, stop[axis], self.patch_overlap[axis]) for axis in range(n_axis)]
        axis_indices = np.meshgrid(*axis_indices, indexing='ij')
        indices = np.column_stack([axis_indices[axis].ravel() for axis in range(n_axis)])
        return indices

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[idx]
        patch_indices = np.zeros(len(indices) * 2, dtype=int).reshape(-1, 2)
        for axis in range(len(indices)):
            patch_indices[axis][0] = indices[axis]
            patch_indices[axis][1] = indices[axis] + self.patch_size[axis]
        slice_result = self.slice_patch(patch_indices)
        return slice_result

    def __next__(self):
        if self.index < self.__len__():
            output = self.__getitem__(self.index)
            self.index += 1
            return output
        else:
            raise StopIteration

    def slice_patch(self, patch_indices):
        if self.image is not None and not isinstance(self.image, dict):
            slices = self.get_slices(self.image, patch_indices)
            patch = self.image[slicer(self.image, slices)]
            return patch, patch_indices
        elif self.image is not None and isinstance(self.image, dict):
            patch_dict = {}
            for key in self.image.keys():
                slices = self.get_slices(self.image[key], patch_indices)
                patch_dict[key] = self.image[key][slicer(self.image[key], slices)]
            return patch_dict, patch_indices
        else:
            return patch_indices

    def get_slices(self, image, patch_indices):
        non_spatial_dims = len(image.shape) - len(self.spatial_size)
        if self.spatial_first:
            slices = [None] * non_spatial_dims
            slices.extend([index_pair.tolist() for index_pair in patch_indices])
        else:
            slices = [index_pair.tolist() for index_pair in patch_indices]
            slices.extend([None] * non_spatial_dims)
        return slices


class _EdgeGridSampler(_CropGridSampler):
    def __init__(self, spatial_size, patch_size, patch_overlap=None, image=None, spatial_first=True):
        """
        TODO Redo doc

        An N-dimensional grid sampler that should mainly be used for inference. The image is divided into a grid with each grid cell having the size of patch_size. The grid can have overlap if patch_overlap is specified.
        If patch_size is not a multiple of image_size then the remainder part of the image is not padded, but instead patches are sampled at the edge of the image of size patch_size like this:
        ----------------------
        |                | X |
        |                | X |
        |                | X |
        |                | X |
        |----------------| X |
        |X  X  X  X  X  X  X |
        ----------------------
        The grid sampler only returns image patches if image is set.
        Otherwise, only the patch indices w_ini, w_fin, h_ini, h_fin, d_ini, d_fin are returned. They can be used to extract the patch from the image like this:
        img = img[w_ini:w_fin, h_ini:h_fin, d_ini:d_fin] (Example for a 3D image)
        Requiring only size parameters instead of the actual image makes the grid sampler file format independent if desired.

        :param image: The image data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions. Can also be a dict of multiple images.
        If None then patch indices (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...) are returned instead.
        :param spatial_size: The shape of the image without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param patch_overlap: The shape of the patch overlap without batch and channel dimensions. If None then the patch overlap is equal to patch_size.
        """
        super().__init__(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, image=image, spatial_first=spatial_first)

    def compute_indices(self):
        n_axis = len(self.spatial_size)
        stop = [self.spatial_size[axis] - self.patch_size[axis] + 1 for axis in range(n_axis)]
        axis_indices = [np.arange(0, stop[axis], self.patch_overlap[axis]) for axis in range(n_axis)]
        for axis in range(n_axis):
            if axis_indices[axis][-1] != self.spatial_size[axis] - self.patch_size[axis]:
                axis_indices[axis] = np.append(axis_indices[axis], [self.spatial_size[axis] - self.patch_size[axis]],
                                               axis=0)
        axis_indices = np.meshgrid(*axis_indices, indexing='ij')
        indices = np.column_stack([axis_indices[axis].ravel() for axis in range(n_axis)])
        return indices


class _AdaptiveGridSampler(_CropGridSampler):
    def __init__(self, spatial_size, patch_size, patch_overlap=None, image=None, spatial_first=True):
        # TODO: When used in ChunkedGridSampler the adaptive patches should have a minimum size of patch size
        # TODO: Do doc
        super().__init__(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, image=image, spatial_first=spatial_first)

    def compute_indices(self):
        n_axis = len(self.spatial_size)
        stop = [self.spatial_size[axis] for axis in range(n_axis)]
        axis_indices = [np.arange(0, stop[axis], self.patch_overlap[axis]) for axis in range(n_axis)]
        axis_indices = np.meshgrid(*axis_indices, indexing='ij')
        indices = np.column_stack([axis_indices[axis].ravel() for axis in range(n_axis)])
        return indices

    def __getitem__(self, idx):
        indices = self.indices[idx]
        patch_indices = np.zeros(len(indices) * 2, dtype=int).reshape(-1, 2)
        for axis in range(len(indices)):
            patch_indices[axis][0] = indices[axis]
            patch_indices[axis][1] = min(indices[axis] + self.patch_size[axis], self.spatial_size[axis])
        slice_result = self.slice_patch(patch_indices)
        return slice_result


class _ChunkedGridSampler(_CropGridSampler):
    def __init__(self, spatial_size, patch_size, chunk_size, patch_overlap=None, image=None, spatial_first=True, mode: str = "sample_edge"):
        self.chunk_size = chunk_size
        self.mode = mode
        super().__init__(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, image=image, spatial_first=spatial_first)

        self.compute_length()
        self.chunk_index = 0
        self.patch_index = 0

    def compute_indices(self):
        self.grid_sampler = GridSampler(spatial_size=self.spatial_size, patch_size=self.chunk_size, patch_overlap=self.chunk_size - self.patch_size, mode=self.mode)
        # TODO: Check if ChunkGridSampler still works with AdaptiveGridSampler
        # self.grid_sampler = AdaptiveGridSampler(image_size=self.image_size, patch_size=self.chunk_size, patch_overlap=self.chunk_size - self.patch_size, min_overlap=self.patch_size)
        self.chunk_sampler = []
        self.chunk_sampler_offset = []

        for chunk_indices in self.grid_sampler:
            chunk_indices = chunk_indices.reshape(-1, 2)
            chunk_size = copy.copy(chunk_indices[:, 1] - chunk_indices[:, 0])
            self.chunk_sampler.append(
                _EdgeGridSampler(spatial_size=chunk_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap))
            self.chunk_sampler_offset.append(copy.copy(chunk_indices[:, 0]))
        return None

    def compute_length(self):
        self.cumsum_length = [0]
        self.cumsum_length.extend([len(sampler) for sampler in self.chunk_sampler])
        self.cumsum_length = np.cumsum(self.cumsum_length)

    def __len__(self):
        return self.cumsum_length[-1]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration
        chunk_id = np.argmax(self.cumsum_length > idx)
        patch_id = idx - self.cumsum_length[chunk_id - 1]
        chunk_id -= 1  # -1 in order to remove the [0] appended at the start of self.cumsum_length

        patch_indices = copy.copy(self.chunk_sampler[chunk_id].__getitem__(patch_id))
        patch_indices += self.chunk_sampler_offset[chunk_id].reshape(-1, 1)

        slice_result = self.slice_patch(patch_indices)
        if self.image is None:
            slice_result = slice_result, chunk_id
        else:
            slice_result = *slice_result, chunk_id
        return slice_result


# class ResizeSampler(Dataset):
#     def __init__(self, sampler, target_size, image_size=None, patch_size=None, patch_overlap=None, mode='edge'):
#         self.sampler = sampler
#         self.target_size = target_size
#         self.mode = mode
#         # self.size_conversion_factor = size_conversion_factor
#         # Required due to otherwise shitty rounding errors in resize_indices()  # Still leads to rounding errors...
#         # self.size_conversion_factor = np.floor(np.max(self.sampler.chunk_sampler_offset[1]) * size_conversion_factor) / np.max(self.sampler.chunk_sampler_offset[1])
#         self.resized_sampler = _EdgeGridSampler(spatial_size=image_size, patch_size=patch_size,
#                                                 patch_overlap=patch_overlap)
#         self.sanity_check()
#
#     def sanity_check(self):
#         if len(self.sampler) != len(self.resized_sampler):
#             raise RuntimeError("Lengths of sampler ({}) and resized_sampler ({}) do not match.".format(self.sampler.cumsum_length[-1], self.resized_sampler.length[-1]))
#
#     def __getitem__(self, idx):
#         output = self.sampler.__getitem__(idx)
#         resized_patch_indices = self.resized_sampler.__getitem__(idx)
#         if len(output) == 2 and not isinstance(output[0], dict):
#             patch, patch_indices = output
#             patch = ski_transform.resize(patch, output_shape=self.target_size, order=1, mode=self.mode)
#             # patch_indices = self.resize_indices(patch_indices)
#             return patch, resized_patch_indices
#         elif len(output) == 2 and isinstance(output[0], dict):
#             patch_dict, patch_indices = output
#             for key in patch_dict.keys():
#                 patch_dict[key] = ski_transform.resize(patch_dict[key], output_shape=self.target_size, order=1, mode=self.mode)
#             # patch_indices = self.resize_indices(patch_indices)
#             return patch_dict, resized_patch_indices
#         else:
#             patch_indices = output
#             # patch_indices = self.resize_indices(patch_indices)
#             return resized_patch_indices
#
#     # def resize_indices(self, patch_indices):
#     #     patch_indices = np.rint(patch_indices * self.size_conversion_factor).astype(np.int64)
#     #     return patch_indices
#
#     def __len__(self):
#         return len(self.sampler)
#
#
# class ChunkedResizeSampler(Dataset):
#     def __init__(self, sampler, target_size, image_size=None, patch_size=None, patch_overlap=None, chunk_size=None, mode='edge'):
#         self.sampler = sampler
#         self.target_size = target_size
#         self.mode = mode
#         # self.size_conversion_factor = size_conversion_factor
#         # Required due to otherwise shitty rounding errors in resize_indices()  # Still leads to rounding errors...
#         # self.size_conversion_factor = np.floor(np.max(self.sampler.chunk_sampler_offset[1]) * size_conversion_factor) / np.max(self.sampler.chunk_sampler_offset[1])
#         self.resized_sampler = _ChunkedGridSampler(spatial_size=image_size, patch_size=patch_size,
#                                                    patch_overlap=patch_overlap, chunk_size=chunk_size)
#         self.sanity_check()
#
#     def sanity_check(self):
#         if self.sampler.cumsum_length[-1] != self.resized_sampler.cumsum_length[-1]:
#             raise RuntimeError("Lengths of sampler ({}) and resized_sampler ({}) do not match.".format(self.sampler.cumsum_length[-1], self.resized_sampler.cumsum_length[-1]))
#
#     def __getitem__(self, idx):
#         output = self.sampler.__getitem__(idx)
#         resized_patch_indices = self.resized_sampler.__getitem__(idx)[0]
#         if len(output) == 2 and not isinstance(output[0], dict):
#             patch, (patch_indices, chunk_id) = output
#             patch = ski_transform.resize(patch, output_shape=self.target_size, order=1, mode=self.mode)
#             # patch_indices = self.resize_indices(patch_indices)
#             return patch, (resized_patch_indices, chunk_id)
#         elif len(output) == 2 and isinstance(output[0], dict):
#             patch_dict, (patch_indices, chunk_id) = output
#             for key in patch_dict.keys():
#                 patch_dict[key] = ski_transform.resize(patch_dict[key], output_shape=self.target_size, order=1, mode=self.mode)
#             # patch_indices = self.resize_indices(patch_indices)
#             return patch_dict, (resized_patch_indices, chunk_id)
#         else:
#             (patch_indices, chunk_id) = output
#             # patch_indices = self.resize_indices(patch_indices)
#             return (resized_patch_indices, chunk_id)
#
#     # def resize_indices(self, patch_indices):
#     #     patch_indices = np.rint(patch_indices * self.size_conversion_factor).astype(np.int64)
#     #     return patch_indices
#
#     def __len__(self):
#         return len(self.sampler)


# class SamplerDataset(Dataset):
#     def __init__(self, sampler):
#         self.sampler = sampler
#
#     def __getitem__(self, idx):
#         output = self.sampler.__getitem__(idx)
#         if len(output) == 2 and not isinstance(output[0], dict):
#             patch, patch_indices = output
#             patch = patch[np.newaxis, ...].astype(np.float32)
#             return patch, patch_indices
#         elif len(output) == 2 and isinstance(output[0], dict):
#             patch_dict, patch_indices = output
#             for key in patch_dict.keys():
#                 patch_dict[key] = patch_dict[key][np.newaxis, ...].astype(np.float32)
#             return patch_dict, patch_indices
#         else:
#             return output
#
#     def __len__(self):
#         return len(self.sampler)


# class GridSamplerDataset(GridSampler, Dataset):
#     def __init__(self, image=None, image_size=None, patch_size=None, patch_overlap=None):
#         GridSampler.__init__(self, image, image_size, patch_size, patch_overlap)
#
#     def __getitem__(self, idx):
#         output = GridSampler.__getitem__(self, idx)
#         if len(output) == 2 and not isinstance(output[0], dict):
#             patch, patch_indices = output
#             patch = patch[np.newaxis, ...].astype(np.float32)
#             return patch, patch_indices
#         elif len(output) == 2 and isinstance(output[0], dict):
#             patch_dict, patch_indices = output
#             for key in patch_dict.keys():
#                 patch_dict[key] = patch_dict[key][np.newaxis, ...].astype(np.float32)
#             return patch_dict, patch_indices
#         else:
#             return output


# class GridResamplerDataset(GridSamplerDataset):
#     def __init__(self, image=None, image_size=None, patch_size=None, patch_overlap=None, target_size=None):
#         """
#         An N-dimensional grid sampler that should mainly be used for inference. The image is divided into a grid with each grid cell having the size of patch_size. The grid can have overlap if patch_overlap is specified.
#         If patch_size is not a multiple of image_size then the remainder part of the image is not padded, but instead patches are sampled at the edge of the image of size patch_size like this:
#         ----------------------
#         |                | X |
#         |                | X |
#         |                | X |
#         |                | X |
#         |----------------| X |
#         |X  X  X  X  X  X  X |
#         ----------------------
#         The grid sampler only returns image patches if image is set.
#         Otherwise, only the patch indices w_ini, w_fin, h_ini, h_fin, d_ini, d_fin are returned. They can be used to extract the patch from the image like this:
#         img = img[w_ini:w_fin, h_ini:h_fin, d_ini:d_fin] (Example for a 3D image)
#         Requiring only size parameters instead of the actual image makes the grid sampler file format independent if desired.
#
#         :param image: The image data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions. Can also be a dict of multiple images.
#         If None then patch indices (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...) are returned instead.
#         :param image_size: The shape of the image without batch and channel dimensions. Always required.
#         :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
#         :param patch_overlap: The shape of the patch overlap without batch and channel dimensions. If None then the patch overlap is equal to patch_size.
#         :param target_size: The target size that should be used to resample the patch.
#         """
#         super().__init__(image, image_size, patch_size, patch_overlap)
#         self.target_size = target_size
#         self.size_conversion_factor = (self.target_size / self.patch_size)[0]
#
#     def __getitem__(self, idx):
#         output = GridSampler.__getitem__(self, idx)
#         if len(output) == 2 and not isinstance(output[0], dict):
#             patch, patch_indices = output
#             patch = ski_transform.resize(patch, output_shape=self.target_size, order=1)
#             patch = patch[np.newaxis, ...].astype(np.float32)
#             patch_indices = np.rint(patch_indices * self.size_conversion_factor).astype(np.int32)
#             return patch, patch_indices
#         elif len(output) == 2 and isinstance(output[0], dict):
#             patch_dict, patch_indices = output
#             for key in patch_dict.keys():
#                 patch = patch_dict[key]
#                 patch = ski_transform.resize(patch, output_shape=self.target_size, order=1)
#                 patch_dict[key] = patch[np.newaxis, ...].astype(np.float32)
#             patch_indices = np.rint(patch_indices * self.size_conversion_factor).astype(np.int32)
#             return patch_dict, patch_indices
#         else:
#             output = np.rint(output * self.size_conversion_factor).astype(np.int32)
#             return output


class UniformSampler:
    def __init__(self, subjects, spatial_dims, patch_size, length, seed=None, channel_first=True):
        self.subjects = subjects
        self.spatial_dims = spatial_dims
        self.patch_size = np.asarray(patch_size)
        self.length = length
        self.channel_first = channel_first
        self.seeded_patches = self.seed_patches(seed)

    def seed_patches(self, seed):
        if seed is None:
            return None
        else:
            np.random.seed(seed)
            seeded_patches = {}
            for idx in range(self.length):
                subject_name, patch_indices, class_id = self.get_patch()
                seeded_patches[idx] = {"subject_name": subject_name, "patch_indices": patch_indices, "class_id": class_id}
            np.random.seed()
            return seeded_patches

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.seeded_patches is None:
            subject_name, patch_indices, class_id = self.get_patch()
        else:
            subject_name = self.seeded_patches[idx]["subject_name"]
            patch_indices = self.seeded_patches[idx]["patch_indices"]
            class_id = self.seeded_patches[idx]["class_id"]

        patch_dict = {}
        for key in self.subjects[subject_name].keys():
            slices = self.get_slices(self.subjects[subject_name][key], patch_indices)
            patch_dict[key] = self.subjects[subject_name][key][slicer(self.subjects[subject_name][key], slices)]
        patch_dict["patch_indices"] = patch_indices
        patch_dict["subject_name"] = subject_name
        # mask = patch_dict[aug.SEG] == class_id
        # has_class = np.max(mask) == 1
        # print("class: {}, has_class: {}".format(class_id, has_class))
        return patch_dict

    def get_patch(self):
        subject_idx = random.randint(0, len(self.subjects) - 1)
        subject_name = self.subjects.index(subject_idx)
        pos = self.random_position(self.patch_size, self.subjects[subject_name][aug.IMAGE].shape[-self.spatial_dims:])
        patch_indices = np.stack((pos, pos + self.patch_size), axis=1)
        return subject_name, patch_indices, None

    def __next__(self):
        if self.index < self.length:
            output = self.__getitem__(self.index)
            self.index += 1
            return output
        else:
            raise StopIteration

    def get_slices(self, image, patch_indices):
        non_spatial_dims = len(image.shape) - self.spatial_dims
        if self.channel_first:
            slices = [None] * non_spatial_dims
            slices.extend([index_pair.tolist() for index_pair in patch_indices])
        else:
            slices = [index_pair.tolist() for index_pair in patch_indices]
            slices.extend([None] * non_spatial_dims)
        return slices

    def random_position(self, patch_size, image_shape):
        pos = [random.randint(0, image_shape[axis] - patch_size[axis]) for axis in range(len(self.patch_size))]
        return pos


class WeightedSampler(UniformSampler):
    def __init__(self, subjects, spatial_dims, patch_size, length, population, class_weights=None, seed=None, channel_first=True):
        self.population = population
        self.class_weights = class_weights
        super().__init__(subjects, spatial_dims, patch_size, length, seed, channel_first)

    def get_patch(self):
        position, subject_name, class_id = self.population.get_sample(self.class_weights)
        image_shape = self.population.map_shapes[subject_name]
        position = self.random_position_around_point(position, self.patch_size, image_shape, self.patch_size - 1)
        patch_indices = np.stack((position, position + self.patch_size), axis=1)
        return subject_name, patch_indices, class_id

    def random_position_around_point(self, position, patch_size, image_shape, max_movement):
        min_pos = [max(0, position[axis] - (patch_size[axis] - max_movement[axis])) for axis in range(len(patch_size))]
        max_pos = [min(position[axis], image_shape[axis] - patch_size[axis]) for axis in range(len(patch_size))]
        pos = [random.randint(min_pos[axis], max_pos[axis]) if min_pos[axis] < max_pos[axis] else max_pos[axis] for axis in range(len(self.patch_size))]
        return pos


if __name__ == '__main__':
    # image_size = (1000, 1000, 1000)  # (2010, 450, 2010)
    # patch_size = (100, 100, 100)
    # patch_overlap = (100, 100, 100)

    image_size = (301,)  # (2010, 450, 2010)
    patch_size = (20,)
    patch_overlap = (10,)
    chunk_size = (60,)
    grid_sampler = _ChunkedGridSampler(spatial_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap,
                                       chunk_size=chunk_size)

    # image_size = (299,)  # (2010, 450, 2010)
    # patch_size = (50,)
    # patch_overlap = (40,)
    # # chunk_size = (50,)
    # grid_sampler = GridSampler(image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap)

    print(len(grid_sampler))

    for i, indices in enumerate(grid_sampler):
        print("Iteration: {}, indices: {}".format(i, indices))
