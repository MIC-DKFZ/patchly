import numpy as np
from samplify.slicer import slicer
from samplify import utils
import copy
import random
# import augmentify as aug
from typing import Union, Optional, Tuple
import numpy.typing as npt


class GridSampler:
    def __init__(self, spatial_size: Union[Tuple, npt.ArrayLike], patch_size: Union[Tuple, npt.ArrayLike], patch_overlap: Optional[Union[Tuple, npt.ArrayLike]] = None,
                 chunk_size: Optional[Union[Tuple, npt.ArrayLike]] = None, image: Optional[npt.ArrayLike] = None, spatial_first: bool = True, mode: str = 'sample_edge', pad_kwargs: dict = None):
        """
        TODO description
        If no image is given then only patch indices (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...) are returned instead.

        :param image: The image in an array-like format (Numpy, Samplify.Subject, Tensor, Zarr, Dask, ...) that can be memory-mapped.
        Samplify.subject is directly supported and will sample the same patch from all stored images.
        The image can have an arbitrary number of additional non-spatial dimensions.
        :param spatial_size: The spatial shape of the image. The spatial shape excludes the channel, batch and any other non-spatial dimensionality.
        :param patch_size: The spatial shape of the patch. The patch shape excludes the channel, batch and any other non-spatial dimensionality.
        :param patch_overlap: The spatial shape of the patch overlap. If None then the patch overlap is equal to the patch size.
        The patch overlap excludes the channel, batch and any other non-spatial dimensionality.
        :param chunk_size: The spatial shape of the chunk size. If given, the image is divided into chunks and patches are sampled from a single chunk until the chunk is depleted.
        This enables patch sampling of larger-than-RAM images in combination with memory-mapped arrays. The chunk size is required to be a multiple of the patch size.
        The chunk size excludes the channel, batch and any other non-spatial dimensionality. An in-depth explanation of chunk sampling can be found here: LINK
        :param spatial_first: Denotes that the spatial dimensions of the image are located before the non-spatial dimensions e.g. (Width, Height, Channel).
        Otherwise, the reverse is true e.g. (Channel, Width, Height).
        :param mode: TODO
        """
        self.image = image
        self.spatial_size = np.asarray(spatial_size)
        self.patch_size = np.asarray(patch_size)
        self.patch_overlap = self.set_patch_overlap(patch_overlap, patch_size)
        self.chunk_size = self.set_chunk_size(chunk_size)
        self.spatial_first = spatial_first
        self.mode = mode
        self.pad_kwargs = pad_kwargs
        self.pad_width = None
        self.check_sanity()
        self.sampler = self.create_sampler()

    def set_patch_overlap(self, patch_overlap, patch_size):
        if patch_overlap is None:
            patch_overlap = patch_size
        else:
            patch_overlap = np.asarray(patch_overlap)
        return patch_overlap

    def set_chunk_size(self, parameter):
        if parameter is None:
            parameter = None
        else:
            parameter = np.asarray(parameter)
        return parameter

    def check_sanity(self):
        if self.image is not None and not hasattr(self.image, '__getitem__'):
            raise RuntimeError("The given image is not ArrayLike.")
        if self.spatial_first and self.image is not None and (self.image.shape[:len(self.spatial_size)] != tuple(self.spatial_size)):
            raise RuntimeError("The spatial size of the given image {} is unequal to the given spatial size {}.".format(self.image.shape[:len(self.spatial_size)], self.spatial_size))
        if (not self.spatial_first) and self.image is not None and (self.image.shape[-len(self.spatial_size):] != tuple(self.spatial_size)):
            raise RuntimeError("The spatial size of the given image {} is unequal to the given spatial size {}.".format(self.image.shape[-len(self.spatial_size):], self.spatial_size))
        if np.any(self.patch_size > self.spatial_size):
            raise RuntimeError("The patch size ({}) cannot be greater than the spatial size ({}) in one or more dimensions.".format(self.patch_size, self.spatial_size))
        if self.patch_overlap is not None and np.any(self.patch_overlap > self.patch_size):
            raise RuntimeError("The patch overlap ({}) cannot be greater than the patch size ({}) in one or more dimensions.".format(self.patch_overlap, self.patch_size))
        if self.chunk_size is not None and np.any(self.chunk_size > self.spatial_size):
            raise RuntimeError("The chunk size ({}) cannot be greater than the spatial size ({}) in one or more dimensions.".format(self.chunk_size, self.spatial_size))
        if self.chunk_size is not None and np.any(self.patch_size >= self.chunk_size):
            raise RuntimeError("The patch size ({}) cannot be greater or equal to the chunk size ({}) in one or more dimensions.".format(self.patch_size, self.chunk_size))
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
        if self.mode.startswith('pad_') and (self.image is None or not isinstance(self.image, np.ndarray)):
            raise RuntimeError("The given sampling mode ({}) requires the image to be given and as type np.ndarray.".format(self.mode))
        if self.mode.startswith('pad_') and self.chunk_size is not None:
            raise RuntimeError("The given sampling mode ({}) is not compatible with chunk sampling.".format(self.mode))
        
    def create_sampler(self):
        if self.mode == "sample_edge" and self.chunk_size is None:
            sampler = _EdgeGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, spatial_first=self.spatial_first)
        elif self.mode == "sample_edge" and self.chunk_size is not None:
            sampler = _ChunkGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, chunk_size=self.chunk_size,
                                        spatial_first=self.spatial_first)
        elif self.mode == "sample_adaptive" and self.chunk_size is None:
            sampler = _AdaptiveGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, spatial_first=self.spatial_first)
        elif self.mode == "sample_adaptive" and self.chunk_size is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        elif self.mode == "sample_crop" and self.chunk_size is None:
            sampler = _CropGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, spatial_first=self.spatial_first)
        elif self.mode == "sample_crop" and self.chunk_size is not None:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        elif self.mode.startswith('pad_'):
            self.pad_image()
            sampler = _CropGridSampler(image=self.image, spatial_size=self.spatial_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap, spatial_first=self.spatial_first)
        else:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        return sampler

    def pad_image(self):
        if self.mode.startswith('pad_end_'):
            pad_width_after = np.asarray(self.spatial_size) - np.asarray(self.image.shape)
            pad_width_after = np.clip(pad_width_after, a_min=0, a_max=None)
            self.spatial_size += pad_width_after
            pad_width_after = pad_width_after[..., np.newaxis]
            pad_width = np.hstack((np.zeros_like(pad_width_after), pad_width_after))
            pad_mode = self.mode[8:]
        elif self.mode.startswith('pad_edges_'):
            pad_width_after = np.asarray(self.spatial_size) - np.asarray(self.image.shape)
            pad_width_after = np.clip(pad_width_after, a_min=0, a_max=None)
            self.spatial_size += pad_width_after
            pad_width_before = pad_width_after // 2
            pad_width_after = pad_width_after - pad_width_before
            pad_width_after = pad_width_after[..., np.newaxis]
            pad_width_before = pad_width_before[..., np.newaxis]
            pad_width = np.hstack((pad_width_before, pad_width_after))
            pad_mode = self.mode[10:]
        else:
            raise RuntimeError("The given sampling mode ({}) is not supported.".format(self.mode))

        if self.pad_kwargs is None:
            self.pad_kwargs = {}
        self.image = np.pad(self.image, pad_width, mode=pad_mode, **self.pad_kwargs)
        self.pad_width = pad_width

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self):
        return self.sampler.__len__()

    def __getitem__(self, idx):
        return self.sampler.__getitem__(idx)

    def __next__(self):
        return self.sampler.__next__()


class _CropGridSampler:
    def __init__(self, spatial_size: np.ndarray, patch_size: np.ndarray, patch_overlap: np.ndarray, image: Optional[npt.ArrayLike] = None, spatial_first: bool = True):
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
        self.indices, self.patch_sizes = self.compute_patches()

    def compute_patches(self):
        n_axis = len(self.spatial_size)
        stop = [self.spatial_size[axis] - self.patch_size[axis] + 1 for axis in range(n_axis)]
        axis_indices = [np.arange(0, stop[axis], self.patch_overlap[axis]) for axis in range(n_axis)]
        patch_sizes = [[self.patch_size[axis]] * len(axis_indices[axis]) for axis in range(n_axis)]
        axis_indices = np.meshgrid(*axis_indices, indexing='ij')
        patch_sizes = np.meshgrid(*patch_sizes, indexing='ij')
        indices = np.column_stack([axis_indices[axis].ravel() for axis in range(n_axis)])
        patch_sizes = np.column_stack([patch_sizes[axis].ravel() for axis in range(n_axis)])
        return indices, patch_sizes

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
            patch_indices[axis][1] = indices[axis] + self.patch_sizes[idx][axis]
        patch_result = self.get_patch_result(patch_indices)
        return patch_result

    def __next__(self):
        if self.index < self.__len__():
            output = self.__getitem__(self.index)
            self.index += 1
            return output
        else:
            raise StopIteration

    def get_patch_result(self, patch_indices):
        if self.image is not None and not isinstance(self.image, dict):
            slices = utils.add_non_spatial_indices(self.image, patch_indices, self.spatial_size, self.spatial_first)
            patch = self.image[slicer(self.image, slices)]
            return patch, patch_indices
        elif self.image is not None and isinstance(self.image, dict):
            patch_dict = {}
            for key in self.image.keys():
                slices = utils.add_non_spatial_indices(self.image[key], patch_indices, self.spatial_size, self.spatial_first)
                patch_dict[key] = self.image[key][slicer(self.image[key], slices)]
            return patch_dict, patch_indices
        else:
            return patch_indices


class _EdgeGridSampler(_CropGridSampler):
    def __init__(self, spatial_size: np.ndarray, patch_size: np.ndarray, patch_overlap: np.ndarray, image: Optional[npt.ArrayLike] = None, spatial_first: bool = True):
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

    def compute_patches(self):
        n_axis = len(self.spatial_size)
        stop = [self.spatial_size[axis] - self.patch_size[axis] + 1 for axis in range(n_axis)]
        axis_indices = [np.arange(0, stop[axis], self.patch_overlap[axis]) for axis in range(n_axis)]
        for axis in range(n_axis):
            if axis_indices[axis][-1] != self.spatial_size[axis] - self.patch_size[axis]:
                axis_indices[axis] = np.append(axis_indices[axis], [self.spatial_size[axis] - self.patch_size[axis]],
                                               axis=0)
        patch_sizes = [[self.patch_size[axis]] * len(axis_indices[axis]) for axis in range(n_axis)]
        axis_indices = np.meshgrid(*axis_indices, indexing='ij')
        patch_sizes = np.meshgrid(*patch_sizes, indexing='ij')
        indices = np.column_stack([axis_indices[axis].ravel() for axis in range(n_axis)])
        patch_sizes = np.column_stack([patch_sizes[axis].ravel() for axis in range(n_axis)])
        return indices, patch_sizes


class _AdaptiveGridSampler(_CropGridSampler):
    def __init__(self, spatial_size: np.ndarray, patch_size: np.ndarray, patch_overlap: np.ndarray, image: Optional[npt.ArrayLike] = None, spatial_first: bool = True):
        # TODO: When used in ChunkedGridSampler the adaptive patches should have a minimum size of patch size
        # TODO: Do doc
        super().__init__(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, image=image, spatial_first=spatial_first)

    def compute_patches(self):
        n_axis = len(self.spatial_size)
        stop = [self.spatial_size[axis] for axis in range(n_axis)]
        axis_indices = [np.arange(0, stop[axis], self.patch_overlap[axis]) for axis in range(n_axis)]
        patch_sizes = [[self.patch_size[axis]] * len(axis_indices[axis]) for axis in range(n_axis)]
        for axis in range(n_axis):
            patch_sizes[axis][-1] = self.spatial_size[axis] - axis_indices[axis][-1]
        axis_indices = np.meshgrid(*axis_indices, indexing='ij')
        patch_sizes = np.meshgrid(*patch_sizes, indexing='ij')
        indices = np.column_stack([axis_indices[axis].ravel() for axis in range(n_axis)])
        patch_sizes = np.column_stack([patch_sizes[axis].ravel() for axis in range(n_axis)])
        return indices, patch_sizes


class _ChunkGridSampler(_CropGridSampler):
    def __init__(self, spatial_size: np.ndarray, patch_size: np.ndarray, chunk_size: np.ndarray, patch_overlap: np.ndarray, image: Optional[npt.ArrayLike] = None, spatial_first: bool = True, mode: str = 'sample_edge'):
        self.chunk_size = chunk_size
        self.mode = mode
        super().__init__(spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, image=image, spatial_first=spatial_first)

        self.compute_length()
        self.chunk_index = 0
        self.patch_index = 0

    def compute_patches(self):
        self.grid_sampler = GridSampler(spatial_size=self.spatial_size, patch_size=self.chunk_size, patch_overlap=self.chunk_size - self.patch_size, mode=self.mode)
        self.chunk_sampler = []
        self.chunk_sampler_offset = []

        for chunk_indices in self.grid_sampler:
            chunk_indices = chunk_indices.reshape(-1, 2)
            chunk_size = copy.copy(chunk_indices[:, 1] - chunk_indices[:, 0])
            self.chunk_sampler.append(
                _CropGridSampler(spatial_size=chunk_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap))
            self.chunk_sampler_offset.append(copy.copy(chunk_indices[:, 0]))
        return None, None

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

        patch_result = self.get_patch_result(patch_indices)
        if self.image is None:
            patch_result = patch_result, chunk_id
        else:
            patch_result = *patch_result, chunk_id
        return patch_result
