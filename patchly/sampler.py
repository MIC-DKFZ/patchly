import numpy as np
from patchly.slicer import slicer
from patchly import utils
from typing import Union, Optional, Tuple
import numpy.typing as npt
from enum import Enum


class SamplingMode(Enum):
    SAMPLE_EDGE = 1
    SAMPLE_ADAPTIVE = 2
    SAMPLE_CROP = 3
    SAMPLE_SQUEEZE = 4
    PAD_UNKNOWN = 5


class GridSampler:
    def __init__(self, spatial_size: Union[Tuple, npt.ArrayLike], patch_size: Union[Tuple, npt.ArrayLike], step_size: Optional[Union[Tuple, npt.ArrayLike]] = None,
                 image: Optional[npt.ArrayLike] = None, spatial_first: bool = True, mode: SamplingMode = SamplingMode.SAMPLE_SQUEEZE, pad_kwargs: dict = None):
        """
        TODO description
        If no image is given then only patch bbox (w_start, w_end, h_start, h_end, d_start, d_end, ...) are returned instead.

        :param image: The image in an array-like format (Numpy, Tensor, Zarr, Dask, ...) that can be memory-mapped.
        The image can have an arbitrary number of additional non-spatial dimensions.
        :param spatial_size: The spatial shape of the image. The spatial shape excludes the channel, batch and any other non-spatial dimensionality.
        :param patch_size: The spatial shape of the patch. The patch shape excludes the channel, batch and any other non-spatial dimensionality.
        :param step_size: The spatial shape of the patch offset. If None then the patch offset is equal to the patch size.
        The patch offset excludes the channel, batch and any other non-spatial dimensionality.
        This enables patch sampling of larger-than-RAM images in combination with memory-mapped arrays. The chunk size is required to be a multiple of the patch size.
        The chunk size excludes the channel, batch and any other non-spatial dimensionality. An in-depth explanation of chunk sampling can be found here: LINK
        :param spatial_first: Denotes that the spatial dimensions of the image are located before the non-spatial dimensions e.g. (Width, Height, Channel).
        Otherwise, the reverse is true e.g. (Channel, Width, Height).
        :param mode: TODO
        """
        self.image_h = image
        self.image_size_s = np.asarray(spatial_size)
        self.patch_size_s = np.asarray(patch_size)
        self.step_size_s = self.set_step_size(step_size, patch_size)
        self.spatial_first = spatial_first
        self.mode = mode
        self.pad_kwargs = pad_kwargs
        self.pad_width = None
        self.check_sanity()
        self.sampler = self.create_sampler()

    def set_step_size(self, step_size_s, patch_size_s):
        if step_size_s is None:
            step_size_s = patch_size_s
        else:
            step_size_s = np.asarray(step_size_s)
        return step_size_s

    def check_sanity(self):
        if self.image_h is not None and not hasattr(self.image_h, '__getitem__'):
            raise RuntimeError("The given image is not ArrayLike.")
        if self.spatial_first and self.image_h is not None and (self.image_h.shape[:len(self.image_size_s)] != tuple(self.image_size_s)):
            raise RuntimeError("The spatial size of the given image {} is unequal to the given spatial size {}.".format(self.image_h.shape[:len(self.image_size_s)], self.image_size_s))
        if (not self.spatial_first) and self.image_h is not None and (self.image_h.shape[-len(self.image_size_s):] != tuple(self.image_size_s)):
            raise RuntimeError("The spatial size of the given image {} is unequal to the given spatial size {}.".format(self.image_h.shape[-len(self.image_size_s):], self.image_size_s))
        if np.any(self.patch_size_s > self.image_size_s):
            raise RuntimeError("The patch size ({}) cannot be greater than the spatial size ({}) in one or more dimensions.".format(self.patch_size_s, self.image_size_s))
        if self.step_size_s is not None and np.any(self.step_size_s > self.patch_size_s):
            raise RuntimeError("The patch offset ({}) cannot be greater than the patch size ({}) in one or more dimensions.".format(self.step_size_s, self.patch_size_s))
        if len(self.image_size_s) != len(self.patch_size_s):
            raise RuntimeError("The dimensionality of the patch size ({}) is required to be the same as the spatial size ({})."
                               .format(self.patch_size_s, self.image_size_s))
        if self.step_size_s is not None and len(self.image_size_s) != len(self.step_size_s):
            raise RuntimeError("The dimensionality of the patch offset ({}) is required to be the same as the spatial size ({})."
                               .format(self.step_size_s, self.image_size_s))
        if self.mode.name.startswith('PAD_') and (self.image_h is None or not isinstance(self.image_h, np.ndarray)):
            raise RuntimeError("The given sampling mode ({}) requires the image to be given and as type np.ndarray.".format(self.mode))
        
    def create_sampler(self):
        if self.mode == SamplingMode.SAMPLE_EDGE:
            sampler = _EdgeGridSampler(image_h=self.image_h, image_size_s=self.image_size_s, patch_size_s=self.patch_size_s, step_size_s=self.step_size_s, spatial_first=self.spatial_first)
        elif self.mode == SamplingMode.SAMPLE_ADAPTIVE:
            sampler = _AdaptiveGridSampler(image_h=self.image_h, image_size_s=self.image_size_s, patch_size_s=self.patch_size_s, step_size_s=self.step_size_s, spatial_first=self.spatial_first)
        elif self.mode == SamplingMode.SAMPLE_CROP:
            sampler = _CropGridSampler(image_h=self.image_h, image_size_s=self.image_size_s, patch_size_s=self.patch_size_s, step_size_s=self.step_size_s, spatial_first=self.spatial_first)
        elif self.mode == SamplingMode.SAMPLE_SQUEEZE:
            sampler = _SqueezeGridSampler(image_h=self.image_h, image_size_s=self.image_size_s, patch_size_s=self.patch_size_s, step_size_s=self.step_size_s, spatial_first=self.spatial_first)
        elif self.mode.name.startswith('PAD_'):
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
            self.pad_image()
            sampler = _CropGridSampler(image_h=self.image_h, image_size_s=self.image_size_s, patch_size_s=self.patch_size_s, step_size_s=self.step_size_s, spatial_first=self.spatial_first)
        else:
            raise NotImplementedError("The given sampling mode ({}) is not supported.".format(self.mode))
        return sampler

    def pad_image(self):
        if self.mode.startswith('pad_end_'):
            pad_width_after = np.asarray(self.image_size_s) - np.asarray(self.image_h.shape)
            pad_width_after = np.clip(pad_width_after, a_min=0, a_max=None)
            self.image_size_s += pad_width_after
            pad_width_after = pad_width_after[..., np.newaxis]
            pad_width = np.hstack((np.zeros_like(pad_width_after), pad_width_after))
            pad_mode = self.mode[8:]
        elif self.mode.startswith('pad_edges_'):
            pad_width_after = np.asarray(self.image_size_s) - np.asarray(self.image_h.shape)
            pad_width_after = np.clip(pad_width_after, a_min=0, a_max=None)
            self.image_size_s += pad_width_after
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
        self.image_h = np.pad(self.image_h, pad_width, mode=pad_mode, **self.pad_kwargs)
        self.pad_width = pad_width

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self):
        return self.sampler.__len__()

    def __getitem__(self, idx):
        return self.sampler.__getitem__(idx)

    def __next__(self):
        return self.sampler.__next__()
    
    def _get_bbox(self, idx):
        return self.sampler._get_bbox(idx)


class _CropGridSampler:
    def __init__(self, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, image_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True):
        """
        TODO Redo doc

        An N-dimensional grid sampler that should mainly be used for inference. The image is divided into a grid with each grid cell having the size of patch_size. The grid can have offset if step_size is specified.
        If patch_size is not a multiple of image_size then the remainder part of the image is not sampled.
        The grid sampler only returns image patches if image is set.
        Otherwise, only the patch bbox w_start, w_end, h_start, h_end, d_start, d_end are returned. They can be used to extract the patch from the image like this:
        img = img[w_start:w_end, h_start:h_end, d_start:d_end] (Example for a 3D image)
        Requiring only size parameters instead of the actual image makes the grid sampler file format independent if desired.

        :param image: The image data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions. Can also be a dict of multiple images.
        If None then patch bbox (w_start, w_end, h_start, h_end, d_start, d_end, ...) are returned instead.
        :param spatial_size: The shape of the image without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param step_size: The shape of the patch offset without batch and channel dimensions. If None then the patch offset is equal to patch_size.
        """
        self.image_h = image_h
        self.image_size_s = image_size_s
        self.patch_size_s = patch_size_s
        self.step_size_s = step_size_s
        self.spatial_first = spatial_first
        self.patch_positions_s, self.patch_sizes_s = self.compute_patches()

    def compute_patches(self):
        n_axis_s = len(self.image_size_s)
        stop_s = [self.image_size_s[axis] - self.patch_size_s[axis] + 1 for axis in range(n_axis_s)]
        axis_positions_s = [np.arange(0, stop_s[axis], self.step_size_s[axis]) for axis in range(n_axis_s)]
        patch_sizes_s = [[self.patch_size_s[axis]] * len(axis_positions_s[axis]) for axis in range(n_axis_s)]
        axis_positions_s = np.meshgrid(*axis_positions_s, indexing='ij')
        patch_sizes_s = np.meshgrid(*patch_sizes_s, indexing='ij')
        patch_positions_s = np.column_stack([axis_positions_s[axis].ravel() for axis in range(n_axis_s)])
        patch_sizes_s = np.column_stack([patch_sizes_s[axis].ravel() for axis in range(n_axis_s)])
        return patch_positions_s, patch_sizes_s

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return len(self.patch_positions_s)

    def __getitem__(self, idx):
        patch_bbox_s = self._get_bbox(idx)
        patch_result = self.get_patch_result(patch_bbox_s)
        return patch_result

    def __next__(self):
        if self.index < self.__len__():
            output = self.__getitem__(self.index)
            self.index += 1
            return output
        else:
            raise StopIteration
        
    def _get_bbox(self, idx):
        patch_position_s = self.patch_positions_s[idx]
        patch_bbox_s = np.zeros(len(patch_position_s) * 2, dtype=int).reshape(-1, 2)
        for axis in range(len(patch_bbox_s)):
            patch_bbox_s[axis][0] = patch_position_s[axis]
            patch_bbox_s[axis][1] = patch_position_s[axis] + self.patch_sizes_s[idx][axis]
        return patch_bbox_s

    def get_patch_result(self, patch_bbox_s):
        if self.image_h is not None and not isinstance(self.image_h, dict):
            patch_bbox_h = utils.bbox_s_to_bbox_h(patch_bbox_s, self.image_h, self.spatial_first)
            patch_h = self.image_h[slicer(self.image_h, patch_bbox_h)]
            return patch_h, patch_bbox_s
        else:
            return patch_bbox_s


class _EdgeGridSampler(_CropGridSampler):
    def __init__(self, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, image_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True):
        """
        TODO Redo doc

        An N-dimensional grid sampler that should mainly be used for inference. The image is divided into a grid with each grid cell having the size of patch_size. The grid can have offset if step_size is specified.
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
        Otherwise, only the patch bbox w_start, w_end, h_start, h_end, d_start, d_end are returned. They can be used to extract the patch from the image like this:
        img = img[w_start:w_end, h_start:h_end, d_start:d_end] (Example for a 3D image)
        Requiring only size parameters instead of the actual image makes the grid sampler file format independent if desired.

        :param image: The image data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions. Can also be a dict of multiple images.
        If None then patch bbox (w_start, w_end, h_start, h_end, d_start, d_end, ...) are returned instead.
        :param spatial_size: The shape of the image without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param step_size: The shape of the patch offset without batch and channel dimensions. If None then the patch offset is equal to patch_size.
        """
        super().__init__(image_size_s=image_size_s, patch_size_s=patch_size_s, step_size_s=step_size_s, image_h=image_h, spatial_first=spatial_first)

    def compute_patches(self):
        n_axis_s = len(self.image_size_s)
        stop_s = [self.image_size_s[axis] - self.patch_size_s[axis] + 1 for axis in range(n_axis_s)]
        axis_positions_s = [np.arange(0, stop_s[axis], self.step_size_s[axis]) for axis in range(n_axis_s)]
        for axis in range(n_axis_s):
            if axis_positions_s[axis][-1] != self.image_size_s[axis] - self.patch_size_s[axis]:
                axis_positions_s[axis] = np.append(axis_positions_s[axis], [self.image_size_s[axis] - self.patch_size_s[axis]],
                                               axis=0)
        patch_sizes_s = [[self.patch_size_s[axis]] * len(axis_positions_s[axis]) for axis in range(n_axis_s)]
        axis_positions_s = np.meshgrid(*axis_positions_s, indexing='ij')
        patch_sizes_s = np.meshgrid(*patch_sizes_s, indexing='ij')
        patch_positions_s = np.column_stack([axis_positions_s[axis].ravel() for axis in range(n_axis_s)])
        patch_sizes_s = np.column_stack([patch_sizes_s[axis].ravel() for axis in range(n_axis_s)])
        return patch_positions_s, patch_sizes_s


class _AdaptiveGridSampler(_CropGridSampler):
    def __init__(self, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, image_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True, min_patch_size_s: np.ndarray = None):
        # TODO: When used in ChunkedGridSampler the adaptive patches should have a minimum size of patch size
        # TODO: Do doc
        self.min_patch_size_s = min_patch_size_s
        if self.min_patch_size_s is not None and np.any(self.min_patch_size_s > patch_size_s):
            raise RuntimeError("The minimum patch size ({}) cannot be greater than the actual patch size ({}) in one or more dimensions.".format(self.min_patch_size_s, patch_size_s))
        super().__init__(image_size_s=image_size_s, patch_size_s=patch_size_s, step_size_s=step_size_s, image_h=image_h, spatial_first=spatial_first)

    def compute_patches(self):
        n_axis_s = len(self.image_size_s)
        stop = [self.image_size_s[axis] for axis in range(n_axis_s)]
        axis_positions_s = [np.arange(0, stop[axis], self.step_size_s[axis]) for axis in range(n_axis_s)]
        patch_sizes_s = [[self.patch_size_s[axis]] * len(axis_positions_s[axis]) for axis in range(n_axis_s)]
        for axis in range(n_axis_s):
            for index in range(len(axis_positions_s[axis])):
                # If part of this patch is extending beyonf the image
                if self.image_size_s[axis] < axis_positions_s[axis][index] + patch_sizes_s[axis][index]:
                    patch_sizes_s[axis][index] = self.image_size_s[axis] - axis_positions_s[axis][index]
                    # If there is a minimum patch size, give the patch at least minimum patch size
                    if self.min_patch_size_s is not None and patch_sizes_s[axis][index] < self.min_patch_size_s[axis]:
                        axis_positions_s[axis][index] = self.image_size_s[axis] - self.min_patch_size_s[axis]
                        patch_sizes_s[axis][index] = self.min_patch_size_s[axis]                
        axis_positions_s = np.meshgrid(*axis_positions_s, indexing='ij')
        patch_sizes_s = np.meshgrid(*patch_sizes_s, indexing='ij')
        positions_s = np.column_stack([axis_positions_s[axis].ravel() for axis in range(n_axis_s)])
        patch_sizes_s = np.column_stack([patch_sizes_s[axis].ravel() for axis in range(n_axis_s)])
        return positions_s, patch_sizes_s


class _SqueezeGridSampler(_CropGridSampler):
    def __init__(self, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, image_h: Optional[npt.ArrayLike] = None, spatial_first: bool = True):
        super().__init__(image_size_s=image_size_s, patch_size_s=patch_size_s, step_size_s=step_size_s, image_h=image_h, spatial_first=spatial_first)

    def compute_patches(self):
        n_axis_s = len(self.image_size_s)
        stop_s = [self.image_size_s[axis] - self.patch_size_s[axis] + 1 for axis in range(n_axis_s)]
        axis_positions_s = [np.arange(0, stop_s[axis], self.step_size_s[axis]) for axis in range(n_axis_s)]
        for axis in range(n_axis_s):
            if axis_positions_s[axis][-1] + self.patch_size_s[axis] < self.image_size_s[axis]:
                axis_positions_s[axis] = np.concatenate((axis_positions_s[axis], [axis_positions_s[axis][-1] + self.step_size_s[axis]]))
        axis_squeeze_s = [(axis_positions_s[axis][-1] + self.patch_size_s[axis]) - self.image_size_s[axis] for axis in range(n_axis_s)]  ###
        additional_offset_s = [axis_squeeze_s[axis] // (len(axis_positions_s[axis]) - 1) for axis in range(n_axis_s)]
        remainder_offset_s = [axis_squeeze_s[axis] % (len(axis_positions_s[axis]) - 1) for axis in range(n_axis_s)]
        axis_positions_s = [axis_positions_s[axis] - np.arange(0, len(axis_positions_s[axis])) * additional_offset_s[axis] for axis in range(n_axis_s)]
        for axis in range(n_axis_s):
            axis_positions_s[axis][-1] -= remainder_offset_s[axis]
        patch_sizes_s = [[self.patch_size_s[axis]] * len(axis_positions_s[axis]) for axis in range(n_axis_s)]
        axis_positions_s = np.meshgrid(*axis_positions_s, indexing='ij')
        patch_sizes_s = np.meshgrid(*patch_sizes_s, indexing='ij')
        patch_positions_s = np.column_stack([axis_positions_s[axis].ravel() for axis in range(n_axis_s)])
        patch_sizes_s = np.column_stack([patch_sizes_s[axis].ravel() for axis in range(n_axis_s)])
        return patch_positions_s, patch_sizes_s
