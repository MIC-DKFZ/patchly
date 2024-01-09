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
    def __init__(self, image: npt.ArrayLike, spatial_size: Union[Tuple, npt.ArrayLike], patch_size: Union[Tuple, npt.ArrayLike], step_size: Optional[Union[Tuple, npt.ArrayLike]] = None, 
                 spatial_first: bool = True, mode: SamplingMode = SamplingMode.SAMPLE_SQUEEZE, pad_kwargs: dict = None):
        """
        Initializes the GridSampler object with specified parameters for sampling patches from an image.

        A complete overview of how the Sampler and Aggregator work and an in-depth explanation of the features can be found in OVERVIEW.md.

        :param image: npt.ArrayLike - The image from which patches will be sampled. Can be None, if only patch bboxes are relevant.
        :param spatial_size: Union[Tuple, npt.ArrayLike] - The size of the spatial dimensions of the image.
        :param patch_size: Union[Tuple, npt.ArrayLike] - The size of the patches to be sampled.
        :param step_size: Optional[Union[Tuple, npt.ArrayLike]] - The step size between patches. Defaults to the same as patch_size if None.
        :param spatial_first: bool - Indicates whether spatial dimensions come first in the image array. Defaults to True.
        :param mode: SamplingMode - The sampling mode to use, which affects how patch borders are handled. Defaults to SamplingMode.SAMPLE_SQUEEZE.
        :param pad_kwargs: dict - Additional keyword arguments for numpy's pad function, used in certain padding modes. Defaults to None.
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

    def set_step_size(self, step_size_s: Union[Tuple, np.ndarray], patch_size_s: Union[Tuple, np.ndarray]) -> np.ndarray:
        """
        Sets the step size for patch sampling. If the step size is not provided, it defaults to the patch size.

        :param step_size_s: Union[Tuple, np.ndarray] - The desired step size for sampling patches. If None, it will default to the patch size.
        :param patch_size_s: Union[Tuple, np.ndarray] - The size of the patches to be sampled.
        :return: np.ndarray - The adjusted or default step size.
        """
        if step_size_s is None:
            step_size_s = patch_size_s
        else:
            step_size_s = np.asarray(step_size_s)
        return step_size_s

    def check_sanity(self):
        """
        Checks the sanity of the initialized GridSampler parameters. It validates the compatibility of the image, patch size, step size, and spatial dimensions. 
        Raises runtime errors if any incompatibility or inconsistency is found in the provided parameters.
        """
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
        """
        Creates an appropriate sampler based on the specified sampling mode. This method initializes different types of grid samplers like EdgeGridSampler, 
        AdaptiveGridSampler, CropGridSampler, or SqueezeGridSampler depending on the mode selected during the GridSampler initialization. 
        Raises NotImplementedError if an unsupported mode is specified.
        """
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
        """
        Pads the image based on the selected padding mode and parameters. This method adjusts the image's shape according to the specified padding strategy,
        applying numpy's pad function with the given pad_kwargs. It updates the image size and pad width attributes of the GridSampler instance.
        Raises RuntimeError if an unsupported padding mode is provided.
        """
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
        """
        Returns an iterator for the GridSampler. This method allows the GridSampler to be used in iterator contexts, enabling iteration over the sampled patches.
        """
        return self.sampler.__iter__()

    def __len__(self):
        """
        Returns the total number of patches that will be sampled by the GridSampler. This method allows users to determine the number of patches that will be generated 
        based on the initialized spatial size, patch size, and step size.
        """
        return self.sampler.__len__()

    def __getitem__(self, idx: int):
        """
        Retrieves the patch and patch location at the specified index. This method allows for direct access to a specific patch based on its index in the sequence of all patches 
        generated by the GridSampler.

        :param idx: int - The index of the patch to retrieve.
        :return: The patch and patch location at the specified index.
        """
        return self.sampler.__getitem__(idx)

    def __next__(self):
        """
        Advances the iterator and returns the next patch in the sequence. This method is part of the iterator protocol, enabling the GridSampler to be used in 
        contexts where an iterator is required, such as in a for loop.
        """
        return self.sampler.__next__()
    
    def _get_bbox(self, idx: int) -> np.ndarray:
        """
        Retrieves the bounding box coordinates of the patch at the specified index. This internal method is used to determine the spatial location of a patch within the larger image.

        :param idx: int - The index of the patch for which the bounding box is required.
        :return: np.ndarray - The bounding box coordinates of the specified patch.
        """
        return self.sampler._get_bbox(idx)


class _CropGridSampler:
    def __init__(self, image_h: npt.ArrayLike, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, spatial_first: bool = True):
        """
        Initializes the _CropGridSampler object, a subclass of GridSampler, for sampling patches using the crop sampling strategy, discarding all patches extending over the image.

        A complete overview of how the Sampler and Aggregator work and an in-depth explanation of the features can be found in OVERVIEW.md.

        :param image_h: npt.ArrayLike - The image from which patches will be sampled. Can be None, if only patch bboxes are relevant.
        :param image_size_s: np.ndarray - The size of the spatial dimensions of the image.
        :param patch_size_s: np.ndarray - The size of the patches to be sampled.
        :param step_size_s: np.ndarray - The step size between patches.        
        :param spatial_first: bool - Indicates whether spatial dimensions come first in the image array. Defaults to True.
        """
        self.image_h = image_h
        self.image_size_s = image_size_s
        self.patch_size_s = patch_size_s
        self.step_size_s = step_size_s
        self.spatial_first = spatial_first
        self.patch_positions_s, self.patch_sizes_s = self.compute_patches()

    def compute_patches(self):
        """
        Computes the positions and sizes of patches to be sampled from the image. This method calculates the grid of patches based on the image size, patch size, 
        and step size specified in the initializer of the _CropGridSampler. Discards all patches extending over the image.
        """
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
        """
        Returns an iterator for the _CropGridSampler. This method initializes the iteration process, allowing the _CropGridSampler to be used in iterator contexts,
        enabling iteration over the sampled patches.
        """
        self.index = 0
        return self

    def __len__(self):
        """
        Returns the total number of patches that will be sampled by the _CropGridSampler. This method calculates the length based on the computed patch positions, 
        allowing users to know the number of patches that will be generated for the image.
        """
        return len(self.patch_positions_s)

    def __getitem__(self, idx: int):
        """
        Retrieves the patch and patch location at the specified index from the _CropGridSampler. This method allows direct access to a specific patch, providing the sampled patch and 
        its bounding box based on the index in the sequence of patches generated.

        :param idx: int - The index of the patch to retrieve.
        :return: The patch and patch location at the specified index.
        """
        patch_bbox_s = self._get_bbox(idx)
        patch_result = self.get_patch_result(patch_bbox_s)
        return patch_result

    def __next__(self):
        """
        Advances the iterator and returns the next patch in the sequence from the _CropGridSampler. This method is part of the iterator protocol, enabling the 
        _CropGridSampler to be iterated over in contexts like a for loop, providing patches sequentially.
        """
        if self.index < self.__len__():
            output = self.__getitem__(self.index)
            self.index += 1
            return output
        else:
            raise StopIteration
        
    def _get_bbox(self, idx: int) -> np.ndarray:
        """
        Computes the bounding box for the patch at the specified index. This internal method calculates the spatial coordinates defining the area of the image 
        covered by the patch, facilitating the extraction of the specific patch.

        :param idx: int - The index of the patch for which the bounding box is needed.
        :return: np.ndarray - The bounding box coordinates for the patch at the specified index.
        """
        patch_position_s = self.patch_positions_s[idx]
        patch_bbox_s = np.zeros(len(patch_position_s) * 2, dtype=int).reshape(-1, 2)
        for axis in range(len(patch_bbox_s)):
            patch_bbox_s[axis][0] = patch_position_s[axis]
            patch_bbox_s[axis][1] = patch_position_s[axis] + self.patch_sizes_s[idx][axis]
        return patch_bbox_s

    def get_patch_result(self, patch_bbox_s: np.ndarray):
        """
        Retrieves the patch from the image based on the provided bounding box coordinates. This method extracts the specified patch from the image, handling it 
        according to the configuration of the _CropGridSampler, such as whether the image is a dictionary or a standard array.

        :param patch_bbox_s: np.ndarray - The bounding box coordinates defining the area of the patch to be extracted.
        :return: The extracted patch and its bounding box, or just the bounding box if the image is a dictionary.
        """

        if self.image_h is not None and not isinstance(self.image_h, dict):
            patch_bbox_h = utils.bbox_s_to_bbox_h(patch_bbox_s, self.image_h, self.spatial_first)
            patch_h = self.image_h[slicer(self.image_h, patch_bbox_h)]
            return patch_h, patch_bbox_s
        else:
            return patch_bbox_s


class _EdgeGridSampler(_CropGridSampler):
    def __init__(self, image_h: npt.ArrayLike, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, spatial_first: bool = True):
        """
        Initializes the _EdgeGridSampler object, a subclass of _CropGridSampler, for sampling patches using the edge sampling strategy.

        A complete overview of how the Sampler and Aggregator work and an in-depth explanation of the features can be found in OVERVIEW.md.

        :param image_h: npt.ArrayLike - The image from which patches will be sampled. Can be None, if only patch bboxes are relevant.
        :param image_size_s: np.ndarray - The size of the spatial dimensions of the image.
        :param patch_size_s: np.ndarray - The size of the patches to be sampled.
        :param step_size_s: np.ndarray - The step size between patches.
        :param spatial_first: bool - Indicates whether spatial dimensions come first in the image array. Defaults to True.
        """
        super().__init__(image_size_s=image_size_s, patch_size_s=patch_size_s, step_size_s=step_size_s, image_h=image_h, spatial_first=spatial_first)

    def compute_patches(self):
        """
        Computes the positions and sizes of patches for edge sampling. This method, specific to _EdgeGridSampler, adjusts the patch positions so that the 
        last patch in each dimension aligns with the edge of the image. It extends the functionality of compute_patches in _CropGridSampler to handle the 
        edge sampling strategy.
        """
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
    def __init__(self, image_h: npt.ArrayLike, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, spatial_first: bool = True, min_patch_size_s: np.ndarray = None):
        """
        Initializes the _AdaptiveGridSampler object, a subclass of _CropGridSampler, designed for adaptive patch sampling. This sampler adjusts the last patch 
        size to fit within the image boundaries, potentially reducing the patch size if necessary.

        A complete overview of how the Sampler and Aggregator work and an in-depth explanation of the features can be found in OVERVIEW.md.

        :param image_h: npt.ArrayLike - The image from which patches will be sampled. Can be None, if only patch bboxes are relevant.
        :param image_size_s: np.ndarray - The size of the spatial dimensions of the image.
        :param patch_size_s: np.ndarray - The size of the patches to be sampled.
        :param step_size_s: np.ndarray - The step size between patches.
        :param spatial_first: bool - Indicates whether spatial dimensions come first in the image array. Defaults to True.
        :param min_patch_size_s: np.ndarray - The minimum size for the last patch in each dimension, ensuring it fits within the image. Defaults to None.
        """
        self.min_patch_size_s = min_patch_size_s
        if self.min_patch_size_s is not None and np.any(self.min_patch_size_s > patch_size_s):
            raise RuntimeError("The minimum patch size ({}) cannot be greater than the actual patch size ({}) in one or more dimensions.".format(self.min_patch_size_s, patch_size_s))
        super().__init__(image_size_s=image_size_s, patch_size_s=patch_size_s, step_size_s=step_size_s, image_h=image_h, spatial_first=spatial_first)

    def compute_patches(self):
        """
        Computes the positions and sizes of patches for adaptive sampling. This method, specific to _AdaptiveGridSampler, modifies the size of the last patch 
        in each dimension to ensure it fits within the image boundaries. It may reduce the patch size to a specified minimum, adapting to the image dimensions 
        while maintaining patch coverage.
        """
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
    def __init__(self, image_h: npt.ArrayLike, image_size_s: np.ndarray, patch_size_s: np.ndarray, step_size_s: np.ndarray, spatial_first: bool = True):
        """
        Initializes the _SqueezeGridSampler object, a subclass of _CropGridSampler, designed for squeeze sampling strategy. This sampler adjusts the positions 
        of all patches to ensure that they fit within the image dimensions, slightly increasing overlap between patches if necessary.

        A complete overview of how the Sampler and Aggregator work and an in-depth explanation of the features can be found in OVERVIEW.md.

        :param image_h: npt.ArrayLike - The image from which patches will be sampled. Can be None, if only patch bboxes are relevant.
        :param image_size_s: np.ndarray - The size of the spatial dimensions of the image.
        :param patch_size_s: np.ndarray - The size of the patches to be sampled.
        :param step_size_s: np.ndarray - The step size between patches.
        :param spatial_first: bool - Indicates whether spatial dimensions come first in the image array. Defaults to True.
        """
        super().__init__(image_size_s=image_size_s, patch_size_s=patch_size_s, step_size_s=step_size_s, image_h=image_h, spatial_first=spatial_first)

    def compute_patches(self):
        """
        Computes the positions and sizes of patches for squeeze sampling. This method, specific to _SqueezeGridSampler, adjusts the position of each patch 
        to ensure that all patches fit within the image dimensions. It adjusts the patch positions of all patches to ensure that they fit within the image dimensions, 
        slightly increasing overlap between patches if necessary.
        """
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
