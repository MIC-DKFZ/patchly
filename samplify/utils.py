import numpy as np
import string

try:
    import torch
except:
    torch = None


class LazyArray:
    def __init__(self):
        self._data = None

    def create(self, data):
        self._data = data

    @property
    def data(self):
        if self._data is None:
            raise ValueError("LazyArray has not been initialized.")
        return self._data
    
    @property
    def shape(self):
        if self._data is None:
            raise ValueError("LazyArray has not been initialized.")
        return self._data.shape

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value


def bbox_s_to_bbox_h(bbox_s, image_h, spatial_first):
    dims_n = len(image_h.shape) - len(bbox_s[:, 0])
    if spatial_first:
        bbox_h = [index_pair for index_pair in bbox_s]
        bbox_h.extend([[None]] * dims_n)
    else:
        bbox_h = [[None]] * dims_n
        bbox_h.extend([index_pair for index_pair in bbox_s])
    return bbox_h


def data_s_to_data_h(data_size1_s, data_size2_h, spatial_first):
    dims_n = len(data_size2_h) - len(data_size1_s)
    if dims_n > 0 and spatial_first:
        dims_n = data_size2_h[len(data_size1_s):]
        data_size1_h = (*data_size1_s, *dims_n)
        return data_size1_h
    elif dims_n > 0:
        dims_n = data_size2_h[:-len(data_size1_s)]
        data_size1_h = (*dims_n, *data_size1_s)
        return data_size1_h
    else:
        return data_size1_s
    

def broadcast_to(data, target_shape, spatial_first):
    if spatial_first:
        non_spatial_dims = len(target_shape) - len(data.shape)
        data_reshaped = data
        for _ in range(non_spatial_dims):
            data_reshaped = data_reshaped[..., None]
        data_reshaped = data_reshaped.broadcast_to(target_shape)
    else:
        data_reshaped = data.broadcast_to(target_shape)
    data_reshaped = data_reshaped.copy()
    return data_reshaped


def is_overlapping(bbox1, bbox2):
    """
    Check if two N-D bounding boxes overlap.
    
    Bounding boxes are defined as [[x_start, x_end], [y_start, y_end], ...].
    
    Args:
    - box1, box2: The bounding boxes to check.

    Returns:
    - True if the boxes overlap, False otherwise.
    """

    for (start1, end1), (start2, end2) in zip(bbox1, bbox2):
        if start1 >= end2 or start2 >= end1:
            return False
    return True


def gaussian_kernel_numpy(size, sigma=1./8, dtype=None):
    """Return an N-D Gaussian kernel array."""
    sigma = size * sigma
    
    def gaussian_kernel_1d(size, sigma):
        """Return a 1D Gaussian kernel array."""
        offset = 1
        if size % 2 == 0:  # Fix for even sizes to keep kernel centered
            offset = 0
        axis = np.linspace(-size // 2 + offset, size // 2, size)
        kernel = np.exp(-axis**2 / (2 * sigma**2))
        return kernel / kernel.sum()
    
    kernels = [gaussian_kernel_1d(size_axis, sigma_axis) for size_axis, sigma_axis in zip(size, sigma)]

    chars = string.ascii_lowercase
    subscripts = ""
    for char in chars[:len(kernels)]:
        subscripts += char + ','
    subscripts = subscripts[:-1]

    kernel_nd = np.einsum(subscripts, *kernels)

    kernel_nd[kernel_nd == 0] = np.min(kernel_nd[kernel_nd != 0])

    if dtype is not None:
        kernel_nd = kernel_nd.astype(dtype)

    return kernel_nd


def gaussian_kernel_pytorch(size, sigma=1./8, device='cpu', dtype=None):
    """Return an N-D Gaussian kernel array."""
    sigma = size * sigma
    
    def gaussian_kernel_1d(size, sigma):
        """Return a 1D Gaussian kernel array."""
        offset = 1
        if size % 2 == 0:  # Fix for even sizes to keep kernel centered
            offset = 0
        axis = torch.linspace(-size // 2 + offset, size // 2, size, device=device)
        kernel = torch.exp(-axis**2 / (2 * sigma**2))
        return kernel / kernel.sum()
    
    kernels = [gaussian_kernel_1d(size_axis, sigma_axis) for size_axis, sigma_axis in zip(size, sigma)]

    chars = string.ascii_lowercase
    subscripts = ""
    for char in chars[:len(kernels)]:
        subscripts += char + ','
    subscripts = subscripts[:-1]

    kernel_nd = torch.einsum(subscripts, *kernels)

    kernel_nd[kernel_nd == 0] = torch.min(kernel_nd[kernel_nd != 0])

    if dtype is not None:
        kernel_nd = kernel_nd.to(dtype=dtype)

    return kernel_nd