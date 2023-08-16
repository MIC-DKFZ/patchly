import numpy as np


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
        bbox_h = [index_pair.tolist() for index_pair in bbox_s]
        bbox_h.extend([[None]] * dims_n)
    else:
        bbox_h = [[None]] * dims_n
        bbox_h.extend([index_pair.tolist() for index_pair in bbox_s])
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
            data_reshaped = data_reshaped[..., np.newaxis]
        data_reshaped = np.broadcast_to(data_reshaped, target_shape)
    else:
        data_reshaped = np.broadcast_to(data, target_shape)
    data_reshaped = np.copy(data_reshaped)
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