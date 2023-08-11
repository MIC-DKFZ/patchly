import numpy as np


def add_non_spatial_indices(image, patch_indices, spatial_size, spatial_first):
    non_spatial_dims = len(image.shape) - len(spatial_size)
    if spatial_first:
        slices = [index_pair.tolist() for index_pair in patch_indices]
        slices.extend([[None]] * non_spatial_dims)
    else:
        slices = [[None]] * non_spatial_dims
        slices.extend([index_pair.tolist() for index_pair in patch_indices])
    return slices


def add_non_spatial_dims(spatial_data_size_a, data_size_b, spatial_size, spatial_first):
    non_spatial_dims = len(data_size_b) - len(spatial_size)
    if non_spatial_dims > 0 and spatial_first:
        non_spatial_dims = data_size_b[len(spatial_size):]
        data_size_a = (*spatial_data_size_a, *non_spatial_dims)
        return data_size_a
    elif non_spatial_dims > 0:
        non_spatial_dims = data_size_b[:-len(spatial_size)]
        data_size_a = (*non_spatial_dims, *spatial_data_size_a)
        return data_size_a
    else:
        return spatial_data_size_a
    

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