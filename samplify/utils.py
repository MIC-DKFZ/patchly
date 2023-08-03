def add_non_spatial_indices(image, patch_indices, spatial_size, spatial_first):
    non_spatial_dims = len(image.shape) - len(spatial_size)
    if spatial_first:
        slices = [index_pair.tolist() for index_pair in patch_indices]
        slices.extend([[None]] * non_spatial_dims)
    else:
        slices = [[None]] * non_spatial_dims
        slices.extend([index_pair.tolist() for index_pair in patch_indices])
    return slices