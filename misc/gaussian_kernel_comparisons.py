from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import time
import string
import torch


def gaussian_kernel_scipy(size, sigma=1./8, dtype=None):
    sigma = size * sigma
    center_coords = size // 2
    kernel_nd = np.zeros(size)
    kernel_nd[tuple(center_coords)] = 1
    kernel_nd = gaussian_filter(kernel_nd, sigma, 0, mode='constant', cval=0)
    kernel_nd[kernel_nd == 0] = np.min(kernel_nd[kernel_nd != 0])

    if dtype is not None:
        kernel_nd = kernel_nd.astype(dtype)

    return kernel_nd


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


if __name__ == '__main__':
    num_dims = 2
    power = 5
    offset = 0
    size = np.array([3**power+offset] * num_dims)
    # size = np.array([512] * 2)
    print(size)

    start_time = time.time()
    weights_scipy = gaussian_kernel_scipy(size, dtype=np.float32)
    print("Runtime Scipy: {}s".format(time.time() - start_time))
    start_time = time.time()
    weights_numpy = gaussian_kernel_numpy(size, dtype=np.float32)
    print("Runtime Numpy: {}s".format(time.time() - start_time))
    start_time = time.time()
    weights_pytorch = gaussian_kernel_pytorch(size, dtype=torch.float32)
    print("Runtime Pytorch: {}s".format(time.time() - start_time))
    weights_pytorch = np.asarray(weights_pytorch.cpu())

    print("Scipy vs Numpy")
    print("- Equal?", np.array_equal(weights_scipy, weights_numpy))
    print("- Close?", np.allclose(weights_scipy, weights_numpy, rtol=1.e-4))
    print("- Max diff", np.max(np.abs(weights_scipy - weights_numpy)))
    print("- Sum diff", np.sum(np.abs(weights_scipy - weights_numpy)))

    print("Scipy vs Pytorch")
    print("- Equal?", np.array_equal(weights_scipy, weights_pytorch))
    print("- Close?", np.allclose(weights_scipy, weights_pytorch, rtol=1.e-4))
    print("- Max diff", np.max(np.abs(weights_scipy - weights_pytorch)))
    print("- Sum diff", np.sum(np.abs(weights_scipy - weights_pytorch)))

    if num_dims == 2:
        fig, axes = plt.subplots(5)
        axes[0].imshow(weights_scipy)
        axes[0].set_title("Scipy")
        axes[1].imshow(weights_numpy)
        axes[1].set_title("Numpy")
        axes[2].imshow(weights_pytorch)
        axes[2].set_title("Pytorch")
        axes[3].imshow(np.abs(weights_scipy - weights_numpy))
        axes[3].set_title("Scipy vs Numpy")
        axes[4].imshow(np.abs(weights_scipy - weights_pytorch))
        axes[4].set_title("Scipy vs Pytorch")
        plt.show()