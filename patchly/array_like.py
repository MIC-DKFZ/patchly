import numpy as np
from abc import ABC, abstractmethod
from patchly.utils import gaussian_kernel_numpy, gaussian_kernel_pytorch

try:
    import torch
except:
    torch = None

try:
    import zarr
    TYPE_ZARR_ARRAY = zarr.core.Array if hasattr(zarr.core, "Array") else zarr.core.array.Array
except:
    zarr = None
    TYPE_ZARR_ARRAY = None


def create_array_like(array_type, data, device=None):
    if array_type == np.ndarray or array_type is None:
        return NumpyArray(data, device)
    elif zarr is not None and array_type == TYPE_ZARR_ARRAY:
        return NumpyArray(data, device)
    elif torch is not None and array_type == torch.Tensor:
        if data is not None:
            data = torch.tensor(data)
        return TensorArray(data, device)
    else:
        raise RuntimeError("Given ArrayLike ({}) is not supported.".format(array_type))


class ArrayLike(ABC):
    def __init__(self, data, device=None):
        if isinstance(data, ArrayLike):
            data = data.data
        self.data = data
        self.device = device

    @property
    def shape(self):
        return self.data.shape
    
    def __getitem__(self, item):
        return create_array_like(type(self.data), self.data[item], device=self.device)

    def __setitem__(self, key, value):
        if isinstance(value, ArrayLike):
            value = value.data
        self.data[key] = value

    def __len__(self):
        return len(self.data)
    
    def __add__(self, other):
        return create_array_like(type(self.data), self.data.__add__(other.data), device=self.device)
    
    def __sub__(self, other):
        return create_array_like(type(self.data), self.data.__sub__(other.data), device=self.device)
    
    def __mul__(self, other):
        return create_array_like(type(self.data), self.data.__mul__(other.data), device=self.device)
    
    def __truediv__(self, other):
        return create_array_like(type(self.data), self.data.__truediv__(other.data), device=self.device)
    
    def __iadd__(self, other):
        return create_array_like(type(self.data), self.data.__iadd__(other.data), device=self.device)
    
    def __isub__(self, other):
        return create_array_like(type(self.data), self.data.__isub__(other.data), device=self.device)
    
    def __imul__(self, other):
        return create_array_like(type(self.data), self.data.__imul__(other.data), device=self.device)
    
    def __itruediv__(self, other):
        return create_array_like(type(self.data), self.data.__itruediv__(other.data), device=self.device)
    
    @abstractmethod
    def create_zeros(self, shape, dtype=None):
        pass

    @abstractmethod
    def create_ones(self, shape, dtype=None):
        pass

    @abstractmethod
    def create_gaussian_kernel(self, shape, sigma=1./8, dtype=None):
        pass
    
    @abstractmethod
    def min(self, axis=None):
        pass

    @abstractmethod
    def broadcast_to(self, target_shape):
        pass

    @abstractmethod
    def copy(self):
        pass

    @property
    @abstractmethod    
    def dtype(self):
        pass

    @abstractmethod    
    def astype(self, dtype):
        pass

    @abstractmethod    
    def nan_to_num(self):
        pass

    @abstractmethod    
    def argmax(self, axis=None):
        pass

    @abstractmethod    
    def to(self, device):
        pass


class NumpyArray(ArrayLike):
    def __init__(self, data, device=None):
        super().__init__(data, device)

        if self.data is not None and not (isinstance(self.data, np.ndarray) or isinstance(self.data, zarr.core.Array)):
            raise RuntimeError("Given data is not of type np.ndarray but of type {}".format(type(self.data)))

    def create_zeros(self, shape, dtype=None):
        self.data = np.zeros(shape, dtype)
        return self

    def create_ones(self, shape, dtype=None):
        self.data = np.ones(shape, dtype)
        return self
    
    def create_gaussian_kernel(self, shape, sigma=1./8, dtype=None):
        self.data = gaussian_kernel_numpy(shape, sigma, dtype)
        return self
    
    def min(self, axis=None):
        return self.data.min(axis)

    def broadcast_to(self, target_shape):
        return NumpyArray(np.broadcast_to(self.data, target_shape))

    def copy(self):
        return NumpyArray(np.copy(self.data))

    @property  
    def dtype(self):
        return self.data.dtype
 
    def astype(self, dtype):
        return NumpyArray(self.data.astype(dtype))
   
    def nan_to_num(self):
        return NumpyArray(np.nan_to_num(self.data))
   
    def argmax(self, axis=None):
        return NumpyArray(self.data.argmax(axis))
 
    def to(self, device):
        return self


class TensorArray(ArrayLike):
    def __init__(self, data, device=None):
        super().__init__(data, device)

        if self.data is not None and not isinstance(self.data, torch.Tensor):
            raise RuntimeError("Given data is not of type torch.Tensor but of type {}".format(type(self.data)))
        
        self.dtype_map = {
            np.float16: torch.float16,
            "float16": torch.float16,
            np.float32: torch.float32,
            "float32": torch.float32,
            np.float64: torch.float64, 
            "float64": torch.float64,
            np.uint8: torch.uint8,
            "int8": torch.int8,
            np.uint16: torch.uint8,            
            "int16": torch.int16,
            np.int32: torch.int32,
            "int32": torch.int32,
            np.int64: torch.int64,            
            "int64": torch.int64,
        }

    def create_zeros(self, shape, dtype=None):
        self.data = torch.zeros(tuple(shape), dtype=self._to_torch_dtype(dtype), device=self.device)
        return self

    def create_ones(self, shape, dtype=None):
        self.data = torch.ones(tuple(shape), dtype=self._to_torch_dtype(dtype), device=self.device)
        return self
    
    def create_gaussian_kernel(self, shape, sigma=1./8, dtype=None):
        self.data = gaussian_kernel_pytorch(shape, sigma, self.device, self._to_torch_dtype(dtype))
        return self
    
    def min(self, axis=None):
        return self.data.min(axis)

    def broadcast_to(self, target_shape):
        return TensorArray(torch.broadcast_to(self.data, target_shape), self.device)

    def copy(self):
        return TensorArray(torch.clone(self.data), self.device)

    @property  
    def dtype(self):
        return self.data.dtype
 
    def astype(self, dtype):
        return TensorArray(self.data.to(dtype=self._to_torch_dtype(dtype)), self.device)
   
    def nan_to_num(self):
        return TensorArray(torch.nan_to_num(self.data), self.device)
   
    def argmax(self, axis=None):
        return TensorArray(torch.argmax(self.data, axis), self.device)
 
    def to(self, device):
        return TensorArray(self.data, device)
    
    def _to_torch_dtype(self, dtype):
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            dtype = np.dtype(dtype).type
        if isinstance(dtype, np.dtype):
            dtype = str(dtype)

        torch_dtype = self.dtype_map.get(dtype)

        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return torch_dtype