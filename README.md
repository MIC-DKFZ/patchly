# Samplify

Samplify is a grid sampler for N-dimensional images enabling inference and other processing steps on extremely large images. Especially for 3D images, it has been proven successfully to inference large images patch-wise in a sliding-window approach. Samplify does just that with a very simple interface to sample and aggregate images.

## Features

Samplify supports:
- N-dimensional images (1D, 2D, 3D, ...)
- Sampling and aggregation of images
- Any array-like images (Numpy, Tensor, Zarr, Dask, ...)
- Memory-mapped images
- Patch overlap (here referred to as patch offset)
- All numpy padding techniques
- Images with non-spatial dimensions (color dimension, batch dimension, etc)
- Chunk sampling to minimize memory consumption

## Installation

<!-- You can install `samplify` via [pip]:

    pip install samplify -->

To install Samplify, you'll first need to clone the repository to your local machine.

1. Open your terminal and clone the repository:
    ```cmd
    git clone https://github.com/MIC-DKFZ/samplify.git
    ```
2. Navigate into the cloned repository:
    ```cmd
    cd samplify
    ```
3. Install the project using pip:
    ```cmd
    pip install .
    ```

## Usage

Demonstration on how to use Samplify for sliding-window patchification and subsequent aggregation:
```python
sampler = GridSampler(spatial_size, patch_size, patch_offset, image)
aggregator = Aggregator(sampler, output_size)

for patch, patch_bbox in sampler:
    aggregator.append(patch, patch_bbox)

prediction = aggregator.get_output()
```

## Example

Working example for inference of a 2D RGB image with Samplify in PyTorch:
```python
import numpy as np
from samplify.sampler import GridSampler
from samplify.aggregator import Aggregator
from torch.utils.data import DataLoader, Dataset
import torch

class ExampleDataset(Dataset):
    def __init__(self, sampler):
        self.sampler = sampler

    def __getitem__(self, idx):
        # Get patch
        patch, patch_bbox = self.sampler.__getitem__(idx)
        # Preprocess patch
        patch = patch.transpose(2, 0, 1)
        return patch, patch_bbox

    def __len__(self):
        return len(self.sampler)

def model(x):
    y = torch.rand((x.shape[0], 8, x.shape[2], x.shape[3]))  # Batch, Class, Width, Height
    return y

# Init GridSampler
sampler = GridSampler(image=np.random.random((1000, 1000, 3)), spatial_size=(1000, 1000), patch_size=(100, 100), patch_offset=(50, 50))
# Init dataloader
loader = DataLoader(ExampleDataset(sampler), batch_size=4, num_workers=0, shuffle=False)
# Init aggregator
aggregator = Aggregator(sampler=sampler, output_size=(8, 1000, 1000), spatial_first=False)

# Run inference
with torch.no_grad():
    for patch, patch_bbox in loader:
        patch_prediction = model(patch)
        for i in range(len(patch_prediction)):
            aggregator.append(patch_prediction[i].cpu().numpy(), patch_bbox[i].cpu().numpy())

# Finalize aggregation
prediction = aggregator.get_output()
print("Inference completed!")
print("Prediction shape: ", prediction.shape)

```

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"Samplify" is free and open source software

# Acknowledgements
<img src="HI_Logo.png" height="100px" />

<img src="dkfz_logo.png" height="100px" />

Samplify is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).


