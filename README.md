# Patchly

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/patchly.svg?color=green)](https://github.com/MIC-DKFZ/patchly/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/patchly.svg?color=green)](https://pypi.org/project/patchly)
[![Python Version](https://img.shields.io/pypi/pyversions/patchly.svg?color=green)](https://python.org)
![Unit Tests](https://github.com/MIC-DKFZ/patchly/actions/workflows/test_and_deploy.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/MIC-DKFZ/patchly/branch/main/graph/badge.svg)](https://codecov.io/gh/MIC-DKFZ/patchly)


Patchly is a grid sampler for N-dimensional images enabling inference and other processing steps on extremely large images. Especially for 3D images, it has been proven successfully to inference large images patch-wise in a sliding-window approach. Patchly does just that with a very simple interface to sample and aggregate images.

The main functionalities of Patchly consist of a Sampler, which samples patches from an image based on a grid, and an Aggregator, which assembles the patches back into the shape of the original image. There is a multitude of libraries providing similar functionality already. However, they tend to work only for a limited number of usage scenarios before becoming unusable. 

Patchly is the first library providing an advanced set of features for users working with sophisticated image processing pipelines requiring patch-based processing.

A complete overview of how the Sampler and Aggregator work and an in-depth explanation of the features can be found [here](OVERVIEW.md).

## Feature Summary

Patchly provides the following advanced features:
- N-dimensional image handling (1D, 2D, 3D, ...)
- Multiple border-handling strategies
- Support for any array-like images (Numpy, Tensor, Zarr, Dask, ...)
- Memory-mapped image support
- Patch overlap
- Gaussian patch averaging
- Support for images with non-spatial dimensions (channel dimension, batch dimension, ...)
- Chunk aggregation to minimize memory consumption
- ~~Numpy padding techniques~~

## Installation

You can install `patchly` via [pip](https://pypi.org/project/patchly/):

    pip install patchly

## Usage

Demonstration on how to use Patchly for sliding-window patchification and subsequent aggregation:
```python
sampler = GridSampler(image, spatial_size, patch_size, step_size)
aggregator = Aggregator(sampler, output_size)

for patch, patch_bbox in sampler:
    aggregator.append(patch, patch_bbox)

prediction = aggregator.get_output()
```

## Example

Working example for inference of a 2D RGB image with Patchly in PyTorch:
```python
import numpy as np
from patchly.sampler import GridSampler
from patchly.aggregator import Aggregator
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
sampler = GridSampler(image=np.random.random((1000, 1000, 3)), spatial_size=(1000, 1000), patch_size=(100, 100), step_size=(50, 50))
# Init dataloader
loader = DataLoader(ExampleDataset(sampler), batch_size=4, num_workers=0, shuffle=False)
# Init aggregator
aggregator = Aggregator(sampler=sampler, output_size=(8, 1000, 1000), spatial_first=False, has_batch_dim=True)

# Run inference
with torch.no_grad():
    for patch, patch_bbox in loader:
        patch_prediction = model(patch)
        aggregator.append(patch_prediction, patch_bbox)

# Finalize aggregation
prediction = aggregator.get_output()
print("Inference completed!")
print("Prediction shape: ", prediction.shape)

```

Further examples can be found in `examples`.

## License

Distributed under the terms of the [Apache Software License 2.0](http://www.apache.org/licenses/LICENSE-2.0) license,
"Patchly" is free and open source software

# Acknowledgements
<img src="https://github.com/MIC-DKFZ/patchly/raw/main/resources/HI_Logo.png" height="100px" />

<img src="https://github.com/MIC-DKFZ/patchly/raw/main/resources/dkfz_logo.png" height="100px" />

Patchly is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).


<!--- Samplify will live forever! But so will Chunky (⊙ _ ⊙ ) -->

<!---
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@(///((/.@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@&***,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*/*@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@**,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*&@@@@@@@@@@@@@@@@
@@@@@@@@@@@@&**,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,**@@@@@@@@@@@@
@@@@@@@@@@/,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*@@@@@@@@@@
@@@@@@@&*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*/@@@@@@@
@@@@@@*,,,,,,,,,,,,,,,,,,,,,,,,,,*/#%&@@@&&%(*,,,,,,,,,,,,,,,,,,,,,,,,,,,**@@@@@
@@@@%*,,,,,,,,,,,,,,,,,,,,/&@@@@@#/,,,,,,,,,*#&@@@@@%*,,,,,,,,,,,,,,,,,,,,,(@@@@
@@@(,,,,,,,,,,,,,,,,,,(@@@&*,,,,,,,,,,......,,,,****(@@@&*,,,,,,,,,,,,,,,,,,*@@@
@@&,,,,,,,,,,,,,,,,#@@@*...,,..,,,,,,,,,,,....,,,,******(@@@/,,,,,,,,,,,,,,,,/@@
@(*,,,,,,,,,,,,,*@@@*..........,,,,,,,......,,,,,,,********(@@&,,,,,,,,,,,,,,,/@
@#,,,,,,,,,,,,*@@&,...,*//////,,,,,,,,,,,,,,,,,*//////*******(@@%,,,,,,,,,,,,,*@
@*,,,,,,,,,,,#@@/...,///////////*,,,,,.,,,,,,////////////******(@@(,,,,,,,,,,,,#
@,,,,,,,,,,,%@&,....,/(@@@@@@@#//,,....,,,,,//(@@@@@@@%/********/@@%,,,,,,,,,,,(
/,,,,,,,,,,%@%,......*@@@%.#@@@%*,,....,,,,,,#@@@&,(@@@(**********&@%,,,,,,,,,,*
*,,,,,,,,,#@@,.......,*@@@@@@@*,,,.........,,,,&@@@@@@(***********/@@#,,,,,,,,,*
*,,,,,,,,*@@(...........,,,,,,,,,..,*#%#*,,.,,,,,,,,,,,************/@@*,,,,,,,,*
*,,,,,,,,/@@,,,,,..............,....(%#%%,....,,,,,,,,**************&@/,,,,,,,,*
/,,,,,,,,/@@,,,,,,,..................,&*......,,,,,,,***************&@(,,,,,,,,*
 ,,,,,,,,/@@*,,,,,,..........,.(&#(&&*,,%&(/%%,,,,,,****************@@/,,,,,,,,(
@*,,,,,,,,@@%,.,,,,,,**,,.,....,,,,,,,,,,,,,,,,,,,********///******(@@*,,,,,,,,#
@#,,,,,,,,,@@/......,,*****,,,,,,,,,,,,,,,,,,,,,******/(((/*******/@@(,,,,,,,,*@
@#*,,,,,,,,*@@%,........,,***////*,,,,,..,,,/((((((((((/*********%@@/,,,,,,,,,/@
@@&,,,,,,,,,,*@@@#,............,**///*,,/((((((/**************#@@@(,,,,,,,,,,/@@
@@@(,,,,,,,,,,,,,(&@@@@@@@@@@@@@%/*/@@#&@@**(%@@@@@@@@@@@@@@@@#*,,,,,,,,,,,,/@@@
@@@@&*,,,,,,,,,,,,,,,,,,,,,,,,,,*/(/*,,,,*(((*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,(@@@@
@@@@@@*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*#@@@@@
@@@@@@@&/,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*#@@@@@@@
@@@@@@@@@@@*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,/@@@@@@@@@@
@@@@@@@@@@@@@(/,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*(@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@/*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*#@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@%#/*,,,,,,,,,,,,,,,,,,,,,,,,,*/##@@@@@@@@@@@@@@@@@@@@@@@
                      I will watch you in your sleep ◉‿◉
-->