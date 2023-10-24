import numpy as np
from patchly import GridSampler, Aggregator
from torch.utils.data import DataLoader, Dataset
import torch


def example():
    """
    Example on how tu run chunked inference with minimal memory consumption. No memory exploding softmax predictions.
    """
    # Init image
    image = np.random.random((3, 1000, 1000))  # Channel, Width, Height
    spatial_size = image.shape[-2:]
    patch_size = (100, 100)
    step_size = (50, 50)
    chunk_size = (500, 500)

    # Init GridSampler
    sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, step_size=step_size, spatial_first=False)
    # Convert sampler into a PyTorch dataset
    loader = SamplerDataset(sampler)
    # Init dataloader
    loader = DataLoader(loader, batch_size=4, num_workers=2, shuffle=False, pin_memory=False)
    # Create an empty prediction passed to the aggregator
    prediction = np.zeros(spatial_size, dtype=np.uint8)
    # Init aggregator
    aggregator = Aggregator(sampler=sampler, output=prediction, chunk_size=chunk_size, weights='gaussian', softmax_dim=0, spatial_first=False, has_batch_dim=True)

    # Run inference
    with torch.no_grad():
        for patch, patch_bbox in loader:
            patch_prediction = model(patch)
            aggregator.append(patch_prediction, patch_bbox)

    # Finalize aggregation
    prediction = aggregator.get_output()
    return prediction


class SamplerDataset(Dataset):
    def __init__(self, sampler):
        self.sampler = sampler

    def __getitem__(self, idx):
        return self.sampler.__getitem__(idx)

    def __len__(self):
        return len(self.sampler)


def model(x):
    y = torch.rand((x.shape[0], 8, x.shape[2], x.shape[3]))  # Batch, Class, Width, Height
    return y


if __name__ == '__main__':
    example()