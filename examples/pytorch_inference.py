import numpy as np
from samplify.sampler import GridSampler
from samplify.aggregator import Aggregator
from tqdm import tqdm
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
    patch_overlap = (50, 50)
    chunk_size = (500, 500)

    # Init GridSampler
    sampler = GridSampler(image=image, spatial_size=spatial_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size, spatial_first=False, mode="sample_edge")
    # Convert sampler into a PyTorch dataset
    loader = SamplerDataset(sampler)
    # Init dataloader
    loader = DataLoader(loader, batch_size=4, num_workers=2, shuffle=False, pin_memory=False)
    # Create an empty prediction passed to the aggregator
    prediction = np.zeros(spatial_size, dtype=np.uint8)
    # Init aggregator
    aggregator = Aggregator(sampler=sampler, output=prediction, weights='gaussian', softmax_dim=0)

    # Run inference
    with torch.no_grad():
        for patch, patch_indices, chunk_id in tqdm(loader):
            patch_prediction = model(patch)
            patch_prediction = patch_prediction.cpu().numpy()
            patch_indices = patch_indices.cpu().numpy()
            chunk_id = chunk_id.cpu().numpy()
            for i in range(len(patch_prediction)):
                aggregator.append(patch_prediction[i], patch_indices[i], chunk_id[i])

    # Finalize aggregation
    prediction = aggregator.get_output()
    return prediction


class SamplerDataset(Dataset):
    def __init__(self, sampler):
        self.sampler = sampler
        self.is_chunked = self.sampler.chunk_size is not None

    def __getitem__(self, idx):
        output = self.sampler.__getitem__(idx)
        if not self.is_chunked:
            patch, patch_indices = output
            patch = patch.astype(np.float32)
            return patch, patch_indices
        else:
            patch, patch_indices, chunk_id = output
            patch = patch.astype(np.float32)
            return patch, patch_indices, chunk_id

    def __len__(self):
        return len(self.sampler)


def model(x):
    y = torch.rand((x.shape[0], 8, x.shape[2], x.shape[3]))  # Batch, Class, Width, Height
    return y


if __name__ == '__main__':
    example()