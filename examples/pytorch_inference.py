import numpy as np
from samplify.sampler import GridSampler
from samplify.aggregator import Aggregator
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm


def example():
    # Init GridSampler
    sampler = GridSampler(image=np.random.random((1000, 1000, 3)), spatial_size=(1000, 1000), patch_size=(100, 100), patch_overlap=(50, 50))
    # Init dataloader
    loader = DataLoader(ExampleDataset(sampler), batch_size=4, num_workers=0, shuffle=False)
    # Init aggregator
    aggregator = Aggregator(sampler=sampler, output_size=(8, 1000, 1000), spatial_first=False)

    # Run inference
    with torch.no_grad():
        for patch, patch_bbox in tqdm(loader):
            patch_prediction = model(patch)
            for i in range(len(patch_prediction)):
                aggregator.append(patch_prediction[i].cpu().numpy(), patch_bbox[i].cpu().numpy())

    # Finalize aggregation
    prediction = aggregator.get_output()
    print("Inference completed!")
    print("Prediction shape: ", prediction.shape)


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


if __name__ == '__main__':
    example()