from torch.utils.data import DataLoader
import src
from torch.utils.data.sampler import SequentialSampler
dataset = src.data.datasets.ImageDataset(
    "./dataset/processed/dummy/train/metadata.json")

sampler = src.data.dataloaders.RandomBalanceBatchSampler(
    dataset, classes_count=3, samples_per_class_count=3, iters_count=3)

dataloader = DataLoader(dataset, batch_sampler=sampler)
for i in dataloader:
    print(i[1])
