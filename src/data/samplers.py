import torch
import torchvision
import random
from src.data.datasets import ImageDataset


class RandomBalanceBatchSampler(torch.utils.data.Sampler):
    def __init__(self, datasource: ImageDataset, classes_count=1, samples_per_class_count=1, iters_count=1):
        self.datasource = datasource
        self.classes_count = classes_count
        if self.classes_count > self.datasource.classes_count():
            raise IndexError(
                "There are only " + str(self.datasource.classes_count()) + " classes in the dataset, cannot get " + str(self.classes_count) + " classes")
        self.samples_per_class_count = samples_per_class_count
        self.iters_count = iters_count

    def __len__(self):
        return 1

    def __iter__(self):
        for _ in range(self.iters_count):
            class_idxs = self.datasource.get_random_class_idxs(
                self.classes_count)
            batch = []
            for classid in class_idxs:
                for _ in range(self.samples_per_class_count):
                    idx = random.randint(
                        0, self.datasource.samples_count(classid)-1)
                    batch.append((classid, idx))
            yield batch
