
# TODO: Process data, including extracting, re-scaling, etc.
# Intermediate data would be stored in dataset/interim
# Final data would be stored in dataset/processed
# A metadata of {filepath: classname} should be generated and stored in dataset/processed

import cv2
import os
from collections import deque
import zipfile
import shutil
import json
from multiprocessing import Pool

from src.data.datasets import ImageDataset
from src.data.samplers import RandomBalanceBatchSampler
from torch.utils.data.dataloader import DataLoader

def prepare_data(self):
    # Create training set and testing set
    training_set = ImageDataset(
        metadata_path="./dataset/processed/dummy/train/metadata.json")
    training_set, validation_set = training_set.random_split(left_rate=0.9)
    testing_set = ImageDataset(
        metadata_path="./dataset/processed/dummy/test/metadata.json")

    self.training_set = training_set
    self.validation_set = validation_set
    self.testing_set = testing_set


def train_loader(self):
    return DataLoader(
        self.training_set,
        batch_sampler=RandomBalanceBatchSampler(
            datasource=self.training_set,
            classes_count=class_per_batch,
            samples_per_class_count=samples_per_class_per_batch,
            iters_count=train_iters_per_epoch
        ))


def val_loader(self):
    return DataLoader(
        self.validation_set,
        batch_sampler=RandomBalanceBatchSampler(
            datasource=self.validation_set,
            classes_count=class_per_batch,
            samples_per_class_count=samples_per_class_per_batch,
            iters_count=val_iters_per_epoch
        ))


def test_loader(self):
    return DataLoader(
        self.testing_set,
        batch_sampler=RandomBalanceBatchSampler(
            datasource=self.testing_set,
            classes_count=class_per_batch,
            samples_per_class_count=samples_per_class_per_batch,
            iters_count=test_iters_per_epoch
        ))
if __name__ == "__main__":
    prepare_data(None)
