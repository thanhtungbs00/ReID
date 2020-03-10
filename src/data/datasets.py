import torch
import json
import cv2
import numpy as np
import random


def imread(path):
    return cv2.imread(path)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path=None, transform=None):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.filename_to_class_idx = dict(
            self.metadata["filename_to_class_idx"])
        self.classnames = list(self.metadata["classnames"])
        self.filename_list = list(self.filename_to_class_idx.keys())
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        img = imread(self.filename_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.filename_to_class_idx[filename]

    def __len__(self):
        return len(self.filename_list)

    def __str__(self):
        return json.dumps({
            "filename_to_class_idx": self.filename_to_class_idx,
            "classnames": self.classnames
        })


def train_test_split(dataset, split_rate=0.5):
    t = int(split_rate * len(dataset))
    left, right = torch.utils.data.random_split(dataset, [t, len(dataset)-t])


if __name__ == "__main__":
    dataset = ImageDataset("./dataset/processed/metadata.json")
    print(len(dataset))
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True)
    # for i in dataloader:
    #     print(i)
    #     break
