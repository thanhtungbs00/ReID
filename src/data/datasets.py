import torch
import json
import cv2
import numpy as np
import random


# Configurations

def imread(path):
    img = cv2.imread(path)
    img = np.transpose(img, axes=(2, 0, 1))
    return img/256.0


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path=None, transform=None):
        if metadata_path is not None:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.filename_to_class_idx = dict(
                self.metadata["filename_to_class_idx"])
            self.classnames = list(self.metadata["classnames"])
            self.filename_list = list(self.filename_to_class_idx.keys())
            self.transform = transform

    def random_split(self, left_rate=0.5):
        filenames = [x for x in self.filename_list]
        random.shuffle(filenames)
        left_count = int(len(filenames)*left_rate)
        filenames_left, filenames_right = filenames[:
                                                    left_count], filenames[left_count:]
        filename_to_classidx_left = dict()
        for filename in filenames_left:
            filename_to_classidx_left[filename] = self.filename_to_class_idx[filename]
        filename_to_classidx_right = dict()
        for filename in filenames_right:
            filename_to_classidx_right[filename] = self.filename_to_class_idx[filename]

        left_dataset = ImageDataset()
        left_dataset.filename_to_class_idx = filename_to_classidx_left
        left_dataset.classnames = self.classnames
        left_dataset.filename_list = filenames_left
        left_dataset.transform = self.transform

        right_dataset = ImageDataset()
        right_dataset.filename_to_class_idx = filename_to_classidx_right
        right_dataset.classnames = self.classnames
        right_dataset.filename_list = filenames_right
        right_dataset.transform = self.transform

        return left_dataset, right_dataset

    def random_split_per_class(self, left_rate=0.5):
        classes = list(set(self.filename_to_class_idx.values()))
        random.shuffle(classes)
        left_index = int(left_rate*len(classes))
        left_classes, right_classes = classes[:left_index], classes[left_index:]

        filename_to_classidx_left = dict()
        filename_to_classidx_right = dict()
        for filename in self.filename_to_class_idx:
            if self.filename_to_class_idx[filename] in left_classes:
                filename_to_classidx_left[filename] = self.filename_to_class_idx[filename]
            else:
                filename_to_classidx_right[filename] = self.filename_to_class_idx[filename]

        left_dataset = ImageDataset()
        left_dataset.filename_to_class_idx = filename_to_classidx_left
        left_dataset.classnames = self.classnames
        left_dataset.filename_list = list(filename_to_classidx_left.keys())
        left_dataset.transform = self.transform

        right_dataset = ImageDataset()
        right_dataset.filename_to_class_idx = filename_to_classidx_right
        right_dataset.classnames = self.classnames
        right_dataset.filename_list = list(filename_to_classidx_right.keys())
        right_dataset.transform = self.transform

        return left_dataset, right_dataset

    def __getitem__(self, idx):
        if type(idx) is int:
            filename = self.filename_list[idx]
            img = imread(filename)
            if img is None:
                return None
            if self.transform is not None:
                img = self.transform(img)
            return img, self.filename_to_class_idx[filename]

        elif type(idx) is tuple or type(idx) is list:
            classidx, itemidx = idx
            subidx = [
                x for x in self.filename_to_class_idx if self.filename_to_class_idx[x] == classidx]
            if len(subidx) == 0:
                raise IndexError("Class index not found in the dataset")
            # print(subidx)
            if itemidx >= len(subidx):
                raise IndexError(
                    "Class doesn't have enough samples for index " + str(itemidx))
            filename = subidx[itemidx]
            img = imread(filename)
            if img is None:
                return None
            if self.transform is not None:
                img = self.transform(img)
            return img, classidx
        else:
            raise BaseException("Type mismatch in index for __getitem__, expect 'int' or 2-tuple instead of" + str(type(idx)))

    def __len__(self):
        return len(self.filename_list)

    def __str__(self):
        return json.dumps({
            "filename_to_class_idx": self.filename_to_class_idx,
            "classnames": self.classnames
        })

    def classes_count(self):
        return len(set(self.filename_to_class_idx.values()))

    def get_random_class_idxs(self, classes_count):
        classes = set(self.filename_to_class_idx.values())
        return random.sample(classes, classes_count)

    def samples_count(self, class_idx):
        return len([x for x in self.filename_to_class_idx if self.filename_to_class_idx[x] == class_idx])


def train_test_split(dataset, split_rate=0.5):
    t = int(split_rate * len(dataset))
    left, right = torch.utils.data.random_split(dataset, [t, len(dataset)-t])


if __name__ == "__main__":
    pass
