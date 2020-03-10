
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
# Configurations

image_width = 32
image_height = 32
accepted_exts = ['.jpg', '.jpeg', '.png']
class_per_batch = 4
samples_per_class_per_batch = 4
# Main method


def prepare_data(self):
    print("Preparing data")
    print("Downloading...")
    download()
    print("Extracting...")
    extract()
    print("Transforming...")
    process()
    print("Generating metadata...")
    generate_metadata()
    print("Cleaning...")
    clean()

    # Create training set and testing set
    training_set = ImageDataset(
        metadata_path="./dataset/processed/dummy/train/metadata.json")
    training_set, validation_set = training_set.random_split(left_rate=0.9)
    testing_set = ImageDataset(
        metadata_path="./dataset/processed/dummy/test/metadata.json")

    self.training_set = training_set
    self.validation_set = validation_set
    self.testing_set = testing_set


def train_loader(self: Classifier):
    return DataLoader(
        self.training_set,
        batch_sampler=RandomBalanceBatchSampler(
            datasource=self.training_set,
            classes_count=class_per_batch,
            samples_per_class_count=samples_per_class_per_batch
        ))


def val_loader(self):
    return DataLoader(
        self.validation_set,
        batch_sampler=RandomBalanceBatchSampler(
            datasource=self.validation_set,
            classes_count=class_per_batch,
            samples_per_class_count=samples_per_class_per_batch
        ))


def test_loader(self):
    return DataLoader(
        self.testing_set,
        batch_sampler=RandomBalanceBatchSampler(
            datasource=self.testing_set,
            classes_count=class_per_batch,
            samples_per_class_count=samples_per_class_per_batch
        ))


def download():
    return


def extract():
    with zipfile.ZipFile("./dataset/raw/dummy.zip", 'r') as zip_ref:
        zip_ref.extractall("./dataset/interim/")


def transform(args):
    image_src, image_dst = args
    img = cv2.imread(image_src, cv2.IMREAD_COLOR)
    if img is None:
        return False
    img = cv2.resize(img, (image_width, image_height))
    os.makedirs(os.path.dirname(image_dst), exist_ok=True)
    return cv2.imwrite(image_dst, img)


def process():
    src_prefix = './dataset/interim/'
    dst_prefix = './dataset/processed/'
    process_queue = deque(os.listdir(src_prefix))
    processed = set()
    process_name_list = []
    while len(process_queue) > 0:
        filename = process_queue[-1]
        process_queue.pop()
        if filename in processed:
            continue
        processed.add(filename)
        if os.path.isdir(src_prefix+filename):
            filename += '/'
            for newname in os.listdir(src_prefix + filename):
                process_queue.append(filename + newname)
        else:
            _, ext = os.path.splitext(src_prefix + filename)
            if ext in accepted_exts:
                process_name_list.append((
                    src_prefix + filename, dst_prefix + filename))
    for item in process_name_list:
        transform(item)


def get_classname_from_path(path):
    s = os.path.dirname(path)
    while s[-1] == '/' or s[-1] == '\\':
        s = s[:-1]
    while '/' in s or '\\' in s:
        s = s[1:]
    return s.lower()


def generate_metadata():
    # TODO: Sửa cái này lại
    training_src = "./dataset/processed/dummy/train/"
    training_dst = "./dataset/processed/dummy/train/"
    testing_src = "./dataset/processed/dummy/test/"
    testing_dst = "./dataset/processed/dummy/test/"
    train_filename_to_class_idx = dict()
    test_filename_to_class_idx = dict()
    classname_to_class_idx = dict()
    for src, dst, f2c_dict in [(training_src, training_dst, train_filename_to_class_idx), (testing_src, testing_dst, test_filename_to_class_idx)]:
        process_queue = deque(os.listdir(dst))
        while len(process_queue) > 0:
            filename = process_queue[-1]
            process_queue.pop()
            if os.path.isdir(dst + filename):
                filename += '/'
                for newname in os.listdir(dst + filename):
                    process_queue.append(filename + newname)
            else:
                _, ext = os.path.splitext(dst + filename)
                if ext in accepted_exts:
                    classname = get_classname_from_path(dst+filename)
                    if classname not in classname_to_class_idx:
                        classname_to_class_idx[classname] = len(
                            classname_to_class_idx)
                    f2c_dict[dst+filename] = classname_to_class_idx[classname]
        classnames = [x[0] for x in sorted(
            classname_to_class_idx.items(), key=lambda x: x[1])]
        train_metadata = {
            "filename_to_class_idx": train_filename_to_class_idx,
            "classnames":  classnames
        }
        test_metadata = {
            "filename_to_class_idx": test_filename_to_class_idx,
            "classnames":  classnames
        }

        with open(training_dst + "metadata.json", 'w') as file:
            json.dump(train_metadata, file)
        with open(testing_dst + "metadata.json", 'w') as file:
            json.dump(test_metadata, file)


def clean():
    for item in os.listdir("./dataset/interim/"):
        if os.path.isfile("./dataset/interim/" + item):
            os.remove("./dataset/interim/" + item)
        else:
            shutil.rmtree("./dataset/interim/" + item)


if __name__ == "__main__":
    prepare_data(None)
