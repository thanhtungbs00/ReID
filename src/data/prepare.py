# Prepare data
# This file is changeable due to the change of dataset's organization
#
# TODO: download data, do the pre-processing and store it to the './dataset/processed/<datasetname>/'
# The metadata.json should be generated alongside with each dataset, which has the following format:
#
#     {
#     "filename_to_class_idx": {
#         "./dataset/processed/dummy/test/truck/0009.jpg": 0,
#         "./dataset/processed/dummy/test/truck/0008.jpg": 1,
#         "./dataset/processed/dummy/test/truck/0007.jpg": 2
#     },
#     "classnames": [
#         "truck",
#         "ship",
#         "horse"
#     ]
# }
#
# Some hard-coded part such as filepath, or specific processing scripts is allowed in this file.

import json
import os
import zipfile
from collections import deque
import cv2
import multiprocessing


with open("./config.json", 'r') as f:
    config = json.load(f)
image_width = config["image_width"]
image_height = config["image_height"]
accepted_exts = config["accepted_exts"]
workers = config["workers"]


def prepare_data():

    print("Preparing data")
    print(" Downloading...")
    download()
    print(" Extracting...")
    extract()
    print(" Transforming...")
    process()
    print(" Generating metadata...")
    generate_metadata()
    print(" Cleaning...")
    clean()


def download():
    return


def extract():
    for dataset in os.listdir("./dataset/raw/"):
        dataset_name, ext = os.path.splitext(dataset)
        if ext == ".zip":
            with zipfile.ZipFile("./dataset/raw/" + dataset, 'r') as zip_ref:
                print("     Extracting \"" + dataset_name + "\"")
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
    """TODO: For each image in interim folder, if it is a sample, transform it and save into to processed folder"""
    queue = deque(os.listdir("./dataset/interim/"))
    file_list = deque([])
    src_path = "./dataset/interim/"
    dst_path = "./dataset/processed/"
    while len(queue) != 0:
        filename = queue.popleft()
        full_src_dir = src_path + filename
        if os.path.isdir(full_src_dir):
            for item in os.listdir(src_path + filename):
                queue.append(filename + "/" + item)
        else:
            _, ext = os.path.splitext(src_path + filename)
            if ext in accepted_exts:
                file_list.append(filename)

    # for item in file_list:
    #     transform((src+item, dst+item))
    pool = multiprocessing.Pool(workers)
    pool.map(transform, [(src_path+item, dst_path+item) for item in file_list])
    pool.close()


def get_classname_from_path(path):
    s = os.path.dirname(path)
    while s[-1] == '/' or s[-1] == '\\':
        s = s[:-1]
    while '/' in s or '\\' in s:
        s = s[1:]
    return s.lower()


def generate_metadata():
    # TODO: Sửa cái này lại
    for dataset in os.listdir("./dataset/processed/"):
        if os.path.isfile("./dataset/processed/" + dataset):
            continue
        train_filename_to_class_idx = dict()
        test_filename_to_class_idx = dict()
        classname_to_class_idx = dict()

        queue = deque()
        file_list = deque([])
        train_path = "./dataset/processed/" + dataset + "/train/"
        test_path = "./dataset/processed/" + dataset + "/test/"

        queue = deque(os.listdir(train_path))
        while len(queue) != 0:
            filename = queue.popleft()
            full_src_dir = train_path + filename
            if os.path.isdir(full_src_dir):
                for item in os.listdir(train_path+filename):
                    queue.append(filename + "/" + item)
            else:
                _, ext = os.path.splitext(train_path + filename)
                if ext in accepted_exts:
                    classname = get_classname_from_path(train_path + filename)
                    if classname not in classname_to_class_idx:
                        classname_to_class_idx[classname] = len(
                            classname_to_class_idx)
                    train_filename_to_class_idx[train_path +
                                                filename] = classname_to_class_idx[classname]
        queue = deque(os.listdir(test_path))
        while len(queue) != 0:
            filename = queue.popleft()
            full_src_dir = test_path + filename
            if os.path.isdir(full_src_dir):
                for item in os.listdir(test_path+filename):
                    queue.append(filename + "/" + item)
            else:
                _, ext = os.path.splitext(test_path + filename)
                if ext in accepted_exts:
                    classname = get_classname_from_path(test_path + filename)
                    if classname not in classname_to_class_idx:
                        classname_to_class_idx[classname] = len(
                            classname_to_class_idx)
                    test_filename_to_class_idx[test_path +
                                               filename] = classname_to_class_idx[classname]
        classnames = list(classname_to_class_idx.keys())
        classnames = sorted(
            classnames, key=lambda x: classname_to_class_idx[x])
        metadata = {
            "train_filename_to_class_idx": train_filename_to_class_idx,
            "test_filename_to_class_idx": test_filename_to_class_idx,
            "classnames": classnames}
        with open("./dataset/processed/" + dataset + "/metadata.json", 'w') as f:
            json.dump(metadata, f)


def clean():
    for item in os.listdir("./dataset/interim/"):
        if os.path.isfile("./dataset/interim/" + item):
            os.remove("./dataset/interim/" + item)
        else:
            shutil.rmtree("./dataset/interim/" + item)


if __name__ == "__main__":
    # prepare_data()
    # extract()
    process()
    generate_metadata()
