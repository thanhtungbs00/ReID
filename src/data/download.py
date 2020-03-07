''' This file contains downloading scripts '''

# TODO: Downloading scripts here

import os
import shutil
import zipfile

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


fileurl = "/mnt/c/dogs-vs-cats.zip"
def download():
    shutil.copy2(src=fileurl, dst="./data/raw/")

def extract():
    with zipfile.ZipFile("./data/raw/dogs-vs-cats.zip", 'r') as zip_ref:
        zip_ref.extractall("./data/interim/")
    with zipfile.ZipFile("./data/interim/train.zip", 'r') as zip_ref:
        zip_ref.extractall("./data/interim/train/")
    


if __name__ == "__main__":
    download()
    extract()
