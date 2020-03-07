
import os
import json


datapath = "./data/processed/"
metadatapath = "./data/processed/metadata.json"
support_exts = ['.jpg', '.jpeg', '.png']

metadata = dict()

# In this case, each subfolder contains images belonged to a specific class
for classname in os.listdir(datapath):
    if os.path.isdir(datapath + "/" + classname):
        for filename in os.listdir(datapath + '/' + classname):
            _, file_ext = os.path.splitext(
                datapath + '/' + classname + '/' + filename)
            if file_ext in support_exts:
                metadata[datapath + '/' + classname +
                         '/' + filename] = classname

with open(metadatapath, 'w') as fp:
    json.dump(metadata, fp)
