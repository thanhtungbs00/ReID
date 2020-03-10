

# def extract():
#     with zipfile.ZipFile("./dataset/raw/dogs-vs-cats.zip", 'r') as zip_ref:
#         zip_ref.extractall("./dataset/interim/")
#     with zipfile.ZipFile("./dataset/interim/train.zip", 'r') as zip_ref:
#         zip_ref.extractall("./dataset/interim/")


# def transform(image_src, image_dst):
#     img = cv2.imread(image_src, cv2.IMREAD_COLOR)
#     if img is None:
#         return False
#     img = cv2.resize(img, (image_width, image_height))
#     os.makedirs(os.path.dirname(image_dst), exist_ok=True)
#     return cv2.imwrite(image_dst, img)


# def process():
#     src_prefix = './dataset/interim/'
#     dst_prefix = './dataset/processed/'
#     process_queue = deque(os.listdir(src_prefix))
#     processed = set()
#     while len(process_queue) > 0:
#         filename = process_queue[-1]
#         process_queue.pop()
#         if filename in processed:
#             continue
#         processed.add(filename)
#         if os.path.isdir(src_prefix+filename):
#             filename += '/'
#             for newname in os.listdir(src_prefix + filename):
#                 process_queue.append(filename + newname)
#         else:
#             _, ext = os.path.splitext(src_prefix + filename)
#             if ext in accepted_exts:
#                 transform(src_prefix + filename, dst_prefix + filename)
#                 print("Writing " + filename)


# def generate_metadata():
#     metadata = dict()
#     dst = './dataset/processed/'
#     process_queue = deque(os.listdir(dst))
#     processed = set()
#     while len(process_queue) > 0:
#         filename = process_queue[-1]
#         process_queue.pop()
#         if filename in processed:
#             continue
#         processed.add(filename)
#         if os.path.isdir(dst + filename):
#             filename += '/'
#             for newname in os.listdir(dst + filename):
#                 process_queue.append(filename + newname)
#         else:
#             _, ext = os.path.splitext(dst + filename)
#             if ext in accepted_exts:
#                 metadata[dst + filename] = 0 if "cat" in filename.lower() else 1
#     metadata = {
#         "filename_to_class_idx": metadata,
#         "classnames":  ["cat", "dog"]

#     }
#     with open(dst + "metadata.json", 'w') as file:
#         json.dump(metadata, file)


# def clean():
#     for item in os.listdir("./dataset/interim/"):
#         if os.path.isfile("./dataset/interim/" + item):
#             os.remove("./dataset/interim/" + item)
#         else:
#             shutil.rmtree("./dataset/interim/" + item)


# if __name__ == "__main__":
#     # extract()
#     # process()
#     generate_metadata()
#     # clean()
