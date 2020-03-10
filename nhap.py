from src.data import ImageDataset

a = ImageDataset("./dataset/processed/metadata.json")
print(a[0])
