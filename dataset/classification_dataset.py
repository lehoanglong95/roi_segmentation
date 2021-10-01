import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import albumentations as albu
import torch
import torch.nn as nn
from torchvision.transforms import transforms

class ClassificationDataset(Dataset):

    def __init__(self, csv_file, mask_file, old_dir=None, new_dir=None, augmentation=None, preprocessing_fn=None):
        self.old_dir = old_dir
        self.new_dir = new_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing_fn
        df = pd.read_csv(csv_file)
        self.image_folders = df.image.values
        self.mask_folders = df[mask_file].values

    def __getitem__(self, i):
        if self.old_dir and self.new_dir:
            image_folder = self.image_folders[i].replace(self.old_dir, self.new_dir)
            mask_folder = self.mask_folders[i].replace(self.old_dir, self.new_dir)
        else:
            image_folder = self.image_folders[i]
            mask_folder = self.mask_folders[i]
        image = cv2.imread(image_folder)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_folder)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        if len(np.unique(mask > 0)) == 2:
            label = np.array([0.0, 1.0])
        else:
            label = np.array([1.0, 0.0])
        return image, label

    def __len__(self):
        return len(self.image_folders)


def torch_transform(image, mask):
    image = image.astype("float32")
    mask = mask.transpose(2, 0, 1)
    print(image.shape)
    ttf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = ttf(image)
    print(image.shape)
    return {"image": image, "mask": transforms.ToTensor()(mask)}

if __name__ == '__main__':
    dataset = ClassificationDataset("./pancrea_training_dataset.csv", "mask")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    import matplotlib.pyplot as plt
    for image, mask , label in dataloader:
        plt.imshow(mask[0])
        plt.show()
        print(label)