import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import re
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision.transforms import transforms

class NewDataset(Dataset):

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
        # if "pancrea" in image_folder:
        #     if len(np.unique(mask > 0)) == 2:
        #         label = np.array([[0, 1]])
        #     else:
        #         label = np.array([[1, 0]])
        #     return image, mask, label
        return image, mask, image_folder, mask_folder

    def __len__(self):
        return len(self.image_folders)

def get_training_augmentation():
    train_transform = [

        albu.PadIfNeeded(min_height=640, min_width=640, always_apply=True, border_mode=0),
        albu.CenterCrop(height=640, width=640),
        albu.RandomCrop(height=640, width=640, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)
#
#
# def thresholding_inclusive(img_list, submission=False):
#     levels = np.linspace(0, 0.9, 10)
#     mean_slice = np.mean(np.asarray(img_list), axis=0)
#     if submission:
#         mean_slice[mean_slice >= 0.5] = 1
#
#     if len(mean_slice.shape) == 3:  # kidney dataset has an extra dim
#         mean_slice = np.squeeze(mean_slice)
#     mean_slice = np.floor(mean_slice * 10)
#     return [(mean_slice >= level).astype(int) for level in range(1, 10)]

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
    import matplotlib.pyplot as plt
    # for threshold in np.linspace(0.1, 0.9, 9):
    #     dataset = NewDataset("./training_dataset_task1.csv_per_threshold.csv", f"mask_{threshold}", augmentation=get_training_augmentation(),
    #                          preprocessing_fn=torch_transform)
    #     train_loader_task1 = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    #     for image, mask in train_loader_task1:
    #         f, axes = plt.subplots(1, 2)
    #         axes[0].imshow(image[0])
    #         axes[1].imshow(mask[0,:,:,0])
    #         plt.show()
    # model = smp.DeepLabV3Plus(
    #     encoder_name="efficientnet-b7",
    #     encoder_weights="imagenet",
    #     classes=1,
    #     activation="sigmoid",
    #     aux_params={"classes": 2, "dropout": 0.5, "activation": None}
    # )
    #a = torch.rand((3, 3, 640, 640))
    #mask, label = model(a)
    #print(mask.shape)
    #print(label.shape)
    # loss = nn.BCEWithLogitsLoss(weight=torch.Tensor([0.1,0.9]))
    for file in ["pancrea_training_dataset.csv"]:
    # for file in ["./training_dataset_task1.csv", "./training_dataset_task2.csv", "./training_dataset_task3.csv",
    #              "./val_dataset_task1.csv", "./val_dataset_task2.csv", "./val_dataset_task3.csv"]:
    #     if "task1" in file:
        task = "task1"
    #     elif "task2" in file:
    #         task = "task2"
    #     elif "task3" in file:
    #         task = "task3"
        dataset = NewDataset(file, "mask")
        train_loader_task1 = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    #     import matplotlib.pyplot as plt
        df = pd.DataFrame(columns=["image", "mask_0.1", "mask_0.2", "mask_0.3", "mask_0.4", "mask_0.5", "mask_0.6",
                                   "mask_0.7", "mask_0.8", "mask_0.9"])
        for image, mask, label, image_folder in train_loader_task1:
            output = [image_folder[0]]
            if len(np.unique(mask)) > 1:
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    threshold = round(threshold, 1)
                    new_gt = np.zeros(mask[0].shape)
                    new_gt[mask[0] >= threshold] = 1
                    threshold_str = str(threshold).replace(".", "_")

                    # if "brain-tumor" not in image_folder[0]:
                    #     np.save(f"{image_folder[0][:-10]}/{task}_threshold_{threshold_str}.npy", new_gt)
                    #     output.append(f"{image_folder[0][:-10]}/{task}_threshold_{threshold_str}.npy")
                    # else:
                    m = re.search("image_[0-9]{1,3}.png", image_folder[0])
                    image_index = re.search("[0-9]{1,3}", m.group(0)).group(0)
                    new_image_folder = re.sub("image_[0-9]{1,3}.png", "", image_folder[0])
                    np.save(f"{new_image_folder}/{task}_threshold_{threshold_str}_{image_index}.npy", new_gt)
                    output.append(f"{new_image_folder}/{task}_threshold_{threshold_str}_{image_index}.npy")
                temp_df = pd.DataFrame([output], columns=["image", "mask_0.1", "mask_0.2", "mask_0.3", "mask_0.4",
                                                          "mask_0.5", "mask_0.6", "mask_0.7", "mask_0.8", "mask_0.9"])
                df = df.append(temp_df)
        df.to_csv(f"./{file}_per_threshold.csv", index=False)
        # print(image_folder[0].split("/")[3])
    #     print(image.shape)
    #    a, b, c = np.where(mask > 0)
    #    a = set(a)
    #    temp = np.zeros((3, 2)).astype("float32")
    #    for e in range(len(temp)):
    #        if e in a:
    #            temp[e][0] = 0
    #            temp[e][1] = 1
    #        else:
    #            temp[e][0] = 1
    #            temp[e][1] = 0
    #    print(temp)
    #    print(loss(label, torch.from_numpy(temp)))
        # image, mask = dataset[i]

    # mask = np.expand_dims(mask, axis=0)
    # mask = np.expand_dims(mask, axis=0)
    # print(np.min(mask))
    # print(np.max(mask))
    # mask = torch.from_numpy(mask)
    # a = thresholding_inclusive(mask)
    # for e in a:
    #     print(e.shape)
    #     plt.imshow(e)
    #     plt.show()
    #     print(image.shape)
        #f, axes = plt.subplots(1, 2)
        #axes[0].imshow(mask[0])
        #axes[1].imshow(mask[1])
        #plt.show()