from torch.utils import data
import csv
import os
import numpy as np
from utils.constant import DatasetType


class RoiSegmentationDataset(data.Dataset):

    def __init__(self, csv_file, dataset_type, file_names, transform=None, new_root_dir=None, old_root_dir=None):
        if not isinstance(file_names, dict):
            raise TypeError("file_names must be dict")
        self.new_root_dir = new_root_dir
        self.old_root_dir = old_root_dir
        self.csv_file = csv_file
        self.dataset_type = dataset_type
        self.file_names = file_names
        self.transform = transform
        self._get_file_names(self.csv_file, self.dataset_type, self.file_names)

    def _get_file_names(self, dataset_file, dataset_type, file_names):
        for file_name in file_names.keys():
            setattr(self, file_name, [])
        with open(dataset_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if dataset_type != DatasetType.ALL:
                    if int(row[1]) == dataset_type:
                        if self.old_root_dir and self.new_root_dir:
                            dsc_file = row[0].replace(self.old_root_dir, self.new_root_dir)
                        else:
                            dsc_file = row[0]
                        data_file_dir = dsc_file
                        for k, v in file_names.items():
                            getattr(self, k).append(os.path.join(data_file_dir, v))
                else:
                    if self.old_root_dir and self.new_root_dir:
                        dsc_file = row[0].replace(self.old_root_dir, self.new_root_dir)
                    else:
                        dsc_file = row[0]
                    data_file_dir = dsc_file
                    for k, v in file_names.items():
                        getattr(self, k).append(os.path.join(data_file_dir, v))

    def __len__(self):
        return len(getattr(self, list(self.file_names.keys())[0]))

    def __getitem__(self, item):
        outputs = []
        for key in self.file_names.keys():
            sample = np.load(getattr(self, key)[item])
            outputs.append(sample)
        if self.transform:
            outputs = self.transform(outputs)
        return tuple(outputs)

    def __repr__(self):
        label_names = [file_name for file_name in self.file_names]
        return "_".join(label_names)

if __name__ == '__main__':
    import torchvision.transforms as transforms
    from data_augmentation.padding import Padding
    from data_augmentation.horizontal_flip import HorizontalFlip
    from data_augmentation.rescale_and_normalize import RescaleAndNormalize
    from utils.constant import TargetSize
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import waitforbuttonpress
    from skimage.util import montage
    import torch
    from torch.utils import data
    a = RoiSegmentationDataset("/home/longlh/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv",
                                DatasetType.TRAIN,
                               {'ADC_inputs': 'ADC/sampled_input.npy',
                                'ADC_mask': 'ADC/mask.npy',
                                'DWI_inputs': 'DWI/sampled_input.npy',
                                'DWI_mask': 'DWI/mask.npy',
                                'labels': 'ADC/sampled_gt.npy'},
                               transform=transforms.Compose([Padding(TargetSize(192, 192)), RescaleAndNormalize()]),
                               old_root_dir="/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training",
                               new_root_dir="/home/longlh/hard_2/roi_numpy",
                               )
    config = {"batch_size": 1,
                "shuffle": False,
                "num_workers": 5}
    b = data.DataLoader(a, **config)
    from collections import Counter
    import matplotlib.pyplot as plt
    from skimage.util import montage
    pre_1 = []
    pre_2 = []
    # ws = set()
    # hs = set()
    for idx, (ADC_inputs, ADC_mask, DWI_inputs, DWI_mask, gt) in enumerate(b):
        print(ADC_inputs.shape)
        print(ADC_mask.shape)
        print(DWI_inputs.shape)
        print(DWI_mask.shape)
        print(gt.shape)
        break
        # plt.show()
        # pre_1.append(int(torch.unique(gt, return_counts=True)[1][1]))
        # print(pre_1)
        # n, c, w, h = input.shape
        # ws.add(w)
        # hs.add(h)
        # print(f"INPUT SHAPE: {input.shape}")
        # print(f"GT SHAPE: {gt.shape}")
        # print(f"INPUT MIN: {torch.min(input)}")
        # print(f"INPUT MAX: {torch.max(input)}")
        # print(f"INPUT MEAN: {torch.mean(input)}")
        # print(f"INPUT STD: {torch.std(input)}")
        # print(f"LABEL UNIQUE: {torch.unique(gt)}")
        # print(idx)
        # numpy_input = torch.squeeze(input, dim=0).numpy()
        # numpy_gt = torch.squeeze(gt, dim=0).numpy()
        # plt.imshow(montage(numpy_input))
        # plt.show()
        # plt.imshow(montage(numpy_gt))
        # plt.show()
    # for idx, (input, gt) in enumerate(b):
    #     pre_2.append(int(torch.unique(gt, return_counts=True)[1][1]))
    # print(len(pre_1))
    # print(len(pre_2))
    # print(0 in pre_1)
    # print(0 in pre_2)
    # print(Counter(pre_1) == Counter(pre_2))
    # print(Counter(ws))
    # print(Counter(hs))
