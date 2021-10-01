import torch
import torch.nn.functional as F
from evaluation_metric import *
from network.unet_3d.u_net import UNet
from utils.constant import *
from dataset.roi_segmentation_dataset import RoiSegmentationDataset
from data_augmentation.padding import Padding
from data_augmentation.rescale_and_normalize import RescaleAndNormalize
from torchvision.transforms import transforms
from torch.utils import data
import numpy as np

if __name__ == '__main__':
    net = UNet(2, 2, 1, False, torch.device("cuda:0"), torch.device("cuda:1"))
    net.load_state_dict(torch.load("/home/longlh/hard_2/PycharmProjects/roi_segmentation/"
                                   "pretrain/UNet_0.5DiceLoss_0.5FocalLoss_Adam_ADC_inputs_ADC_mask_DWI_inputs_DWI_mask_labels_2020-12-09-00-48-54_seed15/model_108.pth")[
                            "model_state_dict"])
    # / home / longlh / hard_2 / PycharmProjects / roi_segmentation / roi_segmentation_dataset.csv
    dataset = RoiSegmentationDataset("/home/longlh/hard_2/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv",
                                     DatasetType.TEST,
                                     {'ADC_inputs': 'ADC/sampled_input.npy',
                                      'ADC_mask': 'ADC/sampled_input.npy',
                                      'DWI_inputs': 'DWI/sampled_input.npy',
                                      'DWI_mask': 'DWI/sampled_input.npy',
                                      'labels': 'ADC/sampled_gt.npy'},
                                     transform=transforms.Compose(
                                         [Padding(TargetSize(224, 224)), RescaleAndNormalize()]),
                                     new_root_dir="/home/longlh/hard_2/roi_numpy",
                                     old_root_dir="/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training"
                                     )
    config = {"batch_size": 1,
              "shuffle": False,
              "num_workers": 5}
    data_generator = data.DataLoader(dataset, **config)
    # transforms.Normalize
    device = torch.device("cuda:0")
    from collections import defaultdict
    a = defaultdict(list)
    print(len(data_generator))
    for idx, (ADC_inputs, ADC_mask, DWI_inputs, DWI_mask, labels, file_name) in enumerate(data_generator):
        patient = file_name[0].split("/")[5]
        print(patient)
        number_of_pixel_in_roi = labels.sum()
        ADC_inputs = ADC_inputs.to(device, dtype=torch.float)
        ADC_mask = ADC_mask.to(device, dtype=torch.long)
        DWI_inputs = DWI_inputs.to(device, dtype=torch.float)
        DWI_mask = DWI_mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)
        # print(ADC_inputs.shape)
        # ADC_inputs[ADC_mask == 0] = 0
        # DWI_inputs[DWI_mask == 0] = 0
        # labels[DWI_mask == 0] = 0
        ADC_inputs = torch.unsqueeze(ADC_inputs, dim=1)
        DWI_inputs = torch.unsqueeze(DWI_inputs, dim=1)
        # print(ADC_inputs.shape)
        inputs = torch.cat([ADC_inputs, DWI_inputs], dim=1)
        # np.save(f"/home/longlh/hard_2/roi_numpy/BP908 20181120/ADC/new_input.npy", inputs.squeeze().cpu().numpy())
        # np.save(f"/home/longlh/hard_2/roi_numpy/BP908 20181120/ADC/new_gt.npy", labels.squeeze().cpu().numpy())
        # print(inputs.shape)
        predict = net(inputs)
        # np.save("/home/longlh/hard_2/roi_numpy/BP908 20181120/ADC/predict.npy", predict.squeeze().cpu().numpy())
        a[patient].append(3 * np.count_nonzero(predict.cpu().numpy()))
    folder = "/home/longlh/hard_2/PycharmProjects/roi_segmentation/test_dataset.csv"
    # folder = "/home/longlh/hard_2/PycharmProjects/registration/error_file.csv"
    import pandas as pd
    from utils.register import ADC
    file_names = pd.read_csv(f"{folder}", header=None,
                             names=["file_names", "type"])["file_names"]
    for file in file_names:
        patient = file.split("/")[5]
        new_file = file.replace("CMC AI Auto Stroke VOL _Training", "roi_numpy")
        new_gt = np.load(f"{new_file}/ADC/sampled_gt.npy")
        new_number_of_voxels = np.count_nonzero(new_gt)
        a[patient].append(new_number_of_voxels * 3)
    b = []
    for k, v in a.items():
        b.append(abs(v[0] - v[1]))
        print(f"{k}, {v[0]}, {v[1]}")
    print(sum(b) / len(b))
