import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop
from data_augmentation.padding import Padding
from data_augmentation.horizontal_flip import HorizontalFlip
from data_augmentation.rescale_and_normalize import RescaleAndNormalize
import torch
from training_config.roi_segmentation_base_config import RoiSegmentationBaseConfig
from utils.constant import *


class _Config(RoiSegmentationBaseConfig):
    #'/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx'
    def __init__(self, csv_file="/home/longlh/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv",
                 old_root_dir="/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training",
                 new_root_dir="/home/longlh/hard_2/roi_numpy"):
        super(_Config, self).__init__(csv_file, old_root_dir, new_root_dir)
        self.model_parallel = False
        self.network_architecture = {
            "file": "network/unet_3d/u_net",
            "parameters": {
                "input_channel": 2,
                "output_channel": 2,
                "soft_dim": 1,
                "is_training": True,
                "device": torch.device("cuda: 1")
            }
        }
        self.loss_weights = [0.5, 0.5]
        self.loss = {
            "loss1": {
                "file": "criteria/dice_loss",
                "parameters": {
                    "device": torch.device("cuda: 1")
                }
            },
            "loss2": {
                "file": "criteria/focal_loss",
                "parameters": {
                    "device": torch.device("cuda: 1")
                }
            }
        }
        setattr(self, DatasetTypeString.TRAIN, {
            "dataset": {
                "file": "dataset/roi_segmentation_dataset",
                "parameters": {
                    "csv_file": f"{csv_file}",
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "dataset_type": DatasetType.TRAIN,
                    "file_names": {'ADC_inputs': 'ADC/sampled_input.npy',
                                   'ADC_mask': 'ADC/mask.npy',
                                   'DWI_inputs': 'DWI/sampled_input.npy',
                                   'DWI_mask': 'DWI/mask.npy',
                                   'labels': 'ADC/sampled_gt.npy'},
                    "transform": transforms.Compose([Padding(TargetSize(192, 192)), RescaleAndNormalize(), HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 5,
            }
        })


config = _Config().__dict__
print(config)