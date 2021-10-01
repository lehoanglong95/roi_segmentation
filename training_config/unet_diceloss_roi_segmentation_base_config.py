import torchvision.transforms as transforms
from data_augmentation.padding import Padding
from data_augmentation.fill_in import FillIn
from data_augmentation.horizontal_flip import HorizontalFlip
import torch
from training_config.roi_segmentation_base_config import RoiSegmentationBaseConfig
from utils.constant import *


class _Config(RoiSegmentationBaseConfig):
    #'/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx'
    def __init__(self, csv_file="/home/longlh/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv",
                 old_root_dir=None, new_root_dir=None):
        super(_Config, self).__init__(csv_file, old_root_dir, new_root_dir)
        self.model_parallel = False
        self.network_architecture = {
            "file": "network/u_net",
            "parameters": {
                "input_channel": 50,
                "output_channel": 100,
                "soft_dim": 1,
                "is_training": True
            }
        }
        self.loss = {
            "loss1": {
                "file": "criteria/dice_loss",
                "parameters": {
                    "device": torch.device("cuda:1")
                }
            },
            "loss2": {
                "file": "criteria/cross_entropy_loss",
                "parameters": {
                    "device": torch.device("cuda:1")
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
                    "file_names": {'inputs': 'input.npy',
                                   'labels': 'gt.npy'},
                    "transform": transforms.Compose([Padding(TargetSize(320, 320)), FillIn(50), HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 8,
                "shuffle": True,
                "num_workers": 5,
            }
        })


config = _Config().__dict__
print(config)