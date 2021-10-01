import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop
from data_augmentation.padding import Padding
from data_augmentation.horizontal_flip import HorizontalFlip
from data_augmentation.rescale_and_normalize import RescaleAndNormalize
import torch
from training_config.roi_segmentation_base_config import RoiSegmentationBaseConfig
from utils.constant import *


class _Config(RoiSegmentationBaseConfig):
    def __init__(self, csv_file="/home/compu/data/long/projects/roi_segmentation/roi_segmentation_dataset.csv",
                 old_root_dir="/home/longle/long_data/brain_lesion_segmentation_clean_data",
                 new_root_dir="/home/compu/data/long/data/brain_lesion_segmentation_clean_data"):
        super(_Config, self).__init__(csv_file, old_root_dir, new_root_dir)
        self.validation_leap = 4
        self.model_parallel = False
        self.network_architecture = {
            "file": "network/double_unet/double_unet",
            "parameters": {
                "input_channel": 2,
                "output_channel": 1,
            }
        }
        self.loss_weights = [1]
        self.loss = {
            "loss1": {
                "file": "criteria/dice_loss",
                "parameters": {
                    "device": torch.device("cuda:2")
                }
            }
            # "loss2": {
            #     "file": "criteria/focal_loss",
            #     "parameters": {
            #         "device": torch.device("cuda:2")
            #     }
            # }
        }
        self.val_loss = {
            "loss1": {
                "file": "criteria/dice_loss",
                "parameters": {
                    "device": torch.device("cuda:2")
                }
            },
            "loss2": {
                "file": "criteria/focal_loss",
                "parameters": {
                    "device": torch.device("cuda:2")
                }
            },
            "loss3": {
                "file": "criteria/iou",
                "parameters": {
                    "device": torch.device("cuda:2")
                }
            }
        }
        self.val_loss_weights = [1, 1, 1]
        setattr(self, DatasetTypeString.TRAIN, {
            "dataset": {
                "file": "dataset/roi_segmentation_dataset",
                "parameters": {
                    "csv_file": f"{csv_file}",
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "dataset_type": DatasetType.TRAIN,
                    "file_names": {'ADC_inputs': 'after_registration_no_empty_slices_adc.npy',
                                   'DWI_inputs': 'after_registration_no_empty_slices_dwi.npy',
                                   'ADC_mask': 'final_mask.npy',
                                   'DWI_mask': 'final_mask.npy',
                                   'labels': 'after_registration_no_empty_slices_gt.npy'},
                    "transform": transforms.Compose([Padding(TargetSize(224, 224)), RescaleAndNormalize(), HorizontalFlip(0.5)])
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