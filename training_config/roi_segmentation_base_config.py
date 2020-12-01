from utils.constant import *
from training_config.base_config import BaseConfig
from data_augmentation.horizontal_flip import HorizontalFlip
from data_augmentation.padding import Padding
from data_augmentation.rescale_and_normalize import RescaleAndNormalize
import torchvision.transforms as transforms
import torch

class RoiSegmentationBaseConfig(BaseConfig):

    def __init__(self, csv_file, old_root_dir, new_root_dir):
        super(RoiSegmentationBaseConfig, self).__init__()
        self.epochs = 600
        self.val_loss = {
            "loss1": {
                "file": "criteria/dice_loss",
                "parameters": {
                    "device": torch.device("cuda: 1")
                }
            },
            # "loss2": {
            #     "file": "criteria/focal_loss",
            #     "parameters": {
            #         "device": torch.device("cuda: 1")
            #     }
            # }
        }
        self.validation_leap = 4
        self.val_loss_weights = [1]
        self.optimizer = {
            "name": OptimizerType.ADAM,
            "parameters": {
                "init_setup": {
                    "lr": 1e-5,
                    "betas": (0.9, 0.999,),
                    "eps": 10 ** -8
                }
            }
        }
        self.lr_scheduler = {
            "name": OptimizerLrScheduler.ReduceLROnPlateau,
            "parameters": {
                "factor": 0.5,
                "patience": 3
            }
        }
        setattr(self, DatasetTypeString.VAL, {
            "dataset": {
                "file": "dataset/roi_segmentation_dataset",
                "parameters": {
                    "csv_file": f"{csv_file}",
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "dataset_type": DatasetType.VAL,
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
        setattr(self, DatasetTypeString.TEST, {
            "dataset": {
                "file": "dataset/roi_segmentation_dataset",
                "parameters": {
                    "csv_file": f"{csv_file}",
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "dataset_type": DatasetType.TEST,
                    "file_names": {'ADC_inputs': 'ADC/sampled_input.npy',
                                   'ADC_mask': 'ADC/mask.npy',
                                   'DWI_inputs': 'DWI/sampled_input.npy',
                                   'DWI_mask': 'DWI/mask.npy',
                                   'labels': 'ADC/sampled_gt.npy'},
                    "transform": transforms.Compose([Padding(TargetSize(192, 192)), RescaleAndNormalize()])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 5,
            }
        })


