from utils.constant import *
from training_config.base_config import BaseConfig
from data_augmentation.fill_in import FillIn
from data_augmentation.horizontal_flip import HorizontalFlip
from data_augmentation.padding import Padding
import torchvision.transforms as transforms

class RoiSegmentationBaseConfig(BaseConfig):

    def __init__(self, csv_file, old_root_dir, new_root_dir):
        super(RoiSegmentationBaseConfig, self).__init__()
        self.val_loss = {
            "loss1": {
                "file": "criteria/dice_loss"
            }
        }
        self.val_loss_weights = [1]
        self.loss_weights = [1]
        self.optimizer = {
            "name": OptimizerType.ADAM,
            "parameters": {
                "init_setup": {
                    "lr": 0.001,
                    "betas": (0.9, 0.999,),
                    "eps": 10 ** -8
                }
            }
        }
        setattr(self, DatasetTypeString.VAL, {
            "dataset": {
                "file": "dsc_mrp_dataset",
                "parameters": {
                    "root_dir": f"{root_dir}",
                    "excel_file": f"{excel_file}",
                    "dataset_type": DatasetType.VAL,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'labels_weight': 'phase_maps_medfilt_rs_n_wm.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([Padding(TargetSize(320, 320)), FillIn(50), HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 5
            }
        })
        setattr(self, DatasetTypeString.TEST, {
            "dataset": {
                "file": "dsc_mrp_dataset",
                "parameters": {
                    "root_dir": f"{root_dir}",
                    "excel_file": f"{excel_file}",
                    "dataset_type": DatasetType.TEST,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([Padding(TargetSize(320, 320)), FillIn(50)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 5
            }
        })


