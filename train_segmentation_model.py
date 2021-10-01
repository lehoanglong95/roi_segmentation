import numpy as np
import torch
import albumentations as albu
import cv2
from dataset.new_dataset import NewDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses.dice import DiceLoss
# import
import os
import argparse
import torch.nn as nn

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.Rotate(limit=(-30, 30), p=0.5),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)

# def get_validation_augmentation():
#     """Add paddings to make image shape divisible by 32"""
#     test_transform = [
#         albu.Resize(240, 240, interpolation=cv2.INTER_LINEAR),
#     ]
#     return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

if __name__ == '__main__':
    #######ARG_PARSE#######
    argparser = argparse.ArgumentParser(description="Training Option")
    argparser.add_argument("--data", "-d", type=str, default="ADC")
    argparser.add_argument("--model_name", type=str, default="DeeplabPlus")
    argparser.add_argument("--learning_rate", "-lr", type=float, default=0.0001)
    argparser.add_argument("--backbone", "-b", type=str, default="efficientnet-b1")
    argparser.add_argument("--encoder_weights", "-ew", type=str, default="imagenet")
    argparser.add_argument("--device", "-c", type=str, default="0,1,2,3")
    argparser.add_argument("--classification_head", type=bool, default=False)
    argparser.add_argument("--train_batch", type=int, default=4)
    argparser.add_argument("--val_batch", type=int, default=1)
    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    ENCODER = args.backbone
    ENCODER_WEIGHTS = args.encoder_weights
    # CLASSES = ['car']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_dataset = NewDataset("./pancrea_training_dataset.csv", augmentation=get_training_augmentation(),
                                     preprocessing_fn=get_preprocessing(preprocessing_fn))
    val_dataset = NewDataset("./pancrea_val_dataset.csv", preprocessing_fn=get_preprocessing(preprocessing_fn))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=12, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False, num_workers=4)
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.learning_rate),
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, mode="max")
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    epochs = 100
    max_score = 0
    best_model_for_task1 = ""
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f'./pretrain/best_{args.model_name}_{args.data}_epoch_{i}.pth')
            best_model_for_task1 = f"./pretrain/best_{args.model_name}_{args.data}_epoch_{i}.pth"
            print(f'Model {best_model_for_task1} saved!')
        scheduler.step(valid_logs['iou_score'])
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"learning rate: {current_lr}")


