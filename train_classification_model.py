import numpy as np
import torch
import albumentations as albu
import cv2
# from dataset.classification_dataset import ClassificationDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os
import argparse
import torch.nn as nn
from torchvision.transforms import transforms
from criteria.bce_loss import BceLoss
from network.deep_lab_v3_classification import DeepLabV3Plus

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

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(640, 640, interpolation=cv2.INTER_LINEAR),
        albu.CenterCrop(height=640, width=640),
        # albu.Resize(640, 640, interpolation=cv2.INTER_LINEAR),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')

def torch_transform(image, mask):
    image = image.astype("float32")
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
    mask = mask.transpose(2, 0, 1)
    ttf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = ttf(image)
    return {"image": image, "mask": transforms.ToTensor()(mask)}

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
    argparser.add_argument("--train_data", "-d", type=str)
    argparser.add_argument("--val_data", "-vd", type=str)
    argparser.add_argument("--model_name", type=str, default="DeeplabPlus")
    argparser.add_argument("--learning_rate", "-lr", type=float, default=0.0001)
    argparser.add_argument("--backbone", "-b", type=str, default="efficientnet-b4")
    argparser.add_argument("--encoder_weights", "-ew", type=str, default="imagenet")
    argparser.add_argument("--device", "-c", type=str, default="0,1")
    argparser.add_argument("--classification_head", type=bool, default=False)
    argparser.add_argument("--train_batch", type=int, default=8)
    argparser.add_argument("--val_batch", type=int, default=1)
    argparser.add_argument("--is_distributed", type=bool, default=True)
    argparser.add_argument("--weighted_loss", type=bool, default=False)
    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    ENCODER = args.backbone
    ENCODER_WEIGHTS = args.encoder_weights
    # CLASSES = ['car']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    model = DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
        aux_params={"classes": 2, "dropout": 0.5, "activation": None}
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    if args.is_distributed == True:
        model = nn.DataParallel(model)
    train_dataset_task1 = ClassificationDataset(args.train_data, f"mask", augmentation=get_training_augmentation(),
                                     preprocessing_fn=get_preprocessing(preprocessing_fn))
    val_dataset_task1 = ClassificationDataset(args.val_data, f"mask", augmentation=get_validation_augmentation(),
                                     preprocessing_fn=get_preprocessing(preprocessing_fn))
    train_loader_task1 = DataLoader(train_dataset_task1, batch_size=args.train_batch, shuffle=True, num_workers=12, drop_last=True)
    valid_loader_task1 = DataLoader(val_dataset_task1, batch_size=args.val_batch, shuffle=False, num_workers=4)
    class_loss = BceLoss()
    metrics = [smp.utils.metrics.Accuracy(), smp.utils.metrics.Precision(), smp.utils.metrics.Recall()]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.learning_rate),
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, mode="max")
    train_epoch_task1 = smp.utils.train.TrainEpoch(
        model,
        loss=class_loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch_task1 = smp.utils.train.ValidEpoch(
        model,
        loss=class_loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    epochs = 200
    max_score = 0
    if "ADC" in args.train_data:
        task_name = "ADC"
    elif "DWI" in args.train_data:
        task_name = "DWI"
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch_task1.run(train_loader_task1)
        valid_logs = valid_epoch_task1.run(valid_loader_task1)
        for k, v in valid_logs.items():
            print(k)
        if max_score < valid_logs['accuracy']:
           max_score = valid_logs['accuracy']
           best_model_for_task1 = f"./pretrain/best_{args.model_name}_classification_{task_name}_epoch_{i}.pth"
           if type(model) is nn.DataParallel:
               torch.save(model.module, best_model_for_task1)
           else:
                torch.save(model, best_model_for_task1)
           print(f'Model {best_model_for_task1} saved!')
        scheduler.step(valid_logs['accuracy'])
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"learning rate: {current_lr}")