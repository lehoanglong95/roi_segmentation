from criteria.base_loss import BaseLoss
import torch
import torch.nn.functional as F


class DiceLoss(BaseLoss):

    def __init__(self, device, smooth=1, p=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.set_device(device)

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        loss = 0
        predict = predict.to(self.device)
        target = target.to(self.device)
        predict = F.softmax(predict, dim=1)
        target = F.one_hot(target, num_classes=2).permute(0, 4, 1, 2, 3).contiguous()
        # predict = torch.argmax(torch.softmax(predict, dim=self.target_dim), dim=self.target_dim)
        # predict = raw_predict[:, :, 1, :, :]
        for i in range(predict.shape[1]):
            p = predict[:, i, :, :, :]
            t = target[:, i, :, :, :]
            p = p.contiguous().view(-1)
            t = t.contiguous().view(-1)

            intersection = (p * t).sum()
            predict_sum = torch.sum(p * p)
            target_sum = torch.sum(t * t)
            loss += 1 - ((2. * intersection + self.smooth) / (predict_sum + target_sum + self.smooth))
        return loss / 2.
        # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        #
        # loss = 1 - num / den
        #
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # elif self.reduction == 'none':
        #     return loss
        # else:
        #     raise Exception('Unexpected reduction {}'.format(self.reduction))


if __name__ == '__main__':
    # from network.u_net import UNet
    from network.unet_3d.u_net import UNet
    from utils.constant import *
    from dataset.roi_segmentation_dataset import RoiSegmentationDataset
    from data_augmentation.resize_image import ResizeImage
    from data_augmentation.padding import Padding
    from data_augmentation.rescale_and_normalize import RescaleAndNormalize
    from torchvision.transforms import transforms
    from torch.utils import data
    import numpy as np
    import pathlib
    net = UNet(1, 2, 1, True, torch.device("cuda: 1")).to(torch.device("cuda: 1"))
    net.load_state_dict(torch.load("/home/longlh/PycharmProjects/roi_segmentation/"
                                   "pretrain/UNet_0.2DiceLoss_0.8FocalLoss_Adam_inputs_labels_2020-11-26-01-46-56_seed15/model_180.pth",
                                   map_location=torch.device("cuda: 1"))["model_state_dict"])
    dataset = RoiSegmentationDataset("/home/longlh/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv",
                                     DatasetType.TEST,
                                     {'inputs': 'sampled_input.npy', 'labels': 'sampled_gt.npy'},
                                     transform=transforms.Compose([Padding(TargetSize(192, 192)), RescaleAndNormalize()]),
                                     new_root_dir="/home/longlh/hard_2/roi_numpy",
                                     old_root_dir="/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training"
                                     )
    config = {"batch_size": 1,
                "shuffle": False,
                "num_workers": 5}
    data_generator = data.DataLoader(dataset, **config)
    # transforms.Normalize
    dice_loss = DiceLoss(torch.device("cuda: 1"))
    dice_loss_list = []
    net.eval()
    for idx, (inputs, labels) in enumerate(data_generator):
        inputs = inputs.to(torch.device("cuda: 1"))
        labels = labels.to(torch.device("cuda: 1"))
        inputs = torch.unsqueeze(inputs, dim=1)
        labels = torch.unsqueeze(labels, dim=1)
        predict = net(inputs)
        # dice_loss_list.append({"idx": idx, "vl": float(dice_loss(predict, labels))})
        dice_loss_list.append(float(dice_loss(predict, labels)))
    # dice_loss_list.sort(key=lambda x: x["vl"])
    # for dice in dice_loss_list:
    #     print(dice)
    import statistics
    print(statistics.stdev(dice_loss_list))
    print(statistics.mean(dice_loss_list))
    print(statistics.median(dice_loss_list))
    # print(predict.shape)
        # predict = torch.squeeze(predict, dim=1)
        # predict = torch.squeeze(predict, dim=1)
        # pathlib.Path(f"/home/longlh/PycharmProjects/roi_segmentation/result/{idx}").mkdir(parents=True, exist_ok=True)
        # np.save(f"/home/longlh/PycharmProjects/roi_segmentation/result/{idx}/predict.npy", predict.cpu().detach().numpy())
        # np.save(f"/home/longlh/PycharmProjects/roi_segmentation/result/{idx}/gt.npy", labels.cpu().detach().numpy())
        # print(predict)
    # a = torch.randint(0, 2, (16, 50, 300, 300)).type(torch.FloatTensor)
    # loss = DiceLoss(torch.device("cuda: 0"))
    # print(loss(a, a))