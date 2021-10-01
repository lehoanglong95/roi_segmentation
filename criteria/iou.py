from criteria.base_loss import BaseLoss
import torch
import torch.nn.functional as F


class Iou(BaseLoss):

    def __init__(self, smooth=1e-15, device=torch.device("cuda:0")):
        super(Iou, self).__init__(device)
        self.smooth = smooth

    # TODO: implement focal loss for each lesion
    def forward(self, predict, target):
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target) - intersection
        loss = (intersection + self.smooth) / (union + self.smooth)
        return loss


if __name__ == '__main__':
    iou = Iou()
    pred = torch.Tensor([[[1, 0],
          [0, 1]]])
    gt = torch.Tensor([[[1,0],[0,1]]])
    print(iou(pred, gt))
