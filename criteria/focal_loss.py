from criteria.base_loss import BaseLoss
import torch
import torch.nn.functional as F


class FocalLoss(BaseLoss):

    def __init__(self, gamma=4, alpha=3, size_average=True, device=torch.device("cuda:0")):
        super(FocalLoss, self).__init__(device)
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = 1e-6
        self.size_average = size_average

    # TODO: implement focal loss for each lesion
    def forward(self, predict, target):
        prob = torch.clamp(predict, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * torch.sum(neg_weight * F.logsigmoid(-predict)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss


if __name__ == '__main__':
    focal_loss = FocalLoss()
    pred = torch.Tensor([[[1, 0],
          [0, 1]]])
    gt = torch.Tensor([[[1,0],[0,1]]])
    print(pred.shape)
    print(gt.shape)
    # print(torch.log_softmax(pred, dim=1))
    print(focal_loss(pred, gt))
