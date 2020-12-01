from criteria.base_loss import BaseLoss
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class FocalLoss(BaseLoss):

    def __init__(self, gamma=4, alpha=torch.Tensor([1, 19]), size_average=True, device=torch.device("cuda: 0")):
        super(FocalLoss, self).__init__(device)
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, predict, target):
        # print(torch.unique(target, return_counts=True))
        predict = predict.to(self.device)
        log_prob = F.log_softmax(predict, dim=1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target.type(torch.LongTensor).to(self.device),
            weight=self.alpha.to(self.device),
        )


if __name__ == '__main__':
    focal_loss = FocalLoss()
    pred = torch.Tensor([[[[0.3958, 0.8547],
          [0.7981, 0.3841]],
            [[0.3265, 0.6503],
          [0.7690, 0.1189]]]])
    gt = torch.Tensor([[[1,0],[0,1]]]).type(torch.LongTensor)
    print(pred.shape)
    print(gt.shape)
    # print(torch.log_softmax(pred, dim=1))
    print(focal_loss(pred, gt))
