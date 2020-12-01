from criteria.base_loss import BaseLoss
import torch
import torch.nn as nn

class CrossEntropyLoss(BaseLoss):

    def forward(self, predict, target, target_dim=1):
        print(predict.shape)
        print(target.shape)
        predict = predict.to(self.device)
        target = target.to(self.device)
        predict = torch.log_softmax(predict, dim=target_dim)
        criteria = nn.NLLLoss()
        return criteria(predict, target.type(torch.LongTensor).to(self.device))

if __name__ == '__main__':
    pred = torch.Tensor([[[[0.3958, 0.8547],
                           [0.7981, 0.3841]],
                          [[0.3265, 0.6503],
                           [0.7690, 0.1189]]]])
    print(torch.softmax(pred, dim=1))
    gt = torch.Tensor([[[1, 0], [0, 1]]]).type(torch.LongTensor)
    ce = CrossEntropyLoss()
    print(ce(pred, gt))