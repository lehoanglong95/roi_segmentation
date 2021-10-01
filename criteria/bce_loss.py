from segmentation_models_pytorch.utils import base
import torch

class BceLoss(base.Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor([0.1, 0.9]).to(torch.device("cuda:0")))

    def forward(self, y_pr, y_gt):
        return self.loss(y_pr, y_gt)
