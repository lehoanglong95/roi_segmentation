import torch
import torch.nn as nn

class BaseEvaluationMetric(nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super(BaseEvaluationMetric, self).__init__()
        self.device = device