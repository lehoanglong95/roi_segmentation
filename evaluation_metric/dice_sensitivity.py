import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric


class DiceSensitivity(BaseEvaluationMetric):

    def forward(self, predict, target):
        """

        :param predict: NxDxWxH
        :param target: NxDxWxH
        :return: SE = TP / (TP + FN)
        """
        TP = 0
        FN = 0
        positive_index = torch.where(predict == 1)
        for n, d, w, h in zip(positive_index[0], positive_index[1], positive_index[2], positive_index[3]):
            if target[n, d, w, h] == 1:
                TP += 1
        negative_index = torch.where(predict == 0)
        for n, d, w, h in zip(negative_index[0], negative_index[1], negative_index[2], negative_index[3]):
            if target[n, d, w, h] == 1:
                FN += 1
        return TP / (TP + FN)