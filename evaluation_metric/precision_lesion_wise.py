import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric


class PrecisionLesionWise(BaseEvaluationMetric):

    def forward(self, predict, target):
        """

        :param predict: NxDxWxH
        :param target: NxDxWxH
        :return: Precision = TP / (TP + FP)
        """
        TP = 0
        FP = 0
        positive_index = torch.where(predict == 1)
        for n, d, w, h in zip(positive_index[0], positive_index[1], positive_index[2], positive_index[3]):
            if target[n, d, w, h] == 1:
                TP += 1
            elif target[n, d, w, h] == 0:
                FP += 1
        return TP / (TP + FP)