import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric


class PrecisionLesionWise(BaseEvaluationMetric):

    def forward(self, predict, target):
        """

        :param predict: NxDxWxH
        :param target: NxDxWxH
        :return: Precision = TP / (TP + FP)
        """
        TP = (target[predict == 1] == 1).sum()
        FP = (target[predict == 1] == 0).sum()
        if TP == 0 and FP == 0:
            return 0
        return float(TP) / float((TP + FP))