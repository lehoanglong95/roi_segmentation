import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric

class DiceSimilarityCoefficient(BaseEvaluationMetric):


    def forward(self, predict, target):
        """

        :param predict: NxDxWxH
        :param target: NxDxWxH
        :return: DSC = 2TP / (2TP + FN + FP) with TP: True Positive, FN: False Negative, FP: False Positive
        """
        TP = (target[predict == 1] == 1).sum()
        FN = (target[predict == 0] == 1).sum()
        FP = (target[predict == 1] == 0).sum()
        return float(2 * TP) / float((2 * TP + FN + FP))