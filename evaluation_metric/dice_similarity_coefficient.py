import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric

class DiceSimilarityCoefficient(BaseEvaluationMetric):


    def forward(self, predict, target):
        """

        :param predict: NxDxWxH
        :param target: NxDxWxH
        :return: DSC = 2TP / (2TP + FN + FP) with TP: True Positive, FN: False Negative, FP: False Positive
        """
        TP = 0
        FN = 0
        FP = 0
        positive_index = torch.where(predict == 1)
        for n,d,w,h in zip(positive_index[0], positive_index[1], positive_index[2], positive_index[3]):
            if target[n, d, w, h] == 1:
                TP += 1
            elif target[n, d, w, h] == 0:
                FP += 1
        negative_index = torch.where(predict == 0)
        for n,d,w,h in zip(negative_index[0], negative_index[1], negative_index[2], negative_index[3]):
            if target[n, d, w, h] == 1:
                FN += 1
        return 2 * TP / (2 * TP + FN + FP)