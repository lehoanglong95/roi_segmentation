import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric


class DiceSensitivity(BaseEvaluationMetric):

    def forward(self, predict, target):
        """

        :param predict: NxDxWxH
        :param target: NxDxWxH
        :return: SE = TP / (TP + FN)
        """
        TP = (target[predict == 1] == 1).sum()
        FN = (target[predict == 0] == 1).sum()
        if TP == 0 and FN == 0:
            return 0
        return float(TP) / float((TP + FN))


# if __name__ == '__main__':
