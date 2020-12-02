import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric
from evaluation_metric.precision_lesion_wise import PrecisionLesionWise
from evaluation_metric.recall_lesion_wise import RecallLesionWise

class F1LesionWise(BaseEvaluationMetric):


    def forward(self, predict, target):
        """

        :param predict: NxDxWxH
        :param target: NxDxWxH
        :return: F1 = 2xPxR / (P + R)
        """
        precision = PrecisionLesionWise()
        recall = RecallLesionWise()
        precison_number = precision(predict, target)
        recall_number = recall(predict, target)
        print(precison_number)
        print(recall_number)
        return 2 * precison_number * recall_number / (precison_number + recall_number)

if __name__ == '__main__':
    f1 = F1LesionWise()
    predict = torch.Tensor([[[[1, 1],
          [0, 1]],

         [[0, 0],
          [0, 1]]]])
    target = torch.Tensor([[[[0, 1],
          [1, 1]],

         [[1, 1],
          [1, 0]]]])
    score = f1(predict, target)
    print(score)