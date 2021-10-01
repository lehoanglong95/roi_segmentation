import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric
import numpy as np

class Dice2(BaseEvaluationMetric):

    def forward(self, predict, target):
        true = np.copy(target)
        pred = np.copy(predict)
        true_id = list(np.unique(true))
        pred_id = list(np.unique(pred))

        overall_total = 0
        overall_inter = 0

        true_masks = [np.zeros(true.shape)]
        for t in true_id[1:]:
            t_mask = np.array(true == t, np.uint8)
            true_masks.append(t_mask)

        pred_masks = [np.zeros(true.shape)]
        for p in pred_id[1:]:
            p_mask = np.array(pred == p, np.uint8)
            pred_masks.append(p_mask)

        for true_idx in range(1, len(true_id)):
            t_mask = true_masks[true_idx]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            try: # blinly remove background
                pred_true_overlap_id.remove(0)
            except ValueError:
                pass  # just mean no background
            for pred_idx in pred_true_overlap_id:
                p_mask = pred_masks[pred_idx]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                overall_total += total
                overall_inter += inter
        if overall_total == 0:
            return 0
        return 2 * overall_inter / overall_total


# if __name__ == '__main__':
