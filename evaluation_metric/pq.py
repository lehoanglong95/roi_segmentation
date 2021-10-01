import torch
from evaluation_metric.base_evaluation_metric import BaseEvaluationMetric
import numpy as np
from scipy.optimize import linear_sum_assignment

class Pq(BaseEvaluationMetric):

    def forward(self, predict, target, match_iou=0.5):
        assert match_iou >= 0.0, "Cant' be negative"

        true = np.copy(target)
        pred = np.copy(predict)
        true_id_list = list(np.unique(true))
        pred_id_list = list(np.unique(pred))

        true_masks = [None, ]
        for t in true_id_list[1:]:
            t_mask = np.array(true == t, np.uint8)
            true_masks.append(t_mask)

        pred_masks = [None, ]
        for p in pred_id_list[1:]:
            p_mask = np.array(pred == p, np.uint8)
            pred_masks.append(p_mask)

        # prefill with value
        pairwise_iou = np.zeros([len(true_id_list) - 1,
                                 len(pred_id_list) - 1], dtype=np.float64)

        # caching pairwise iou
        for true_id in true_id_list[1:]:  # 0-th is background
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id - 1, pred_id - 1] = iou
        #
        if match_iou >= 0.5:
            paired_iou = pairwise_iou[pairwise_iou > match_iou]
            pairwise_iou[pairwise_iou <= match_iou] = 0.0
            paired_true, paired_pred = np.nonzero(pairwise_iou)
            paired_iou = pairwise_iou[paired_true, paired_pred]
            paired_true += 1  # index is instance id - 1
            paired_pred += 1  # hence return back to original
        else:  # * Exhaustive maximal unique pairing
            #### Munkres pairing with scipy library
            # the algorithm return (row indices, matched column indices)
            # if there is multiple same cost in a row, index of first occurence
            # is return, thus the unique pairing is ensure
            # inverse pair to get high IoU as minimum
            paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
            ### extract the paired cost and remove invalid pair
            paired_iou = pairwise_iou[paired_true, paired_pred]

            # now select those above threshold level
            # paired with iou = 0.0 i.e no intersection => FP or FN
            paired_true = list(paired_true[paired_iou > match_iou] + 1)
            paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
            paired_iou = paired_iou[paired_iou > match_iou]

        # get the actual FP and FN
        unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
        unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

        #
        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)
        # get the SQ, no paired has 0 iou so not impact
        sq = paired_iou.sum() / (tp + 1.0e-6)

        return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
