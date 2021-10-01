from criteria.base_loss import BaseLoss
import torch
import torch.nn.functional as F
from evaluation_metric import *
import matplotlib.pyplot as plt
from skimage.util import montage

class DiceLoss(BaseLoss):

    def __init__(self, device, smooth=1, p=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.set_device(device)

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        target = target.to(self.device)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        return loss.mean()

if __name__ == '__main__':
    # from network.u_net import UNet
    from network.unet_3d.u_net import UNet
    from utils.constant import *
    from network.double_unet.double_unet import DoubleUnet
    from network.unet_version2.unet_version2 import UnetVersion2
    from dataset.roi_segmentation_dataset import RoiSegmentationDataset
    from data_augmentation.resize_image import ResizeImage
    from data_augmentation.padding import Padding
    from data_augmentation.rescale_and_normalize import RescaleAndNormalize
    from torchvision.transforms import transforms
    from torch.utils import data
    import numpy as np
    import pathlib

    # evaluation metric
    dice_sensitivity = DiceSensitivity()
    dice_similarity_coefficient = DiceSimilarityCoefficient()
    precision_lesion_wise = PrecisionLesionWise()
    recall_lesion_wise = RecallLesionWise()
    f1_lesion_wise = F1LesionWise()
    dice_2 = Dice2()
    aji = Aji()
    pq = Pq()

    # evaluation metrics for all samples
    dice_sensitivity_l = []
    dice_similarity_coefficient_l = []
    precision_lesion_wise_l = []
    recall_lesion_wise_l = []
    f1_lesion_wise_l = []
    dice_2_l = []
    aji_l = []
    dq_l = []
    sq_l = []
    pq_l = []

    # evaluation metrics for small samples
    dice_sensitivity_small_l = []
    dice_similarity_coefficient_small_l = []
    precision_lesion_wise_small_l = []
    recall_lesion_wise_small_l = []
    f1_lesion_wise_small_l = []
    dice_2_small_l = []
    aji_small_l = []
    dq_small_l = []
    sq_small_l = []
    pq_small_l = []

    # evaluation metrics for medium samples
    dice_sensitivity_medium_l = []
    dice_similarity_coefficient_medium_l = []
    precision_lesion_wise_medium_l = []
    recall_lesion_wise_medium_l = []
    f1_lesion_wise_medium_l = []
    dice_2_medium_l = []
    aji_medium_l = []
    dq_medium_l = []
    sq_medium_l = []
    pq_medium_l = []

    # evaluation metrics for large samples
    dice_sensitivity_large_l = []
    dice_similarity_coefficient_large_l = []
    precision_lesion_wise_large_l = []
    recall_lesion_wise_large_l = []
    f1_lesion_wise_large_l = []
    dice_2_large_l = []
    aji_large_l = []
    dq_large_l = []
    sq_large_l = []
    pq_large_l = []

    # net = UNet(2, 2, 1, False, torch.device("cuda:0"), torch.device("cuda:1"))
    device = torch.device("cuda:2")
    net = UnetVersion2(2, 1).to(device)
    # net = DoubleUnet(input_channel=2, output_channel=1).to(device)
    net.load_state_dict(torch.load("/home/compu/data/long/projects/roi_segmentation/"
                                   "pretrain/UnetVersion2_1DiceLoss_Adam_ADC_inputs_DWI_inputs_ADC_mask_DWI_mask_labels_2021-05-18-07-51-29_seed42/model_40.pth")["model_state_dict"])
    # / home / longlh / hard_2 / PycharmProjects / roi_segmentation / roi_segmentation_dataset.csv
    dataset = RoiSegmentationDataset("/home/compu/data/long/projects/roi_segmentation/roi_segmentation_dataset.csv",
                                     DatasetType.TEST,
                                     {'ADC_inputs': 'after_registration_no_empty_slices_adc.npy',
                                   'DWI_inputs': 'after_registration_no_empty_slices_dwi.npy',
                                   'ADC_mask': 'final_mask.npy',
                                   'DWI_mask': 'final_mask.npy',
                                   'labels': 'after_registration_no_empty_slices_gt.npy'},
                                     transform=transforms.Compose([Padding(TargetSize(224, 224)), RescaleAndNormalize()]),
                                     old_root_dir="/home/longle/long_data/brain_lesion_segmentation_clean_data",
                                     new_root_dir="/home/compu/data/long/data/brain_lesion_segmentation_clean_data"
                                     )
    config = {"batch_size": 1,
                "shuffle": False,
                "num_workers": 2}
    data_generator = data.DataLoader(dataset, **config)
    # transforms.Normalize
    # dice_loss = DiceLoss(torch.device("cuda:1"))
    dice_loss_list = []
    # net.eval()

    print(len(data_generator))
    for idx, (ADC_inputs, DWI_inputs, ADC_mask, DWI_mask, labels) in enumerate(data_generator):
        # patient = folder[0].split("/")[5]
        # print(patient)
        number_of_pixel_in_roi = labels.sum()
        ADC_inputs = ADC_inputs.to(device, dtype=torch.float)
        ADC_mask = ADC_mask.to(device, dtype=torch.long)
        DWI_inputs = DWI_inputs.to(device, dtype=torch.float)
        DWI_mask = DWI_mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)
        # print(ADC_inputs.shape)
        # ADC_inputs[ADC_mask == 0] = 0
        # DWI_inputs[DWI_mask == 0] = 0
        # labels[DWI_mask == 0] = 0
        ADC_inputs = torch.unsqueeze(ADC_inputs, dim=1)
        DWI_inputs = torch.unsqueeze(DWI_inputs, dim=1)
        inputs = torch.cat([ADC_inputs, DWI_inputs], dim=1)
        predict = net(inputs)
        # # predict[DWI_mask == 0] = 0
        # dice_sensitivity_l.append(dice_sensitivity(predict, labels))
        # dice_similarity_coefficient_l.append(dice_similarity_coefficient(predict, labels))
        # precision_lesion_wise_l.append(precision_lesion_wise(predict, labels))
        # recall_lesion_wise_l.append(recall_lesion_wise(predict, labels))
        # f1_lesion_wise_l.append(f1_lesion_wise(predict, labels))
        # dice_2_l.append(dice_2(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        # aji_l.append(aji(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        # temp_pq = pq(predict.cpu().detach().numpy(), labels.cpu().detach().numpy())
        # dq_l.append(temp_pq[0][0])
        # sq_l.append(temp_pq[0][1])
        # pq_l.append(temp_pq[0][2])
        # if number_of_pixel_in_roi > 20000:
        #     dice_sensitivity_large_l.append(dice_sensitivity(predict, labels))
        #     dice_similarity_coefficient_large_l.append(dice_similarity_coefficient(predict, labels))
        #     precision_lesion_wise_large_l.append(precision_lesion_wise(predict, labels))
        #     recall_lesion_wise_large_l.append(recall_lesion_wise(predict, labels))
        #     f1_lesion_wise_large_l.append(f1_lesion_wise(predict, labels))
        #     dice_2_large_l.append(dice_2(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        #     aji_large_l.append(aji(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        #     temp_large_pq = pq(predict.cpu().detach().numpy(), labels.cpu().detach().numpy())
        #     dq_large_l.append(temp_pq[0][0])
        #     sq_large_l.append(temp_pq[0][1])
        #     pq_large_l.append(temp_pq[0][2])
        # elif number_of_pixel_in_roi < 2000:
        #     dice_sensitivity_small_l.append(dice_sensitivity(predict, labels))
        #     dice_similarity_coefficient_small_l.append(dice_similarity_coefficient(predict, labels))
        #     precision_lesion_wise_small_l.append(precision_lesion_wise(predict, labels))
        #     recall_lesion_wise_small_l.append(recall_lesion_wise(predict, labels))
        #     f1_lesion_wise_small_l.append(f1_lesion_wise(predict, labels))
        #     dice_2_small_l.append(dice_2(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        #     aji_small_l.append(aji(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        #     temp_small_pq = pq(predict.cpu().detach().numpy(), labels.cpu().detach().numpy())
        #     dq_small_l.append(temp_pq[0][0])
        #     sq_small_l.append(temp_pq[0][1])
        #     pq_small_l.append(temp_pq[0][2])
        # else:
        #     dice_sensitivity_medium_l.append(dice_sensitivity(predict, labels))
        #     dice_similarity_coefficient_medium_l.append(dice_similarity_coefficient(predict, labels))
        #     precision_lesion_wise_medium_l.append(precision_lesion_wise(predict, labels))
        #     recall_lesion_wise_medium_l.append(recall_lesion_wise(predict, labels))
        #     f1_lesion_wise_medium_l.append(f1_lesion_wise(predict, labels))
        #     dice_2_medium_l.append(dice_2(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        #     aji_medium_l.append(aji(predict.cpu().detach().numpy(), labels.cpu().detach().numpy()))
        #     temp_medium_pq = pq(predict.cpu().detach().numpy(), labels.cpu().detach().numpy())
        #     dq_medium_l.append(temp_pq[0][0])
        #     sq_medium_l.append(temp_pq[0][1])
        #     pq_medium_l.append(temp_pq[0][2])
        # dice_loss_list.append({"idx": idx, "vl": float(dice_loss(predict, labels))})
        # dice_loss_list.append(float(dice_loss(predict, labels)))
    # # dice_loss_list.sort(key=lambda x: x["vl"])
    # # for dice in dice_loss_list:
    # #     print(dice)
    # import statistics
    # print(statistics.stdev(dice_loss_list))
    # print(statistics.mean(dice_loss_list))
    # print(statistics.median(dice_loss_list))
    # print(predict.shape)
        # predict = torch.squeeze(predict, dim=1)
        # predict = torch.squeeze(predict, dim=1)
        print(predict.shape)
        pathlib.Path(f"/home/compu/data/long/projects/roi_segmentation/result_relabel/unet_v2/{idx}").mkdir(parents=True, exist_ok=True)
        plt.imshow(montage(DWI_inputs[0][0].cpu().detach().numpy()), cmap='gray')
        plt.contour(montage(labels[0].cpu().detach().numpy()), linewidths=.3, colors='r')
        plt.contour(montage(predict[0][0].cpu().detach().numpy()), linewidths=.3, colors='y')
        plt.savefig(f"/home/compu/data/long/projects/roi_segmentation/result_relabel/unet_v2/{idx}/output.png")
        plt.close()
        # np.save(f"/home/longlh/hard_2/PycharmProjects/roi_segmentation/result_relabel/u_net/{idx}/predict.npy", predict.cpu().detach().numpy())
        # np.save(f"/home/longlh/hard_2/PycharmProjects/roi_segmentation/result_relabel/u_net/{idx}/gt.npy", labels.cpu().detach().numpy())
        # np.save(f"/home/longlh/hard_2/PycharmProjects/roi_segmentation/result_relabel/{patient}/dwi.npy",
        #         DWI_inputs.cpu().detach().numpy())
    # print(sq_l)
    # print(pq_l)
    # # print(precision_lesion_wise_l.index(0))
    # # print(recall_lesion_wise_l.index(0))
    # dice_sensitivity_l = torch.FloatTensor(dice_sensitivity_l)
    # dice_similarity_coefficient_l = torch.FloatTensor(dice_similarity_coefficient_l)
    # precision_lesion_wise_l = torch.FloatTensor(precision_lesion_wise_l)
    # recall_lesion_wise_l = torch.FloatTensor(recall_lesion_wise_l)
    # f1_lesion_wise_l = torch.FloatTensor(f1_lesion_wise_l)
    # dice_2_l = torch.FloatTensor(dice_2_l)
    # aji_l = torch.FloatTensor(aji_l)
    # dq_l = torch.FloatTensor(dq_l)
    # sq_l = torch.FloatTensor(sq_l)
    # pq_l = torch.FloatTensor(pq_l)
    #
    #
    # # dice_sensitivity_small_l = torch.FloatTensor(dice_sensitivity_small_l)
    # # dice_similarity_coefficient_small_l = torch.FloatTensor(dice_similarity_coefficient_small_l)
    # # precision_lesion_wise_small_l = torch.FloatTensor(precision_lesion_wise_small_l)
    # # recall_lesion_wise_small_l = torch.FloatTensor(recall_lesion_wise_small_l)
    # # f1_lesion_wise_small_l = torch.FloatTensor(f1_lesion_wise_small_l)
    # # dice_2_small_l = torch.FloatTensor(dice_2_small_l)
    # # aji_small_l = torch.FloatTensor(aji_small_l)
    # # dq_small_l = torch.FloatTensor(dq_small_l)
    # # sq_small_l = torch.FloatTensor(sq_small_l)
    # # pq_small_l = torch.FloatTensor(pq_small_l)
    # #
    # # dice_sensitivity_medium_l = torch.FloatTensor(dice_sensitivity_medium_l)
    # # dice_similarity_coefficient_medium_l = torch.FloatTensor(dice_similarity_coefficient_medium_l)
    # # precision_lesion_wise_medium_l = torch.FloatTensor(precision_lesion_wise_medium_l)
    # # recall_lesion_wise_medium_l = torch.FloatTensor(recall_lesion_wise_medium_l)
    # # f1_lesion_wise_medium_l = torch.FloatTensor(f1_lesion_wise_medium_l)
    # # dice_2_medium_l = torch.FloatTensor(dice_2_medium_l)
    # # aji_medium_l = torch.FloatTensor(aji_medium_l)
    # # dq_medium_l = torch.FloatTensor(dq_medium_l)
    # # sq_medium_l = torch.FloatTensor(sq_medium_l)
    # # pq_medium_l = torch.FloatTensor(pq_medium_l)
    # #
    # # dice_sensitivity_large_l = torch.FloatTensor(dice_sensitivity_large_l)
    # # dice_similarity_coefficient_large_l = torch.FloatTensor(dice_similarity_coefficient_large_l)
    # # precision_lesion_wise_large_l = torch.FloatTensor(precision_lesion_wise_large_l)
    # # recall_lesion_wise_large_l = torch.FloatTensor(recall_lesion_wise_large_l)
    # # f1_lesion_wise_large_l = torch.FloatTensor(f1_lesion_wise_large_l)
    # # dice_2_large_l = torch.FloatTensor(dice_2_large_l)
    # # aji_large_l = torch.FloatTensor(aji_large_l)
    # # dq_large_l = torch.FloatTensor(dq_large_l)
    # # sq_large_l = torch.FloatTensor(sq_large_l)
    # # pq_large_l = torch.FloatTensor(pq_large_l)
    #
    # for idx, metric in enumerate([dice_sensitivity_l, dice_similarity_coefficient_l, precision_lesion_wise_l, recall_lesion_wise_l,
    #                f1_lesion_wise_l, dice_2_l, aji_l, dq_l, sq_l, pq_l]):
    #     if idx == 0:
    #         metric_name = "dice_sensitivity"
    #     elif idx == 1:
    #         metric_name = "dice_similarity_coefficient"
    #     elif idx == 2:
    #         metric_name = "precision"
    #     elif idx == 3:
    #         metric_name = "recall"
    #     elif idx == 4:
    #         metric_name = "f1"
    #     elif idx == 5:
    #         metric_name = "dice_2"
    #     elif idx == 6:
    #         metric_name = "aji"
    #     elif idx == 7:
    #         metric_name = "dq"
    #     elif idx == 8:
    #         metric_name = "sq"
    #     elif idx == 9:
    #         metric_name = "pq"
    #     print(f"{metric_name}_mean: {metric.mean(0).item()}")
    #     # print(f"{metric_name}_median: {metric.median(0)[0].item()}")
    #     # print(f"{metric_name}_std: {metric.std(0).item()}")
    #     # print(f"{metric_name}_min: {metric.min(0)[0].item()}")
    #     # print(f"{metric_name}_max: {metric.max(0)[0].item()}")
    #     # print(f"{metric_name}_min_idx: {metric.min(0)[1].item()}")
    #     # print(f"{metric_name}_max_idx: {metric.max(0)[1].item()}")
    #
    # for idx, metric in enumerate(
    #         [dice_sensitivity_small_l, dice_similarity_coefficient_small_l, precision_lesion_wise_small_l, recall_lesion_wise_small_l,
    #          f1_lesion_wise_small_l, dice_2_small_l, aji_small_l, dq_small_l, sq_small_l, pq_small_l]):
    #     if idx == 0:
    #         metric_name = "small_dice_sensitivity"
    #     elif idx == 1:
    #         metric_name = "small_dice_similarity_coefficient"
    #     elif idx == 2:
    #         metric_name = "small_precision"
    #     elif idx == 3:
    #         metric_name = "small_recall"
    #     elif idx == 4:
    #         metric_name = "small_f1"
    #     elif idx == 5:
    #         metric_name = "small_dice_2"
    #     elif idx == 6:
    #         metric_name = "small_aji"
    #     elif idx == 7:
    #         metric_name = "small_dq"
    #     elif idx == 8:
    #         metric_name = "small_sq"
    #     elif idx == 9:
    #         metric_name = "small_pq"
    #     print(f"{metric_name}_mean: {metric.mean(0).item()}")
    #     # print(f"{metric_name}_median: {metric.median(0)[0].item()}")
    #     # print(f"{metric_name}_std: {metric.std(0).item()}")
    #     # print(f"{metric_name}_min: {metric.min(0)[0].item()}")
    #     # print(f"{metric_name}_max: {metric.max(0)[0].item()}")
    #     # print(f"{metric_name}_min_idx: {metric.min(0)[1].item()}")
    #     # print(f"{metric_name}_max_idx: {metric.max(0)[1].item()}")
    #
    # for idx, metric in enumerate(
    #         [dice_sensitivity_medium_l, dice_similarity_coefficient_medium_l, precision_lesion_wise_medium_l, recall_lesion_wise_medium_l,
    #          f1_lesion_wise_medium_l, dice_2_medium_l, aji_medium_l, dq_medium_l, sq_medium_l, pq_medium_l]):
    #     if idx == 0:
    #         metric_name = "medium_dice_sensitivity"
    #     elif idx == 1:
    #         metric_name = "medium_dice_similarity_coefficient"
    #     elif idx == 2:
    #         metric_name = "medium_precision"
    #     elif idx == 3:
    #         metric_name = "medium_recall"
    #     elif idx == 4:
    #         metric_name = "medium_f1"
    #     elif idx == 5:
    #         metric_name = "medium_dice_2"
    #     elif idx == 6:
    #         metric_name = "medium_aji"
    #     elif idx == 7:
    #         metric_name = "medium_dq"
    #     elif idx == 8:
    #         metric_name = "medium_sq"
    #     elif idx == 9:
    #         metric_name = "medium_pq"
    #     print(f"{metric_name}_mean: {metric.mean(0).item()}")
    #     # print(f"{metric_name}_median: {metric.median(0)[0].item()}")
    #     # print(f"{metric_name}_std: {metric.std(0).item()}")
    #     # print(f"{metric_name}_min: {metric.min(0)[0].item()}")
    #     # print(f"{metric_name}_max: {metric.max(0)[0].item()}")
    #     # print(f"{metric_name}_min_idx: {metric.min(0)[1].item()}")
    #     # print(f"{metric_name}_max_idx: {metric.max(0)[1].item()}")
    #
    # for idx, metric in enumerate(
    #         [dice_sensitivity_large_l, dice_similarity_coefficient_large_l, precision_lesion_wise_large_l, recall_lesion_wise_large_l,
    #          f1_lesion_wise_large_l, dice_2_large_l, aji_large_l, dq_large_l, sq_large_l, pq_large_l]):
    #     if idx == 0:
    #         metric_name = "large_dice_sensitivity"
    #     elif idx == 1:
    #         metric_name = "large_dice_similarity_coefficient"
    #     elif idx == 2:
    #         metric_name = "large_precision"
    #     elif idx == 3:
    #         metric_name = "large_recall"
    #     elif idx == 4:
    #         metric_name = "large_f1"
    #     elif idx == 5:
    #         metric_name = "large_dice_2"
    #     elif idx == 6:
    #         metric_name = "large_aji"
    #     elif idx == 7:
    #         metric_name = "large_dq"
    #     elif idx == 8:
    #         metric_name = "large_sq"
    #     elif idx == 9:
    #         metric_name = "large_pq"
    #     print(f"{metric_name}_mean: {metric.mean(0).item()}")
    #     print(f"{metric_name}_median: {metric.median(0)[0].item()}")
    #     print(f"{metric_name}_std: {metric.std(0).item()}")
    #     print(f"{metric_name}_min: {metric.min(0)[0].item()}")
    #     print(f"{metric_name}_max: {metric.max(0)[0].item()}")
    #     print(f"{metric_name}_min_idx: {metric.min(0)[1].item()}")
    #     print(f"{metric_name}_max_idx: {metric.max(0)[1].item()}")
        #
        # print(predict)
    # a = torch.randint(0, 2, (16, 50, 300, 300)).type(torch.FloatTensor)
    # loss = DiceLoss(torch.device("cuda:0"))
    # print(loss(a, a))