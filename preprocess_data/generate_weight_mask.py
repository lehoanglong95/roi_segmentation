from skimage import measure
import numpy as np
import os
import csv

if __name__ == '__main__':
    base_dir = "/home/longlh/hard_2/roi_numpy"
    dirs = []
    with open("/home/longlh/hard_2/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        for row in csv_reader:
            dirs.append(row[0].split("/")[-1])
    # dirs = ["BP544 20160307"]
    for d in dirs:
        print(d)
        try:
            gt = np.load(f"{base_dir}/{d}/ADC/non_empty_sampled_gt.npy")
            mask = np.load(f"{base_dir}/{d}/ADC/mask.npy")
            gt = gt * mask
            number_of_pixels_per_slice = gt.shape[1] * gt.shape[2]
            output = np.ones(gt.shape)
            for i in range(gt.shape[0]):
                slice_gt = gt[i]
                number_of_pixel_in_lesion_list = {}
                if 1 in np.unique(slice_gt):
                    blobs_labels = measure.label(slice_gt)
                    for j in range(len(np.unique(blobs_labels)) - 1):
                        # ignore background = 0
                        number_of_pixel_in_lesion_list[(j+1)] = np.sum(blobs_labels == j + 1)
                for idx, number_of_pixel_per_lesion in number_of_pixel_in_lesion_list.items():
                    weight = np.log(1 / (number_of_pixel_per_lesion / number_of_pixels_per_slice))
                    output[i][blobs_labels == idx] = weight
            np.save(f"{base_dir}/{d}/ADC/weight_mask.npy", output)
        except Exception as e:
            print(e)


