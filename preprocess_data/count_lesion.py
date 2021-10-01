from skimage import measure
import numpy as np
import os
import csv

if __name__ == '__main__':
    base_dir = "/home/longlh/hard_2/roi_numpy"
    # dirs = os.listdir(base_dir)
    dirs = []
    with open("/home/longlh/hard_2/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if int(row[1]) == 2:
                d = row[0].replace("/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training", "/home/longlh/hard_2/roi_numpy")
                dirs.append(d)
    with open('/home/longlh/hard_2/PycharmProjects/roi_segmentation/testing_data_lesion_metadata.csv', mode='w') as lesion_file:
        lesion_writer = csv.writer(lesion_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for d in dirs:
            try:
                gt = np.load(f"{d}/ADC/sampled_gt.npy")
                for i in range(gt.shape[0]):
                    slice_gt = gt[i]
                    if 1 in np.unique(slice_gt):
                        blobs_labels = measure.label(slice_gt)
                        for j in range(len(np.unique(blobs_labels)) - 1):
                            # ignore background = 0
                            number_of_pixel_in_lesion = np.sum(blobs_labels == j + 1)
                            # patient_name, slice, lesion_name, number_of_pixel
                            lesion_writer.writerow([d, str(i), str(j + 1), str(number_of_pixel_in_lesion)])
            except:
                continue


