import pandas as pd
import numpy as np
from utils.register import ADC
from collections import defaultdict

if __name__ == '__main__':
    folder = "/home/longlh/hard_2/PycharmProjects/roi_segmentation/test_dataset.csv"
    # folder = "/home/longlh/hard_2/PycharmProjects/registration/error_file.csv"
    file_names = pd.read_csv(f"{folder}", header=None,
                             names=["file_names", "type"])["file_names"]
    print(len(file_names))
    a = defaultdict(list)
    b = []
    for file in file_names:
        patient = file.split("/")[5]
        gt = np.load(f"{file}/ADC/gt.npy")
        number_of_voxels = np.count_nonzero(gt)
        adc = ADC(f"{file}/ADC")
        a[patient].append(number_of_voxels * adc.spacing[0] * adc.spacing[1] * adc.spacing[2])
        new_file = file.replace("CMC AI Auto Stroke VOL _Training", "roi_numpy")
        new_gt = np.load(f"{new_file}/ADC/sampled_gt.npy")
        new_number_of_voxels = np.count_nonzero(new_gt)
        a[patient].append(new_number_of_voxels * 3)
        b.append(abs(a[patient][0] - a[patient][1]))
    for k, v in a.items():
        print(f"{k}, {v[0]}, {v[1]}")
    print(sum(b) / len(b))
