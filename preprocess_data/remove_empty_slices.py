import numpy as np
import os
import csv

if __name__ == '__main__':
    base_dir = "/home/longle/long_data/brain_lesion_segmentation_clean_data"
    l = []
    with open("/home/longle/long_data/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        for row in csv_reader:
            l.append(row[0].split("/")[-1])
    for d in l:
        try:
            gt = np.load(f"{base_dir}/{d}/after_registration_gt.npy")
            adc = np.load(f"{base_dir}/{d}/after_registration_adc.npy")
            dwi = np.load(f"{base_dir}/{d}/after_registration_dwi.npy")
            new_adc = adc[~(adc==0).all((2,1))]
            new_dwi = dwi[~(dwi == 0).all((2, 1))]
            np.save(f"{base_dir}/{d}/after_registration_no_empty_slices_gt.npy", gt[:len(new_adc)])
            np.save(f"{base_dir}/{d}/after_registration_no_empty_slices_adc.npy", new_adc)
            np.save(f"{base_dir}/{d}/after_registration_no_empty_slices_dwi.npy", new_dwi)
        except Exception as e:
            print(e)
            print(d)