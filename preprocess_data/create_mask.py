import numpy as np
import os
from skimage.filters import threshold_triangle
from skimage import morphology
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
            ADC_data = np.load(f"{base_dir}/{d}/after_registration_no_empty_slices_adc.npy")
            adc_arr = []
            for adc in ADC_data:
                thresh_ADC = threshold_triangle(adc)
                binary_adc = morphology.remove_small_holes(
                    morphology.remove_small_objects(
                        adc > thresh_ADC, 500
                    ), 500
                )
                binary_adc = morphology.opening(binary_adc)
                adc_arr.append(binary_adc)
            mask_adc = np.array(adc_arr)
            DWI_data = np.load(f"{base_dir}/{d}/after_registration_no_empty_slices_dwi.npy")
            dwi_arr = []
            for dwi in DWI_data:
                thresh_DWI = threshold_triangle(dwi)
                binary_dwi = morphology.remove_small_holes(
                    morphology.remove_small_objects(
                        dwi > thresh_DWI, 500
                    ), 500
                )
                binary_dwi = morphology.opening(binary_dwi)
                dwi_arr.append(binary_dwi)
            mask_dwi = np.array(dwi_arr)
            final_mask = np.copy(mask_adc)
            final_mask[mask_dwi == 1] = 1
            np.save(f"{base_dir}/{d}/adc_mask.npy", mask_adc)
            np.save(f"{base_dir}/{d}/dwi_mask.npy", mask_dwi)
            np.save(f"{base_dir}/{d}/final_mask.npy", final_mask)
        except Exception as e:
            print(e)
            print(d)
# pth = pth.replace("2017%EB%85%84 %EB%8D%B0%EC%9D%B4%ED%84%B0", "2017년 데이터")
