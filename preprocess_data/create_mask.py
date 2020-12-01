import numpy as np
import os
from skimage.filters import threshold_otsu

if __name__ == '__main__':
    base_dir = "/home/longlh/hard_2/roi_numpy"
    for d in os.listdir(base_dir):
        ADC_data = np.load(f"{base_dir}/{d}/ADC/sampled_input.npy")
        thresh_ADC = threshold_otsu(ADC_data)
        binary_ADC = ADC_data > thresh_ADC
        DWI_data = np.load(f"{base_dir}/{d}/DWI/sampled_input.npy")
        thresh_DWI = threshold_otsu(DWI_data)
        binary_DWI = DWI_data > thresh_DWI
        np.save(f"{base_dir}/{d}/ADC/mask.npy", binary_ADC)
        np.save(f"{base_dir}/{d}/DWI/mask.npy", binary_DWI)