# choose slice which contains roi
import os
import numpy as np
import cv2
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    df = pd.read_csv("./input_files.csv")
    adc_df = pd.DataFrame(columns=["input", "gt"])
    dwi_df = pd.DataFrame(columns=["input", "gt"])
    for idx, row in df.iterrows():
        gt = np.load(row["gt"])
        if len(np.unique(gt)) > 1:
            data_1 = defaultdict(list)
            data_1["input"].append(row["adc"])
            data_1["gt"].append(row["gt"])
            temp1 = pd.DataFrame(data=data_1, columns=["input", "gt"])
            adc_df = adc_df.append(temp1)
            data_2 = defaultdict(list)
            data_2["input"].append(row["adc"])
            data_2["gt"].append(row["gt"])
            temp2 = pd.DataFrame(data=data_2, columns=["input", "gt"])
            dwi_df = dwi_df.append(temp2)
    adc_df.to_csv("./valid_adc_file.csv", index=False)
    dwi_df.to_csv("./valid_dwi_file.csv", index=False)
