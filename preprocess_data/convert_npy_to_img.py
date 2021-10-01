import numpy as np
from skimage.filters import threshold_triangle
from skimage import morphology
import os
import cv2
from multiprocessing import Pool
import pandas as pd
from collections import defaultdict

def preprocess_input_img(input_folder):
    file_names = ["adc_input.npy", "dwi_input.npy", "gt.npy"]
    try:
        standard_shape = np.load(f"{input_folder}/{file_names[0]}").shape
    except:
        return ""
    output = defaultdict(list)
    for file_name in file_names:
        temp = np.load(f"{input_folder}/{file_name}")
        try:
            assert temp.shape == standard_shape
        except:
            return ""
        if "input" in file_name:
            for idx, img in enumerate(temp):
                img = img - np.min(img)
                img = img / np.max(img)
                img = (img * 255).astype(np.uint8)
                threshold = threshold_triangle(img)
                mask = img > threshold
                mask = morphology.remove_small_objects(mask, 100)
                img *= mask
                temp_name = file_name.replace("input", "image")
                temp_name = temp_name.replace(".npy", "") + f"_{idx}"
                cv2.imwrite(f"{input_folder}/{temp_name}.png", img)
                if "adc" in file_name:
                    output["adc"].append(f"{input_folder}/{temp_name}.png")
                else:
                    output["dwi"].append(f"{input_folder}/{temp_name}.png")
        else:
            for idx, img in enumerate(temp):
                temp_name = file_name.replace(".npy", "") + f"_slice_{idx}"
                np.save(f"{input_folder}/{temp_name}.npy", img)
                output["gt"].append(f"{input_folder}/{temp_name}.npy")
    return output

if __name__ == '__main__':
    base_dir = "/data1/long/data/brain_lesion_segmentation_clean_data"
    input_folders = os.listdir(base_dir)
    file_names = [f"{base_dir}/" + e for e in os.listdir(base_dir)]
    p = Pool(16)
    results = p.map(func=preprocess_input_img, iterable=file_names)
    p.close()
    output_df = pd.DataFrame(columns=["adc", "dwi", "gt"])
    # # print(results)
    for result in results:
        if result != "":
            temp = pd.DataFrame(data=result, columns=["adc", "dwi", "gt"])
            output_df = output_df.append(temp)
    output_df.to_csv("../input_files.csv", index=False)