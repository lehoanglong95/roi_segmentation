import xml.etree.ElementTree as ET
import numpy as np
import cv2
from dataclasses import dataclass
import pydicom as pdc
import os
import pandas as pd
import SimpleITK as sitk
from skimage.draw import polygon_perimeter, polygon
from pathlib import Path
from multiprocessing import Pool

@dataclass
class Contour(object):
    slice_number: int
    points: np.ndarray

def fill_image(image, coordinates, color=1):
    r = coordinates[:, 0]
    c = coordinates[:, 1]
    rr, cc = polygon(c, r)
    image[rr, cc] = color
    return image

def save_image(file_name, contours, raw_imgs):
    imgs = raw_imgs.copy()
    # print(imgs.shape)
    for contour in contours:
        slice_number = int(contour.slice_number)
        fill_image(imgs[slice_number], np.int32(contour.points))
    # print(np.count_nonzero(np.array(imgs)))
    np.save(f"{file_name}", np.array(imgs))

def convert_xml_to_contours(file_name):
    contours = []

    tree = ET.parse(file_name)

    root = tree.getroot()
    contours_xml = root.findall("Contour")
    for contour in contours_xml:
        slice_number = contour.find("Slice-number").text
        points = []
        points_xml = contour.findall("Pt")
        for point in points_xml:
            point_str = point.text
            x = int(float(point_str.split(",")[0]))
            y = int(float(point_str.split(",")[1]))
            points.append([x ,y])
        contour = Contour(slice_number, np.array(points))
        contours.append(contour)
    return contours

def load_dicom(file_name: str) -> int:
    outputs = []
    files = os.listdir(file_name)
    dcm_files = sorted([file for file in files if "dcm" in file], key=lambda x: x[:5])
    for dcm_file in dcm_files:
        dataset = pdc.dcmread(f"{file_name}/{dcm_file}")
        try:
            data = dataset.pixel_array
        except:
            image = sitk.ReadImage(f"{file_name}/{dcm_file}")
            data = sitk.GetArrayFromImage(image)[0]
        outputs.append(data)
    np_outputs = np.array(outputs)
    return np_outputs

def process_func(file_name):
    dwi_folder = f"{file_name}/DWI"
    file_name = f"{file_name}/ADC"
    patient = file_name.split("/")[5]
    # if patient in s:
    #     continue
    output_folder = f"/home/longle/ssd_data/brain_lesion_segmentation_clean_data/{patient}"
    os.makedirs(output_folder, exist_ok=True)
    try:
        adc_input = load_dicom(file_name)
        dwi_input = load_dicom(dwi_folder)
    except Exception as e:
        print(e)
        return
    # contours = convert_xml_to_contours(f"{file_name}/VOI2.xml")
    try:
        contours = convert_xml_to_contours(f"{file_name}/RELABEL_VOI.xml")
    except:
        try:
            contours = convert_xml_to_contours(f"{file_name}/VOI2.xml")
        except Exception as e:
            print(e)
            return
    np.save(f"{output_folder}/adc_input.npy", adc_input)
    np.save(f"{output_folder}/dwi_input.npy", dwi_input)
    raw_imgs = np.zeros((adc_input.shape), dtype=np.int32)
    save_image(f"{output_folder}/gt.npy", contours, raw_imgs)
if __name__ == '__main__':
    l = os.listdir("/home/longle/long_data_1/CMC AI Auto Stroke VOL _Training")
    file_names = ["/home/longle/long_data_1/CMC AI Auto Stroke VOL _Training/" + e for e in l]
    p = Pool(16)
    p.map(func=process_func, iterable=file_names)
    p.close()