import xml.etree.ElementTree as ET
import numpy as np
import cv2
from dataclasses import dataclass
import pydicom as pdc
import os
import pandas as pd

@dataclass
class Contour(object):
    slice_number: int
    points: []

def fill_image(image, coordinates, color=1):
    cv2.fillPoly(image, coordinates, color=color)
    return image

def save_image(file_name, contours, raw_imgs):
    imgs = raw_imgs.copy()
    for contour in contours:
        slice_number = int(contour.slice_number)
        fill_image(imgs[slice_number], [np.int32(np.stack(contour.points))])
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
            x = float(point_str.split(",")[0])
            y = float(point_str.split(",")[1])
            points.append([x ,y])
        contour = Contour(slice_number, points)
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
            print(file_name)
            continue
        outputs.append(data)
    np_outputs = np.array(outputs)
    return np_outputs


if __name__ == '__main__':
    file_names = pd.read_csv("/home/longlh/PycharmProjects/roi_segmentation/roi_segmentation_dataset.csv", header=None,
                             names=["file_names", "type"])["file_names"]
    a = []
    for file_name in file_names:
        input = np.load(f"{file_name}/input.npy")
        a.append((input.shape[1], input.shape[2]))
    from collections import Counter
    print(Counter(a))
    # for idx, file_name in enumerate(file_names):
    #     print(f"{idx} / {len(file_names)}")
    #     input = load_dicom(file_name)
    #     contours = convert_xml_to_contours(f"{file_name}/VOI2.xml")
    #     np.save(f"{file_name}/input.npy", input)
    #     raw_imgs = np.zeros((input.shape))
    #     save_image(f"{file_name}/gt.npy", contours, raw_imgs)
    # for shape in shapes:
    #     print(shape)
    # raw_imgs = load_dicom("/Users/LongLH/PycharmProjects/roi_segmentation/ADC")
    # print(np.unique(raw_imgs))
    # contours = convert_xml_to_contours("/Users/LongLH/PycharmProjects/roi_segmentation/ADC/VOI2.xml")
    # save_image("/Users/LongLH/PycharmProjects/roi_segmentation/abc.npy", contours, raw_imgs)


