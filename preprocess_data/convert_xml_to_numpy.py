import xml.etree.ElementTree as ET
import numpy as np
import cv2
from dataclasses import dataclass
import pydicom as pdc
import os

@dataclass
class Contour(object):
    slice_number: int
    points: []

def fill_image(image, coordinates, color=5000):
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

def load_dicom(file: str) -> np.ndarray:
    outputs = []
    files = os.listdir(file)
    dcm_files = [file for file in files if "dcm" in file]
    for dcm_file in dcm_files:
        dataset = pdc.dcmread(f"{file}/{dcm_file}")
        data = dataset.pixel_array
        outputs.append(data)
    np_outputs = np.array(outputs)
    # np_outputs[np_outputs > 1] = 2
    return np_outputs

if __name__ == '__main__':
    raw_imgs = load_dicom("/Users/LongLH/PycharmProjects/roi_segmentation/ADC")
    # print(np.unique(raw_imgs))
    contours = convert_xml_to_contours("/Users/LongLH/PycharmProjects/roi_segmentation/ADC/VOI2.xml")
    save_image("/Users/LongLH/PycharmProjects/roi_segmentation/abc.npy", contours, raw_imgs)


