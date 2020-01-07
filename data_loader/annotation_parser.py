import cv2
from xml.etree import ElementTree as ET
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import random
class AnnotatedInfo:
    def __init__(self):
        self.folder = ""
        self.filename = ""
        self.filepath = ""
        self.depth = 0
        self.label = -1
        self.name = 0
        self.bbox = np.zeros(4, dtype=np.int32)
        
def parseAnnotationXML(xml_path):
    root = ET.parse(xml_path).getroot()
    info = AnnotatedInfo()
    info.filename = root.find("filename").text
    info.folder = root.find("folder").text
    info.depth = int(root.find("size/depth").text)
    #label.segmented = int(root.find("segmented").text)
    obj = root.find("object")
    info.name = int(obj.find("name").text)
    bndbox = obj.find("bndbox")
    
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    info.bbox[0] = xmin
    info.bbox[1] = ymin
    info.bbox[2] = xmax - xmin
    info.bbox[3] = ymax - ymin
    return info

def loadAnnotations(label_path):
    info_dict = {}
    for xml_file in glob.glob(label_path+"/*.xml"):
        info = parseAnnotationXML(xml_file)
        info_dict[info.folder] = info
    return info_dict

def updateLabels(labeled_data_path, info_dict, label):   
    name_list = []
    for folder in os.listdir(labeled_data_path):
        if folder in info_dict:
            info_dict[folder].label = label
            info_dict[folder].filepath = labeled_data_path + "/" + folder + "/" + info_dict[folder].filename
            name_list.append(folder)
    return name_list

def loadDatasetInfo(benign_label_path, malignant_label_path, malignant_data_path, benign_data_path):
    info_dict = loadAnnotations(benign_label_path) # malignant_label_path is not accurate?
    info_dict.update(loadAnnotations(malignant_label_path))
    malignant_names = updateLabels(malignant_data_path, info_dict, 1)
    benign_names = updateLabels(benign_data_path, info_dict, 0)
    return info_dict, benign_names, malignant_names