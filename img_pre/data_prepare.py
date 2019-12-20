# coding=utf8

import numpy as np 
import SimpleITK as sitk
import csv
from glob import glob
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import traceback
import sys
import random
import re
from pprint import pprint
from PIL import Image


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 400] = 0
    image_array[image_array <-1000] = 0

# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(itkimage.GetOrigin()))  # CT原点坐标
    numpySpacing = np.array(list(itkimage.GetSpacing()))  # CT像素间隔
    return numpyImage, numpyOrigin, numpySpacing

def extract_real_cubic_from_mhd(dcim_path,annatation_file,node_path,normalization_output_path):
    '''
      @param: dcim_path :                 the path contains all mhd file
      @param: annatation_file:            the annatation csv file,contains every nodules' coordinate
      @param: plot_output_path:           the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(plot ),every nodule end up withs three size
      @param:normalization_output_path:   the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(after normalization)
    '''
    file_list = glob(dcim_path + "*.mhd")
    file_list_buf = []
    for f in file_list:
        file_list_buf.append(f.replace("\\", "/"))
    file_list = file_list_buf
    # The locations of the nodes
    df_node = pd.read_csv(annatation_file)
    # print(df_node)
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    
    kkk=0
    for img_index, img_file in df_node.iterrows():
        # print(img_file["file"])
        # print(df_node["file"])
        # mini_df = df_node[df_node["file"]==img_file["file"]] #get all nodules associate with file
        # sys.exit(0)
        # file_name = os.path.basename(img_file["file"])   # return *.mhd
        node_x = img_file["coordX"]
        node_y = img_file["coordY"]
        node_z = img_file["coordZ"]
        # node_dia = img_file["diameter_mm"]
        node_file = img_file["file"]
        # node_dia = int(node_dia / 2)
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(node_file)
        if numpyImage.shape[0]>0: # some files may not have a nodule--skipping those

            print("begin to process real nodules...")
            img_array = numpyImage.transpose(2, 1, 0)      # take care on the sequence of axis of v_center ,transfer to x,y,z
            # nodule_pos_str = str(node_x)+"_"+str(node_y)+"_"+str(node_z)
                # every nodules saved into size of 20x20x6,30x30x10,40x40x26
            # imgs1 = np.ndarray([20,20,6],dtype=np.float32)
            # imgs2 = np.ndarray([30,30,10],dtype=np.float32)
            # imgs3 = np.ndarray([40,40,26],dtype=np.float32)
            center = np.array([node_x, node_y, node_z])   # nodule center
            # print(center)
            v_center = np.rint((center - numpyOrigin) / numpySpacing)  # nodule center in voxel space (still x,y,z ordering)
            # print(img_index, v_center[0], v_center[1], v_center[2])
            try:
                img1 = img_array[int(v_center[0] - 20) : int(v_center[0] + 20), int(v_center[1] - 20) : int(v_center[1] + 20), int(v_center[2] - 3) : int(v_center[2] + 3)]
                img2 = img_array[int(v_center[0] - 19) : int(v_center[0] + 21), int(v_center[1] - 20) : int(v_center[1] + 20), int(v_center[2] - 3) : int(v_center[2] + 3)]
                img3 = img_array[int(v_center[0] - 20) : int(v_center[0] + 20), int(v_center[1] - 19) : int(v_center[1] + 21), int(v_center[2] - 3) : int(v_center[2] + 3)]
                img4 = img_array[int(v_center[0] - 20) : int(v_center[0] + 20), int(v_center[1] - 20) : int(v_center[1] + 20), int(v_center[2] - 2) : int(v_center[2] + 4)]
                img5 = img_array[int(v_center[0] - 21) : int(v_center[0] + 19), int(v_center[1] - 20) : int(v_center[1] + 20), int(v_center[2] - 3) : int(v_center[2] + 3)]
                img6 = img_array[int(v_center[0] - 20) : int(v_center[0] + 20), int(v_center[1] - 21) : int(v_center[1] + 19), int(v_center[2] - 3) : int(v_center[2] + 3)]
                img7 = img_array[int(v_center[0] - 20) : int(v_center[0] + 20), int(v_center[1] - 20) : int(v_center[1] + 20), int(v_center[2] - 4) : int(v_center[2] + 2)]
                kkk += 1
                print("===========new node===========" + str(img_index))
                # for shape_i in range(0, img1.shape[2] - 1):
                #     plt.figure(shape_i + 1)
                #     con_img = img1[..., shape_i]
                #     plt.imshow(con_img,cmap='gray')
                #     plt.show()
                np.save(os.path.join(node_path, "images_true_1_%d.npy" % img_index), img1)
                np.save(os.path.join(node_path, "images_true_2_%d.npy" % img_index), img2)
                np.save(os.path.join(node_path, "images_true_3_%d.npy" % img_index), img3)
                np.save(os.path.join(node_path, "images_true_4_%d.npy" % img_index), img4)
                np.save(os.path.join(node_path, "images_true_5_%d.npy" % img_index), img5)
                np.save(os.path.join(node_path, "images_true_6_%d.npy" % img_index), img6)
                np.save(os.path.join(node_path, "images_true_7_%d.npy" % img_index), img7)

                # img1 = img_array[int(v_center[0] - 20) : int(v_center[0] + 20), int(v_center[1] - 20) : int(v_center[1] + 20), int(v_center[2] - 3) : int(v_center[2] + 3)]
                # kkk += 1
                # print("===========new node===========" + str(img_index))
                # for shape_i in range(0, img1.shape[2] - 1):
                #     plt.figure(shape_i + 1)
                #     con_img = img1[..., shape_i]
                #     plt.imshow(con_img,cmap='gray')
                #     plt.show()
                # np.save(os.path.join(node_path, "images_fake_%d.npy" % img_index), img1)

            except Exception as e:
                print(" process images %s error..."%str(node_file))
                print(Exception,":",e)
                traceback.print_exc()


if __name__ == "__main__":
    # for i in range(0, 9):
    #     dcim_path = base_dir + 'subset' + str(i) + "/"
    # # print("extracting image into %s"%normalazation_output_path)
    #     extract_real_cubic_from_mhd(dcim_path, annatation_file, node_path,normalazation_output_path)
    base_dir = '../dataset/'
    normalazation_output_path = '../nor_np'
    annatation_file = '../dataset/annotations.csv'
    for i in range(0, 9):
        # print('here')
        dcim_path = base_dir + 'subset' + str(i) + '/'
        node_path = '../train_node_real40'
        extract_real_cubic_from_mhd(dcim_path, annatation_file, node_path, normalazation_output_path)
    # dcim_path = base_dir + 'subset' + str(9) + '/'
    # node_path = './test_node_real'
    # extract_real_cubic_from_mhd(dcim_path, annatation_file, node_path, normalazation_output_path)



    # img_array = numpyImage.transpose(2,1,0)      # take care on the sequence of axis of v_center ,transfer to x,y,z
    # print(img_array.shape)
    # slice = 538

    # for i in range(0, slice):
    #     plt.figure(i+1)
    #     image = np.squeeze(numpyImage[i, ...])  # if the image is 3d, the slice is integer
    #     plt.imshow(image,cmap='gray')
    #     plt.show()

    # annatation_file = 'I:/LUNA16/annotations.csv'
    # df_node = pd.read_csv(annatation_file)
    # print(df_node)
