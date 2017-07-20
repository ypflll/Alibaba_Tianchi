#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on May 29 2017
@author: pavle
@e-mail: pavle_yao@yahoo.com

This script is used to get positive samples(nodule cube).
"""

import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from glob import glob
import csv
import pandas as pd
import cv2
import func

import time
from tqdm import *


def main():
    #Global Setting
    label_path = 'xxxx/csv/train/'
    data_path = 'xxxx/train_subset00/'

    global file_list
    file_list=glob(data_path+"*.mhd")

    #read the annotations.csv that contains the nodules info
    df_node = pd.read_csv(label_path+"annotations.csv")
    df_node["file"] = df_node["seriesuid"].apply(func.get_filename)
    df_node = df_node.dropna()
    
    for img_file in file_list:
        mini_df = df_node[df_node["file"] == img_file] #get all nodules associate with file
        if len(mini_df)>0:       # some files may not have a nodule--skipping those
            #read the mhd file and get all needed info
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)
            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())
            spacing = spacing[::-1]#img_array is ordered by z,y,x
            
            pix_resampled, spacing = func.resample(img_array, spacing)
            print("Shape before resampling\t", img_array.shape)
            print("Shape after resampling\t", pix_resampled.shape)

            #make sure the orientation is right
            name = str(mini_df["seriesuid"].values)[6:12] + '-'
            nodule_number = mini_df["number"].values
            z,y,x = pix_resampled.shape
            masked_nodule = np.zeros([z,y,x],np.int16)
            mask = np.zeros([z,y,x],np.int16)

            for i in range(len(mini_df)):
                node_x = mini_df["coordX"].values
                node_y = mini_df["coordY"].values
                node_z = mini_df["coordZ"].values
                nodule_num = nodule_number[i]
                
                diameter = mini_df["diameter_mm"].values
                center = np.array([node_x, node_y, node_z])
                center = center[:,i] - origin#x,y,z
                center = [center[2],center[1],center[0]]#pix_resampled is z,y,x order

                window_size = np.array([32, 32, 32])
                zyx_1 = np.round(center - window_size)  # Z, Y, X
                zyx_1 = zyx_1.astype(np.int32)
                zyx_2 = np.round(center + window_size)
                zyx_2 = zyx_2.astype(np.int32)

                nodule_crop = func.get_cube_from_img(pix_resampled, center[2], center[1], center[0], 64)
                mask[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]] = 1

                window_size =  diameter[i] / 2
                zyx_1 = np.round(center - window_size)  # Z, Y, X
                zyx_1 = zyx_1.astype(np.int32)
                zyx_2 = np.round(center + window_size)
                zyx_2 = zyx_2.astype(np.int32)

                masked_nodule[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]] = pix_resampled[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
                np.save('./3D_nodule_and_mask/nodule' + name + '0'*(4-len(str(nodule_num))) + str(nodule_num) + '.npy', nodule_crop)
                np.save('./3D_nodule_and_mask/mask' + name + '0'*(4-len(str(nodule_num))) + str(nodule_num) + '.npy', mask)
                np.save('./3D_nodule_and_mask/masked_nodule' + name + '0'*(4-len(str(nodule_num))) + str(nodule_num) + '.npy', masked_nodule)
                plot_3d(masked_nodule, './3D_nodule_and_mask/masked_nodule' + name + '0'*(4-len(str(nodule_num))) + str(nodule_num))

    print 'Done.'


if __name__ == "__main__":
    main()