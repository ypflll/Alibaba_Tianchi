#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Mar 27 2017
@author: pavle
@e-mail: pavle_yao@yahoo.com

This script is mainly based on: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial.
This script read mhd files, resample and segment it, crop the nodule cube and msak cube.
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
import func

import time
from tqdm import *


def main():
    #Global Setting
    label_path = 'xxx/tianchi_data/csv/train/'
    data_path = 'xxx/tianchi_data/train_subset00/'

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

            pix_resampled = func.resample(img_array, spacing)
            print("Shape before resampling\t", img_array.shape)
            print("Shape after resampling\t", pix_resampled.shape)

            segmented_lungs = func.segment_lung_mask(pix_resampled, False)
            segmented_lungs_fill = func.segment_lung_mask(pix_resampled, True)
            lung_masked = pix_resampled * segmented_lungs_fill
            name = str(mini_df["seriesuid"].values)[2:12]
            np.save('./3D_seg_lung/' + name + '.npy', lung_masked)
            spacing = spacing[::-1]#x,y,z
            
            func.plot_3d(segmented_lungs_fill, name, 0)#plot and save 3d masked lungs

            z,y,x = pix_resampled.shape
            print 'pix_resampled.shape',pix_resampled.shape
            nodule_mask = np.zeros([z,y,x],np.int16)

            for i in range(len(mini_df)):
                node_x = mini_df["coordX"].values
                node_y = mini_df["coordY"].values
                node_z = mini_df["coordZ"].values
                diameter = mini_df["diameter_mm"].values
                center = np.array([node_x, node_y, node_z])
                
                print spacing
                print center[:,i]
                center = (center[:,i] - origin)
                center = [center[2],center[1],center[0]]#pix_resampled is z,y,x order
                window_size =  diameter[i] / 2
                
                zyx_1 = np.round(center - window_size)  # Z, Y, X
                zyx_1 = zyx_1.astype(np.int32)
                zyx_2 = np.round(center + window_size + 1)
                zyx_2 = zyx_2.astype(np.int32)

                nodule_crop = pix_resampled[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
                nodule_mask[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]] = pix_resampled[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
                np.save('./3D_seg_lung/' + str(name) + '_' + str(i) + '.npy', nodule_crop)
                #np.save('./3D_seg_lung/' + str(name) + '_' + str(i) + '.npy', nodule_mask)

                #plot the nodule in 2d and save it
                center_slice = (zyx_2[0] + zyx_1[0])/2
                center_crop = (zyx_2[0] - zyx_1[0])/2

                fig,ax = plt.subplots(2,2,figsize=[8,8])
                ax[0,0].imshow(pix_resampled[center_slice,:],cmap='gray')
                ax[0,1].imshow(segmented_lungs_fill[center_slice,:],cmap='gray')
                ax[1,0].imshow(nodule_mask[center_slice,:],cmap='gray')
                ax[1,1].imshow(nodule_crop[center_crop,:],cmap='gray')
                plt.savefig('./3D_seg_lung/' + str(name) + '_' + str(i) + '_2d' + ".png")
                plt.close()

                #plt.show()

                print 'Nodule %d in image file %s is segmented.' % (i, name)

    print 'Done.'

if __name__ == "__main__":
    main()
