#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on May 29 2017
@author: pavle
@e-mail: pavle_yao@yahoo.com

This script is used to get positive samples(nodule cube).
"""


import scipy.ndimage
import SimpleITK as sitk
import numpy as np
import pandas as pd
import ntpath
import cv2
import shutil
import random
import math
import os
from glob import glob

from collections import defaultdict
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import func



def main():

    #Global Setting
    label_path = 'xxxx/csv/train/'
    data_path = 'xxxx/train_subset00/'
    dst_dir = 'xxxx/samples/'

    global file_list
    file_list=glob(data_path+"*.mhd")

    #read the annotations.csv that contains the nodules info
    df_node = pd.read_csv(label_path+"annotations.csv")
    df_node["file"] = df_node["seriesuid"].apply(func.get_filename)
    df_node = df_node.dropna()
    
    sample_num = 0
    for img_file in file_list:
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())
        spacing = spacing[::-1]#img_array is ordered by z,y,x
        
        pix_resampled, spacing = func.resample(img_array, spacing)
        print("Shape before resampling\t", img_array.shape)
        print("Shape after resampling\t", pix_resampled.shape)

        img_list = []
        mask_list = []
        for i in range(pix_resampled.shape[0]):
            img = pix_resampled[i]
            seg_img, mask = func.get_segmented_lungs(img.copy())
            mask_list.append(mask)
            img = func.normalize(img)
            
            cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", mask * 255)
            img = cv2.imread(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png",0)
            img_list.append(img)
            #cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)
        
        print 'Segmentation done.'

        sample_num = 0
        tries = 0
        while sample_num < 2 and tries < 100:
            coord_z = int(np.random.normal(len(img_list) / 2, len(img_list) / 6))
            coord_z = max(coord_z, 0)
            coord_z = min(coord_z, len(img_list) - 1)
            candidate_map = img_list[coord_z]
            candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)

            non_zero_indices = np.nonzero(candidate_map)
            if len(non_zero_indices[0]) == 0:
                tries = 100
                print 'No edge found.'
                continue

            nonzero_index = random.randint(0, len(non_zero_indices[0]) - 1)
            coord_y = non_zero_indices[0][nonzero_index]
            coord_x = non_zero_indices[1][nonzero_index]

            ok = True
            #check the distance with nodules
            for index, row in df_node.iterrows():
                pos_coord_x = row["coordX"]
                pos_coord_y = row["coordY"]
                pos_coord_z = row["coordZ"]
                diameter = row["diameter_mm"]
                dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
                if dist < (diameter + 48): #  make sure we have a big margin
                    ok = False
                    print("# Too close", (coord_x, coord_y, coord_z))
                    break
            if not ok:
                continue

            sample_num += 1

            patient_id = ntpath.basename(img_file).replace(".mhd", "")
            neg_cube = func.get_cube_from_img(pix_resampled, coord_x, coord_y, coord_z, 48)
            sample_image = img_list[coord_z].copy()
            print sample_image.shape
            print (coord_x, coord_y, coord_z)

            sample_image[coord_y-24:coord_y+24, coord_x-24:coord_x+24] = 255

            plt.subplot(121)
            plt.imshow(candidate_map,cmap = 'gray')
            plt.subplot(122)
            plt.imshow(pix_resampled[coord_z],cmap = 'gray')
            plt.savefig(dst_dir + patient_id[5:] + '_' + str(sample_num).rjust(4, '0') + '.png')
            #plt.show()
            plt.close()

            print 'Get a negative sample at: %f, %f, %f.' % (coord_x, coord_y, coord_z)
            np.save(dst_dir + 'neg_' + patient_id[5:] + '_' + str(sample_num).rjust(4, '0') + '.npy', neg_cube)

        print "Lung %s is done." % patient_id


if __name__ == "__main__":
    main()


