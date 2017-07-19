#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Mar 27 2017
@author: pavle
@e-mail: pavle_yao@yahoo.com

This script read all nodule in world coordinate and trans it to pixel coord.
"""

import os

import math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import csv
from glob import glob
import func

class Nodule:
    def __init__(self):
        self.node_x = 0      
        self.node_y = 0      
        self.node_z = 0      
        self.diam = 0

def get_filename(case):
    for f in file_list:
        if case in f:
            return(f)

def get_origin(file):
    f = open(file)
    a = (f.readlines()[7].split( )[2:])
    return [float(a[0]), float(a[1]), float(a[2])]

def get_spacing(file):
    f = open(file)
    a = (f.readlines()[10].split( )[2:])
    return [float(a[0]), float(a[1]), float(a[2])]

def main():
    #Global Setting
    luna_path = 'xxx/csv/train/'
    luna_subset_path = 'xxx/mhd/'

    nodule = Nodule();

    global file_list
    file_list=glob(luna_subset_path+"*.mhd")

    #read the annotations.csv that contains the nodules info
    df_node = pd.read_csv(luna_path+"annotations.csv")
    df_node["file"] = df_node["seriesuid"].apply(func.get_filename)
    df_node = df_node.dropna()

    rlist = []
    for img_file in file_list:
        mini_df = df_node[df_node["file"] == img_file] #get all nodules associate with file
        if len(mini_df)>0:       # some files may not have a nodule--skipping those
            nodule.node_x = mini_df["coordX"].values
            nodule.node_y = mini_df["coordY"].values
            nodule.node_z = mini_df["coordZ"].values
            nodule.diam = mini_df["diameter_mm"].values

            #read the mhd file and get all needed info
            #itk_img = sitk.ReadImage(img_file)
            #img_array = sitk.GetArrayFromImage(itk_img)
            #origin = np.array(itk_img.GetOrigin())
            #spacing = np.array(itk_img.GetSpacing())
            origin = get_origin(img_file)
            spacing = get_spacing(img_file)
            center = np.array([nodule.node_x, nodule.node_y, nodule.node_z])

            x,y = center.shape #y means the number of nodules in one mhd file
            for i in range(y):
                v_center =np.rint((center[:,i]-origin)/spacing)
                rlist.append([v_center[0,], v_center[1,], v_center[2,], nodule.diam[i], img_file.split('/')[-1]])#this slice contains the center

                slice_num = int(nodule.diam[i] / spacing[2]) / 2 #number of slices that a nodule may appear in
                for k in range(slice_num):
                    diam = 2 * math.sqrt(nodule.diam[i] * nodule.diam[i] / 4 - (k + 1) * (k + 1) * spacing[2] * spacing[2])
                    rlist.append([v_center[0,], v_center[1,], v_center[2,] + k + 1, diam, img_file.split('/')[-1]])
                    rlist.append([v_center[0,], v_center[1,], v_center[2,] - k - 1, diam, img_file.split('/')[-1]])

    csvfile = open('nodule_labeling.csv', 'w')
    writer = csv.writer(csvfile)
    writer.writerows(rlist)
    csvfile.close()

if __name__ == "__main__":
    main()