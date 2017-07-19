#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Jun 12 2017
@author: pavle
@e-mail: pavle_yao@yahoo.com

This script use the trained model to predictnodules.
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

from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import func


CUBE_SIZE = 32
USE_DROPOUT = False
LEARN_RATE = 0.001

PREDICT_STEP = 12
USE_DROPOUT = False

MEAN_PIXEL_VALUE = 41

NEGS_PER_POS = 20
P_TH = 0.6

step = PREDICT_STEP
CROP_SIZE = CUBE_SIZE


def load_patient_images(patient_id, base_dir=None, wildcard="*.*", exclude_wildcards=[]):
    src_dir = base_dir + patient_id
    src_img_paths = glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1, ) + im.shape) for im in images]
    res = np.vstack(images)
    return res

def prepare_image_for_net3D(img):
    img = img.astype(np.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img

def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False, mal=False):
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), border_mode="same")(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(x)

    # 2nd layer group
    x = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.3)(x)

    # 3rd layer group
    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.4)(x)

    # 4th layer group
    x = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1),)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.5)(x)

    last64 = Convolution3D(64, 2, 2, 2, activation="relu", name="last_64")(x)
    out_class = Convolution3D(1, 1, 1, 1, activation="sigmoid", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)

    out_malignancy = Convolution3D(1, 1, 1, 1, activation=None, name="out_malignancy_last")(last64)
    out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    model = Model(input=inputs, output=[out_class, out_malignancy])
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_class": "binary_crossentropy", "out_malignancy": mean_absolute_error}, metrics={"out_class": [binary_accuracy, binary_crossentropy], "out_malignancy": mean_absolute_error})

    if features:
        model = Model(input=inputs, output=[last64])
    model.summary(line_length=140)

    return model


def filter_patient_nodules_predictions(df_nodule_predictions, patient_id, view_size, luna16=False):
    src_dir = settings.LUNA_16_TRAIN_DIR2D2 if luna16 else settings.NDSB3_EXTRACTED_IMAGE_DIR
    patient_mask = load_patient_images(patient_id, src_dir, "*_m.png")
    delete_indices = []
    for index, row in df_nodule_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

        mal_score = row["diameter_mm"]
        start_y = center_y - view_size / 2
        start_x = center_x - view_size / 2
        nodule_in_mask = False
        for z_index in [-1, 0, 1]:
            img = patient_mask[z_index + center_z]
            start_x = int(start_x)
            start_y = int(start_y)
            view_size = int(view_size)
            img_roi = img[start_y:start_y+view_size, start_x:start_x + view_size]
            if img_roi.sum() > 255:  # more than 1 pixel of mask.
                nodule_in_mask = True

        if not nodule_in_mask:
            print("Nodule not in mask: ", (center_x, center_y, center_z))
            if mal_score > 0:
                mal_score *= -1
            df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
        else:
            if center_z < 30:
                print("Z < 30: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score


            if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                print("SUSPICIOUS FALSEPOSITIVE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

            if center_z < 50 and y_perc < 0.30:
                print("SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

    df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
    return df_nodule_predictions


def main():
    base_path = 'xxxx/tianchi/DeepLuna/test&val/'#contains all data folds
    csv_target_path = base_path
    model_path = "xxxx/trained_models/model_luna16_full__fs_best.hd5"
    model = get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=model_path)
    all_predictions_csv = []

    data_paths =[]
    for file_name in os.listdir(base_path):
        if not os.path.isdir(base_path + file_name):
            continue
        data_paths.append(file_name)

    annotation_index = 0
    for i, data_path in enumerate(data_paths):
        data_path = base_path + data_path

        dst_dir = data_path + '_cube/'
        dst_dir_pic = data_path + '_pic/'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        if not os.path.exists(dst_dir_pic):
            os.makedirs(dst_dir_pic)

        file_list=glob(data_path+"/*.mhd")
        
        #predict a lung each time
        for img_file in file_list:
            print("working on ", img_file)
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)
            origin = np.array(itk_img.GetOrigin())
            origin = origin[::-1]
            spacing = np.array(itk_img.GetSpacing())
            spacing = spacing[::-1]#img_array is ordered by z,y,x

            pix_resampled, spacing = func.resample(img_array, spacing)
            print("Shape before resampling\t", img_array.shape)
            print("Shape after resampling\t", pix_resampled.shape)

            patient_id = img_file.split('/')[-1][0:10] + '/'
            if not os.path.exists(dst_dir_pic + patient_id):
                os.makedirs(dst_dir_pic + patient_id)

            #save all 2d image and mask-image
            for i in range(pix_resampled.shape[0]):
                img = pix_resampled[i]
                seg_img, mask = func.get_segmented_lungs(img.copy())
                img = normalize(img)
                
                cv2.imwrite(dst_dir_pic + patient_id + str(i).rjust(4, '0') + "_i.png", img * 255)
                cv2.imwrite(dst_dir_pic + patient_id + str(i).rjust(4, '0') + "_m.png", mask * 255)

            #read 2d image and mask-image
            img_array = load_patient_images(patient_id, dst_dir_pic, "*_i.png", [])
            mask_array = load_patient_images(patient_id, dst_dir_pic, "*_m.png", [])

            #calculate cube number in a lung
            predict_volume_shape_list = [0, 0, 0]
            for dim in range(3):
                dim_indent = 0
                while dim_indent + CROP_SIZE < img_array.shape[dim]:
                    predict_volume_shape_list[dim] += 1
                    dim_indent += step

            predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
            predict_volume = np.zeros(shape=predict_volume_shape, dtype=float)
            print("Predict volume shape: ", predict_volume.shape)
            
            predict_count = 0
            skipped_count = 0
            batch_size = 128
            batch_list = []
            batch_list_coords = []
            #patient_predictions_csv = []
            cube_img = None

            if not os.path.exists(dst_dir + patient_id):
                os.makedirs(dst_dir + patient_id)
            for z in range(0, predict_volume_shape[0]):
                for y in range(0, predict_volume_shape[1]):
                    for x in range(0, predict_volume_shape[2]):
                        #if cube_img is None:
                        cube_num = 'xyz_' + str(x).rjust(4, '0') + '_' + str(y).rjust(4, '0') + '_' + str(z).rjust(4, '0')
                        cube_img = img_array[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                        cube_mask = mask_array[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                        np.save(dst_dir + patient_id + cube_num + '.npy', cube_img)
                        np.save(dst_dir + patient_id + cube_num + '_mask.npy', cube_mask)

                        if cube_mask.sum() < 2000:
                            skipped_count += 1
                            #print('Skip: ', z,y,x)
                        else:
                            #predict a cube here
                            predict_count += 1
                            #print('Predict: ', z,y,x)
                            
                            img_prep = prepare_image_for_net3D(cube_img)
                            batch_list.append(img_prep)
                            batch_list_coords.append((z, y, x))
                            #print(len(batch_list))
                            if len(batch_list) % batch_size == 0:
                                batch_data = np.vstack(batch_list)
                                p = model.predict(batch_data, batch_size=batch_size)
                                for i in range(len(p[0])):
                                    p_z = batch_list_coords[i][0]
                                    p_y = batch_list_coords[i][1]
                                    p_x = batch_list_coords[i][2]
                                    nodule_chance = p[0][i][0]
                                    predict_volume[p_z, p_y, p_x] = nodule_chance
                                    if nodule_chance > P_TH:
                                        p_z = p_z * step + CROP_SIZE / 2
                                        p_y = p_y * step + CROP_SIZE / 2
                                        p_x = p_x * step + CROP_SIZE / 2
                                        p_z, p_y, p_x = func.voxel_to_world([p_z,p_y,p_x], origin, spacing)

                                        #p_z_perc = round(float(p_z) / img_array.shape[0], 4)
                                        #p_y_perc = round(float(p_y) / img_array.shape[1], 4)
                                        #p_x_perc = round(float(p_x) / img_array.shape[2], 4)
                                        diameter_mm = round(p[1][i][0], 4)
                                        # diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                        diameter_perc = round(2 * step / img_array.shape[2], 4)
                                        diameter_perc = round(diameter_mm / img_array.shape[2], 4)
                                        nodule_chance = round(nodule_chance, 4)
                                        patient_predictions_csv_line = [annotation_index, p_x, p_y, p_z, diameter_perc, nodule_chance, diameter_mm]
                                        #patient_predictions_csv.append(patient_predictions_csv_line)
                                        all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                        annotation_index += 1
                                batch_list = []
                                batch_list_coords = []

            print ('Lung %s is prdicted.', patient_id)
            print ('Skip %d cubes, predict %d cubes in lung %s.', skipped_count, predict_count, predict_count)


    df = pd.DataFrame(all_predictions_csv, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"])
    df.to_csv(csv_target_path + 'all_nodules.csv', index=False)
    filter_patient_nodules_predictions(df, patient_id, CROP_SIZE)
    df.to_csv(csv_target_path + 'filtered_nodules.csv', index=False)


if __name__ == "__main__":
    main()
