import glob
import cv2
import math
import sys
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

def load_images(file_names_list):

    n_images = len(file_names_list)

    images = [None for i in range(n_images)]
    for i in range(n_images):
        images[i] = (cv2.imread(file_names_list[i]))

    return images


def read_data(data_dir="training/images/"): 

    train_file_names_list = glob.glob(data_dir + "*.tif")
    n_train_images = len(train_file_names_list)

    # The truth images have the same numbers.
    # We change the path and the extensions.
    # This ensures that train image will
    # correspond to truth image.
    truth_file_names_list = [None for i in range(n_train_images)]
    for i in range(n_train_images):

        if data_dir == "training/images/":
            truth_file_names_list[i] = train_file_names_list[i].replace(".tif","_mask.png").replace("images", "truth")
        elif data_dir == "augmented_images/":
            truth_file_names_list[i] = train_file_names_list[i].replace(".tif",".png").replace("train", "truth")
        else:
            pass

    n_truth_images = len(truth_file_names_list)

    for i in range(n_train_images):
        tri = train_file_names_list[i]
        tui = truth_file_names_list[i]
        print(tri + " " + tui)

    train_images = load_images(train_file_names_list)
    truth_images = load_images(truth_file_names_list)

    return train_images, truth_images


def resize_image_list(image_list, n_width, n_height):

    n_images = len(image_list)

    resized_image_list = [None for i in range(n_images)]
    for i in range(n_images):
        resized_image_list[i] = cv2.resize(image_list[i], 
                                           (n_width, n_height), 
                                           interpolation = cv2.INTER_LINEAR)

    return resized_image_list


def convert_image_list_to_grayscale(image_list):
    
    n_images = len(image_list)

    gray_scale_image_list = [None for i in range(n_images)]
    for i in range(n_images):
        gray_scale_image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)

    return gray_scale_image_list


def plot_two_images(bgr, gs):

    rows = 1
    cols = 2
    f, axs = plt.subplots(rows, cols)

    plt.subplot(rows, cols, 1)
    if len(bgr.shape) == 3:
        plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(bgr)
    plt.colorbar()


    plt.subplot(rows, cols, 2)
    if len(gs.shape) == 3:
        plt.imshow(cv2.cvtColor(gs, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(gs)
    plt.colorbar()

    plt.show()


def augment_data(train_images, 
                 truth_images,
                 ap=[[0.0, 1.0], [180.0, 1.0]]):

    # ap - augment parameters
    # ap[i][0] - rotation angle
    # ap[i][1] - scale

    augmented_images_dir = "augmented_images/"
    is_dir = os.path.isdir(augmented_images_dir)
    if (is_dir == True):
        shutil.rmtree(augmented_images_dir)
        os.makedirs(augmented_images_dir)
    else:
        os.makedirs(augmented_images_dir)


    n_images = len(train_images)
    iw, ih, ic = train_images[0].shape
    mw, mh = truth_images[0].shape


    if ((iw != mw) or (mh != mh)):
        sys.exit("ERROR: Dimension mismatch!")

    for i in range(n_images):
            for j in range(len(ap)):

                rotation_angle = ap[j][0]
                scale = ap[j][1]

                R = cv2.getRotationMatrix2D((iw/2, ih/2), rotation_angle, scale)

                rotated_train_image = cv2.warpAffine(train_images[i], R, (iw, ih))
                rotated_truth_image = cv2.warpAffine(truth_images[i], R, (mw, mh))
        
                cv2.imwrite(augmented_images_dir + \
                            "train_image_id_" + \
                            str(i) + "_" + \
                            str(int(rotation_angle)) + "_" + \
                            str(scale).replace(".","p") + ".tif", \
                            rotated_train_image)

                cv2.imwrite(augmented_images_dir + \
                            "truth_image_id_" + \
                            str(i) + "_" + \
                            str(int(rotation_angle)) + "_" + \
                            str(scale).replace(".","p") + ".png", \
                            rotated_truth_image)







