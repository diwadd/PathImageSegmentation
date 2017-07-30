import glob
import cv2
import math
import sys
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500

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


def plot_two_images(bgr, gs, img):

    rows = 1
    cols = 3
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

    plt.subplot(rows, cols, 3)
    if len(gs.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.colorbar()

    plt.show()


def augment_data(train_images, 
                 truth_images,
                 nw_image=250,
                 nh_image=250,
                 nw_label=50,
                 nh_label=50,
                 augmented_images_dir="augmented_images/",
                 ap=[[0.0, 1.0], [180.0, 1.0]],
                 save_images=True):

    # ap - augment parameters
    # ap[i][0] - rotation angle
    # ap[i][1] - scale

    is_dir = os.path.isdir(augmented_images_dir)
    if (is_dir == True):
        shutil.rmtree(augmented_images_dir)
        os.makedirs(augmented_images_dir)
    else:
        os.makedirs(augmented_images_dir)


    n_images = len(train_images)
    iw, ih, ic = train_images[0].shape
    mw, mh, _ = truth_images[0].shape


    if ((iw != mw) or (mh != mh)):
        sys.exit("ERROR: Dimension mismatch!")

    for i in range(n_images):
        print("Image: " + str(i) + "/" + str(n_images), end="\r")
        for j in range(len(ap)):

            rotation_angle = ap[j][0]
            scale = ap[j][1]

            R = cv2.getRotationMatrix2D((iw/2, ih/2), rotation_angle, scale)

            rotated_train_image = cv2.warpAffine(train_images[i], R, (iw, ih))
            rotated_truth_image = cv2.warpAffine(truth_images[i], R, (mw, mh))

            x_shift = ap[j][2]
            y_shift = ap[j][3]

            M = np.float32([[1, 0, x_shift],[0, 1, y_shift]])

            shifted_train_image = cv2.warpAffine(rotated_train_image, M, (iw, ih))
            shifted_truth_image = cv2.warpAffine(rotated_truth_image, M, (mw, mh))

            grayscale_truth_image = cv2.cvtColor(shifted_truth_image, cv2.COLOR_BGR2GRAY)

            resized_train_image = cv2.resize(shifted_train_image, 
                                             (nw_image, nh_image), 
                                             interpolation = cv2.INTER_LINEAR)

            resized_truth_image = cv2.resize(grayscale_truth_image, 
                                             (nw_label, nh_label), 
                                             interpolation = cv2.INTER_LINEAR)

            if save_images == True:
                cv2.imwrite(augmented_images_dir + \
                            "train_image_id_" + \
                            str(i) + "_" + \
                            str(rotation_angle).replace(".","p") + "_" + \
                            str(x_shift) + "_" + \
                            str(y_shift) + "_" + \
                            str(scale).replace(".","p") + ".tif", \
                            shifted_train_image)

                cv2.imwrite(augmented_images_dir + \
                            "truth_image_id_" + \
                            str(i) + "_" + \
                            str(rotation_angle).replace(".","p") + "_" + \
                            str(x_shift) + "_" + \
                            str(y_shift) + "_" + \
                            str(scale).replace(".","p") + ".png", \
                            shifted_truth_image)

            fn = augmented_images_dir + \
                 "numpy_image_array_id_" + \
                 str(i) + "_" + \
                 str(rotation_angle).replace(".","p") + "_" + \
                 str(x_shift) + "_" + \
                 str(y_shift) + "_" + \
                 str(scale).replace(".","p") + ".npz"

            resized_truth_image[resized_truth_image > 0] = 1

            # This part is for FCN.
            # We destinguish between good and bad tissue so
            # we have just two classes.
            n_classes = 2
            resized_truth_image = to_categorical(resized_truth_image, n_classes)
            resized_truth_image = np.reshape(resized_truth_image, (nw_label, nh_label, n_classes))

            np.savez_compressed(fn, 
                                image=resized_train_image/255.0, 
                                #label=np.reshape(resized_truth_image, (nw_label*nh_label, 1)))
                                label=resized_truth_image)


def load_data_from_npz(file_name_list):

    n_files = len(file_name_list)
    if n_files == 0:
        sys.exit("ERROR: File name list empty.")

    loaded_data = np.load(file_name_list[0])
    image = loaded_data["image"].astype(np.float32)
    label = loaded_data["label"].astype(np.float32)

    iw, ih, ic = image.shape
    mw, mh, mc = label.shape

    x_data = np.zeros((n_files, ih, iw, ic))
    y_data = np.zeros((n_files, mw, mh, mc))

    x_data[0, :, :, :] = image
    y_data[0, :, :, :] = label

    for i in range(1, n_files):
        loaded_data = np.load(file_name_list[i])
        image = loaded_data["image"].astype(np.float32)
        label = loaded_data["label"].astype(np.float32)

        x_data[i, :, :, :] = image
        y_data[i, :, :, :] = label

    return x_data, y_data


