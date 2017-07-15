import glob
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt

def load_images(file_names_list):

    n_images = len(file_names_list)

    images = [None for i in range(n_images)]
    for i in range(n_images):
        images[i] = (cv2.imread(file_names_list[i]))

    return images


def read_training_data(train_dir="training/images/",
                       truth_dir="training/truth/"):


    train_file_names_list = glob.glob(train_dir + "*.tif")
    n_train_images = len(train_file_names_list)

    # The truth images have the same numbers.
    # We change the path and the extensions.
    # This ensures that train image will
    # correspond to truth image.
    truth_file_names_list = [None for i in range(n_train_images)]
    for i in range(n_train_images):
        truth_file_names_list[i] = train_file_names_list[i].replace(".tif","_mask.png").replace("images", "truth")

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
