import glob
import cv2
import math
import random

import numpy as np
import matplotlib.pyplot as plt

#random.seed(1)

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

if __name__ == "__main__":
    print("Starting main.")

    # ------------------------------------------------------------------------
    # Load data        
    # ------------------------------------------------------------------------

    train_images, truth_images = read_training_data()
    iw, ih, ic = train_images[0].shape

    print("Data loaded.")
    print("Number of train images:" + str(len(train_images)))
    print("Number of truth images:" + str(len(truth_images)))
    print("Train image size: " + str(train_images[0].shape))
    print("Truth image size: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Resize truth images        
    # ------------------------------------------------------------------------

    n_width = 100
    n_height = 100

    resized_truth_images = resize_image_list(truth_images, n_width, n_height)
    print("Resized truth images.")
    print("Truth image after resize: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Convert resized truth images to
    # gray scale      
    # ------------------------------------------------------------------------

    grayscale_resized_truth_images = convert_image_list_to_grayscale(resized_truth_images)
    print("Truth images converted to grayscale.")
    print("Truth image after conversion to grayscale: " + str(truth_images[0].shape))
    

    # ------------------------------------------------------------------------
    # Shuffle the data
    # ------------------------------------------------------------------------
    # Shuffle the images
    # We use zip to maintain the
    # correspondence image <-> truth.
    # truth_images is icluded so that we can
    # later compare back resized truth images
    # with the original truth images.
    # ------------------------------------------------------------------------

    train_truth_image_list = list(zip(train_images, grayscale_resized_truth_images, truth_images))
    random.shuffle(train_truth_image_list)
    train_images, grayscale_resized_truth_images, truth_images = zip(*train_truth_image_list)

    print("Data shuffled.")
    print("Number of train images:" + str(len(train_images)))
    print("Number of grayscale resized truth images:" + str(len(grayscale_resized_truth_images)))
    print("Train image size: " + str(train_images[0].shape))
    print("Grayscale resized truth image size: " + str(grayscale_resized_truth_images[0].shape))

    plot_two_images(train_images[0], grayscale_resized_truth_images[0])

    # ------------------------------------------------------------------------
    # Back resize the truth data.
    # ------------------------------------------------------------------------

    back_resized_truth_images = resize_image_list(grayscale_resized_truth_images, iw, ih)
    plot_two_images(truth_images[0], back_resized_truth_images[0])

    # ------------------------------------------------------------------------
    # Scale the images to [0, 1]
    # ------------------------------------------------------------------------

    train_images = [train_images[i]/255.0 for i in range(len(train_images))]
    truth_images = [truth_images[i]/255.0 for i in range(len(truth_images))]





