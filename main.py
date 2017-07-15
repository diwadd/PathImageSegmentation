import random
import sys

import numpy as np

import data_handling as dh
import deep_models as dm

#random.seed(1)

if __name__ == "__main__":
    print("Starting main.")

    # ------------------------------------------------------------------------
    # Load data        
    # ------------------------------------------------------------------------

    train_images, truth_images = dh.read_data(data_dir="augmented_images/")
    iw, ih, ic = train_images[0].shape

    if len(train_images) != len(truth_images):
        sys.exit("ERROR: Dimension mismatch.")
    n_images = len(train_images)

    print("Data loaded.")
    print("Number of train images:" + str(len(train_images)))
    print("Number of truth images:" + str(len(truth_images)))
    print("Train image size: " + str(train_images[0].shape))
    print("Truth image size: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Convert resized truth images to
    # gray scale      
    # ------------------------------------------------------------------------

    grayscale_resized_truth_images = dh.convert_image_list_to_grayscale(truth_images)
    print("Truth images converted to grayscale.")
    print("Truth image after conversion to grayscale: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Resize truth images        
    # ------------------------------------------------------------------------

    nw = 50
    nh = 50

    resized_truth_images = dh.resize_image_list(grayscale_resized_truth_images, nw, nh)
    print("Resized truth images.")
    print("Truth image after resize: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Shuffle the data
    # ------------------------------------------------------------------------
    # We use zip to maintain the
    # correspondence image <-> truth.
    # truth_images is icluded so that we can
    # later compare back resized truth images
    # with the original truth images.
    # ------------------------------------------------------------------------

    train_truth_image_list = list(zip(train_images, resized_truth_images, truth_images))
    random.shuffle(train_truth_image_list)
    train_images, resized_truth_images, truth_images = zip(*train_truth_image_list)

    print("Data shuffled.")
    print("Number of train images:" + str(len(train_images)))
    print("Number of resized truth images:" + str(len(resized_truth_images)))
    print("Train image size: " + str(train_images[0].shape))
    print("Resized truth image size: " + str(resized_truth_images[0].shape))

    dh.plot_two_images(train_images[0], resized_truth_images[0])

    # ------------------------------------------------------------------------
    # Back resize the truth data.
    # ------------------------------------------------------------------------

    back_resized_truth_images = dh.resize_image_list(resized_truth_images, iw, ih)
    dh.plot_two_images(truth_images[0], back_resized_truth_images[0])

    # ------------------------------------------------------------------------
    # Scale the images to [0, 1]
    # ------------------------------------------------------------------------

    train_images = [train_images[i]/255.0 for i in range(n_images)]
    resized_truth_images = [resized_truth_images[i]/255.0 for i in range(n_images)]

    # ------------------------------------------------------------------------
    # List to numpy arrays (~, iw, ih, ic)
    # ------------------------------------------------------------------------


    train_images_array = np.zeros((n_images, iw, ih, ic))
    resized_truth_images_array = np.zeros((n_images, nw*nh))

    for i in range(n_images):
        train_images_array[i, :, :, :] = train_images[i]
        resized_truth_images_array[i, :] = np.reshape(resized_truth_images[i], (nw*nh))


    model = dm.basic_model(iw=iw, 
                           ih=ih, 
                           ic=ic,
                           ow=nw,
                           oh=nh,
                           dropout=0.1,
                           alpha=0.001)

    model.fit(train_images_array,
              resized_truth_images_array,
              epochs=100,
              batch_size=16)


    model.save("model.h5")



