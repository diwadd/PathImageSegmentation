import random
import sys
import glob
import os
import shutil

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model

from keras_fcn.layers import BilinearUpSampling2D

import data_handling as dh
import deep_models as dm

def load_and_resize_image(image_file_name,
                          nw_image=250,
                          nh_image=250):

    image = cv2.imread(image_file_name)

    resized_image = cv2.resize(image, 
                               (nw_image, nh_image), 
                               interpolation = cv2.INTER_LINEAR)
    return resized_image


def prepare_data_for_dispatch(test_images_list,
                              nw_image=250,
                              nh_image=250,
                              nw_label=50,
                              nh_label=50,
                              images_dir="images_for_dispatch/"):

    # ap - augment parameters
    # ap[i][0] - rotation angle
    # ap[i][1] - scale

    is_dir = os.path.isdir(images_dir)
    if (is_dir == True):
        shutil.rmtree(images_dir)
        os.makedirs(images_dir)
    else:
        os.makedirs(images_dir)

    K.get_session()
    model = load_model("model.h5", custom_objects={"BilinearUpSampling2D": BilinearUpSampling2D})

    n_train_images = len(test_images_list)
    for i in range(n_train_images):
        print("i: " + str(i + 1))
        resized_image = load_and_resize_image(test_images_list[i],
                                              nw_image=nw_image,
                                              nh_image=nh_image)

        # The the i000000 image indicator
        # from training/images/i000000.tif
        if test_images_list[i][0:3] == "tra":
            trunk = test_images_list[i][16:-4]
        elif test_images_list[i][0:3] == "tes":
            trunk = test_images_list[i][15:-4]
        else:
            pass

        print(trunk)
        fn = images_dir + str(trunk) + ".npz"

        _, _, ic = resized_image.shape

        resized_image = np.reshape(resized_image, (1, nw_image, nh_image, ic))/255.0
        label = model.predict(resized_image)
        label = np.reshape(label, (nw_label, nh_label))

        label = cv2.resize(label,
                           (dh.DEFAULT_WIDTH, dh.DEFAULT_HEIGHT),
                           interpolation = cv2.INTER_LINEAR)

        f = open(images_dir + trunk + "_mask.txt", "w")

        for v in range(dh.DEFAULT_WIDTH):
            for w in range(dh.DEFAULT_HEIGHT):
                if (label[v][w] > 0.5):
                    f.write("1")
                else:
                    f.write("0")
            f.write("\n")

        f.close()

        #np.savez_compressed(fn, 
        #                    image=resized_image/255.0, 
        #                    trunk=trunk)



    K.clear_session()


if __name__ == "__main__":
    print("Starting main.")

    # ------------------------------------------------------------------------
    # Load data        
    # ------------------------------------------------------------------------

    train_images_list = glob.glob("training/images/*.tif")
    test_images_list = glob.glob("testing/images/*.tif")

    image_list = train_images_list + test_images_list

    prepare_data_for_dispatch(image_list,
                              nw_image=224,
                              nh_image=224,
                              nw_label=224,
                              nh_label=224,
                              images_dir="images_for_dispatch/")




