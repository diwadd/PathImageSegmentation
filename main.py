import random
import sys
import glob

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras_fcn.layers import BilinearUpSampling2D

import data_handling as dh

def data_generator(file_name_list, noli=64):
    
    """
    noli - number of loaded images per yield
    
    """

    while True:
        n = len(file_name_list)
        number_of_image_loads = round(n / noli)
        ptr = 0
        # print("n: " + str(n))
        # print("number_of_image_loads: " + str(number_of_image_loads))

        for i in range(number_of_image_loads):
            # print("We are a i: " + str(i))
            # create numpy arrays of input data
            # and labels, from each line in the file
            mini_batch_fnl = file_name_list[ptr:(ptr + noli)]
            ptr = ptr + noli

            x_data, y_data = dh.load_data_from_npz(mini_batch_fnl)
            yield (x_data, y_data)


def evaluate(model, file_name_list, noli=64):
    
    """
    noli - number of loaded images per yield
    
    """

    n = len(file_name_list)
    number_of_image_loads = round(n / noli)
    ptr = 0
    # print("n: " + str(n))
    # print("number_of_image_loads: " + str(number_of_image_loads))

    mean_score = 0.0
    for i in range(number_of_image_loads):
        # print("We are a i: " + str(i))
        # create numpy arrays of input data
        # and labels, from each line in the file
        mini_batch_fnl = file_name_list[ptr:(ptr + noli)]
        ptr = ptr + noli

        x_data, y_data = dh.load_data_from_npz(mini_batch_fnl)
        score = model.evaluate(x_data, y_data, verbose=0)
        mean_score += score
    mean_score /= number_of_image_loads
    print("Mean score: " + str(mean_score))

        
def bc(t, o):
    return -np.mean(t*np.log(o) + (1-t)*np.log(1-o))



if __name__ == "__main__":

    K.get_session()

    # model = load_model("vgg16_32s_fcn_model.h5")
    model = load_model("vgg16_16s_fcn_model_after_global_epoch_22.h5")

    file_name = "augmented_images/numpy_image_array_id_0_270p0_0_0_1p0.npz"
    loaded_data = np.load(file_name)

    image = loaded_data["image"]
    label = loaded_data["label"]
    label = dh.transform_label(label)

    iw, ih, ic = image.shape

    predicted_label = model.predict(np.reshape(image, (1, iw, ih, ic)))
    predicted_label = np.reshape(predicted_label, (iw, ih, 2))
    predicted_label = dh.transform_label(predicted_label)

    image = (255.0*image).astype(np.uint8)
    label =  label.astype(np.uint8)

    print("image shape: " + str(image.shape))
    print("label shape: " + str(label.shape))
    print("predicted_label shape: " + str(predicted_label.shape))

    #dh.plot_two_images(label, predicted_label[:,:,0], predicted_label[:,:,1])
    dh.plot_two_images(image, label, predicted_label)

    """
    print("label <-> predicted label bc: " + str(bc(label/255.0, predicted_label)))

    predicted_label = cv2.resize(predicted_label,
                                 (500, 500),
                                 interpolation = cv2.INTER_LINEAR)

    label = cv2.resize(label,
                       (500, 500),
                       interpolation = cv2.INTER_LINEAR)


    original_label = cv2.cvtColor(original_label, cv2.COLOR_BGR2GRAY)/255.0

    #print(predicted_label)
    #predicted_label[ predicted_label < 0.5 ] = 0.01
    #predicted_label[ predicted_label >= 0.5 ] = 0.99

    dh.plot_two_images(original_label, predicted_label, original_label - predicted_label)

    print("original_label <-> predicted_label bc: " + str(bc(original_label, predicted_label)))


    small_original_label = cv2.resize(original_label,
                                     (100, 100),
                                     interpolation = cv2.INTER_AREA)

    big_original_label = cv2.resize(small_original_label,
                                    (500, 500),
                                    interpolation = cv2.INTER_CUBIC)



    #print("original_label <-> big_original_label bc: " + str(bc(original_label, big_original_label)))
    """

    K.clear_session()





