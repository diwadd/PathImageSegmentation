import random
import sys
import glob

import numpy as np

from keras import backend as K
from keras.models import load_model

import data_handling as dh
import deep_models as dm

# When using keras_fcn we test the FCN VGG16 network.
# In our test we use the keras implementation by JihongJu
# See: https://github.com/JihongJu/keras-fcn
# As of July 2017 this implementation is published under the MIT License.
from keras_fcn import FCN

random.seed(111)

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

    mean_loss = 0.0
    mean_acc = 0.0
    for i in range(number_of_image_loads):
        # print("We are a i: " + str(i))
        # create numpy arrays of input data
        # and labels, from each line in the file
        mini_batch_fnl = file_name_list[ptr:(ptr + noli)]
        ptr = ptr + noli

        x_data, y_data = dh.load_data_from_npz(mini_batch_fnl)
        score = model.evaluate(x_data, y_data, verbose=0)

        local_loss = score[0]
        local_acc = score[1]

        mean_loss += local_loss
        mean_acc += local_acc
    mean_loss /= number_of_image_loads
    mean_acc /= number_of_image_loads
    print("Mean loss: " + str(mean_loss) + " - mean acc: " + str(mean_acc))
        


if __name__ == "__main__":
    print("Starting main.")

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    data_file_names = glob.glob("augmented_images/*npz")

    n_files = len(data_file_names)

    loaded_data = np.load(data_file_names[0])
    image = loaded_data["image"]
    label = loaded_data["label"]

    iw, ih, ic = image.shape
    nw, nh, nc= label.shape

    random.shuffle(data_file_names)
    train_fraction = 0.6
    valid_fraction = 1.0 - train_fraction
    test_fraction = 0.5

    x_train_fnl = data_file_names[0:int(train_fraction*n_files)]
    temp = data_file_names[int(train_fraction*n_files):]
    x_valid_fnl = temp[0:int(len(temp)/2)]
    x_test_fnl = temp[int(len(temp)/2):]

    print("First train file: " + str(x_train_fnl[0]))

    print("Number of train data files: " + str(len(x_train_fnl)))
    print("Number of valid data files: " + str(len(x_valid_fnl)))
    print("Number of test data files: " + str(len(x_test_fnl)))

    K.get_session()


    new_model = True
    if (new_model == True):
        print("Creating a new model!")
        """
        model = dm.basic_model(iw=iw, 
                               ih=ih, 
                               ic=ic,
                               ow=nw,
                               oh=nh,
                               dropout=0.1,
                               alpha=0.001)
        """

        model = FCN(input_shape=(iw, ih, ic), classes=2,  
                    weights='imagenet', trainable_encoder=True)

        model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        model.summary()


    else:
        model = load_model("model.h5")
        print("Model loaded!")

    noli = 10
    n = len(x_train_fnl)
    number_of_image_loads = round(n / noli)
    print("Number of image loads: " + str(number_of_image_loads))
    n_epochs = 20    
    n_sub_epochs = 5

    print("Pre training evaluation:")
    evaluate(model, x_valid_fnl, noli=64)

    for i in range(n_epochs):
        print("Global epoch: " + str(i) + "/" + str(n_epochs))
        model.fit_generator(data_generator(data_file_names, noli=noli),
                            steps_per_epoch=number_of_image_loads,
                            epochs=n_sub_epochs)
        print("Validation data mean loss: ")
        evaluate(model, x_valid_fnl, noli=64)


        model.save("model.h5")

    print("Test data mean loss: ")
    evaluate(model, x_test_fnl, noli=64)


    K.clear_session()

