import random
import sys
import glob

import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras import optimizers

import tensorflow as tf

import data_handling as dh
import deep_models as dm

# When using keras_fcn we test the FCN VGG16 network.
# In our test we use the keras implementation by JihongJu
# See: https://github.com/JihongJu/keras-fcn
# As of July 2017 this implementation is published under the MIT License.
from keras_fcn import FCN
from keras_fcn.layers import BilinearUpSampling2D

random.seed(111)



def data_generator(file_name_list, noli=20):
    
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


def evaluate(model, file_name_list, noli=20):
    
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

        #print("score: " + str(score))

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


        #model = FCN(input_shape=(iw, ih, ic), classes=2,  
        #            weights='imagenet', trainable_encoder=True)

        model = dm.vgg16_16s_fcn(iw, 
                                 ih, 
                                 ic,
                                 dropout=0.5,
                                 alpha=0.0001,
                                 classes=2)

        opt = optimizers.SGD(lr=0.0001, momentum=0.9, clipvalue=0.5)
        opt = optimizers.Adam(1e-4, clipvalue=0.5)
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])


        vgg16_model = VGG16(weights='imagenet', include_top=False)
        vgg16_model.compile(loss="categorical_crossentropy",
                      optimizer="adadelta")

        index = 12
        print("Our model config: ")
        print(model.layers[index].get_config())

        print("VGG16 model config: ")
        print(vgg16_model.layers[index].get_config())

        print("Our model:")
        layer1 = model.layers[index].get_weights()
        print(layer1[0].shape)
        print(layer1[0][0,0,0,1:5])

        print("VGG16 model:")
        layer1 = vgg16_model.layers[index].get_weights()
        print(layer1[0].shape)
        print(layer1[0][0,0,0,1:5])


        for i in range(len(vgg16_model.layers)):

            print("\n\n\n Layer number: " + str(i))

            layer_config = vgg16_model.layers[i].get_config()
            print("Setting name: " + str(layer_config['name']))
            if layer_config['name'] == 'block5_pool':
                break

            layer_weights = vgg16_model.layers[i].get_weights()

            if len(layer_weights) != 0:
                print(layer1[0].shape)
                model.layers[i].set_weights(vgg16_model.layers[i].get_weights())
            else:
                print("No wieghts to set!")

        print("After setting the weights...")

        print("Our model:")
        layer1 = model.layers[index].get_weights()
        print(layer1[0].shape)
        print(layer1[0][0,0,0,1:5])

        print("VGG16 model:")
        layer1 = vgg16_model.layers[index].get_weights()
        print(layer1[0].shape)
        print(layer1[0][0,0,0,1:5])



        #model.compile(optimizer='adadelta',
        #                  loss='categorical_crossentropy',
        #                  metrics=['accuracy'])

        model.summary()


    else:
        model = load_model("model.h5", custom_objects={"BilinearUpSampling2D": BilinearUpSampling2D})
        print("Model loaded!")
        model.summary()

    noli = 10
    n = len(x_train_fnl)
    number_of_image_loads = round(n / noli)
    print("Number of image loads: " + str(number_of_image_loads))
    n_epochs = 100    
    n_sub_epochs = 5

    print("Pre training evaluation:")
    evaluate(model, x_valid_fnl, noli=20)

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

