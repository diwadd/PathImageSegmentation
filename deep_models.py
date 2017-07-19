import keras

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D


from keras.layers import Dropout 
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras import regularizers
from keras import backend as K

def basic_model(iw=500, # Input width
                ih=500, # Input height 
                ic=3,
                ow=100, # Output width
                oh=100, # Output heigth
                dropout=0.9,
                alpha=0.001):

    input_image = Input((iw, ih, ic))

    x = Conv2D(16, (3, 3), activation="linear", padding="same", name="block_1_layer_1")(input_image)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(16, (3, 3), activation="linear", padding="same", name="block_1_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_1_pooling")(x)

    x = Conv2D(32, (3, 3), activation="linear", padding="same", name="block_2_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(32, (3, 3), activation="linear", padding="same", name="block_2_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_2_pooling")(x)

    x = Conv2D(64, (3, 3), activation="linear", padding="same", name="block_3_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(64, (3, 3), activation="linear", padding="same", name="block_3_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(64, (3, 3), activation="linear", padding="same", name="block_3_layer_3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_3_pooling")(x)

    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_4_layer_1")(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_4_layer_2")(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_4_layer_3")(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_4_pooling")(x)

    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_5_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_5_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_5_layer_3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_5_pooling")(x)

    #x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_6_layer_1")(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(alpha)(x)
    #x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_6_layer_2")(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(alpha)(x)
    #x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_6_layer_3")(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(alpha)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name="block_6_pooling")(x)

    x = Conv2D(256, (3, 3), activation="linear", padding="same", name="block_7_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(256, (3, 3), activation="linear", padding="same", name="block_7_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(256, (3, 3), activation="linear", padding="same", name="block_7_layer_3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_7_pooling")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(4096, activation="linear", name="full_connected_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(dropout)(x)

    x = Dense(4096, activation="linear", name="full_connected_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(dropout)(x)

    x = Dense(ow*oh, activation="sigmoid", name="predictions")(x)

    model = Model(input_image, x, name="vgg16_based")

    model.compile(loss="binary_crossentropy",
                  optimizer="adadelta")

    print("\n ---> Model summary <--- \n")
    model.summary()

    return model

def basic_model_pooling(iw=500, # Input width
                        ih=500, # Input height 
                        ic=3,
                        ow=100, # Output width
                        oh=100, # Output heigth
                        dropout=0.9,
                        alpha=0.001):

    input_image = Input((iw, ih, ic))

    x = Conv2D(16, (3, 3), activation="linear", padding="same", name="block_1_layer_1")(input_image)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(16, (3, 3), activation="linear", padding="same", name="block_1_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_1_pooling")(x)

    x = Conv2D(32, (3, 3), activation="linear", padding="same", name="block_2_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(32, (3, 3), activation="linear", padding="same", name="block_2_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_2_pooling")(x)

    x = Conv2D(64, (3, 3), activation="linear", padding="same", name="block_3_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(64, (3, 3), activation="linear", padding="same", name="block_3_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(64, (3, 3), activation="linear", padding="same", name="block_3_layer_3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_3_pooling")(x)

    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_4_layer_1")(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_4_layer_2")(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_4_layer_3")(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_4_pooling")(x)

    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_5_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_5_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(128, (3, 3), activation="linear", padding="same", name="block_5_layer_3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block_5_pooling")(x)

    x = Conv2D(256, (3, 3), activation="linear", padding="same", name="block_6_layer_1")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(512, (3, 3), activation="linear", padding="same", name="block_6_layer_2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(ow*oh, (3, 3), activation="linear", padding="same", name="block_6_layer_3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name="block_6_pooling")(x)

    #x = Conv2D(256, (3, 3), activation="linear", padding="same", name="block_7_layer_1")(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(alpha)(x)
    #x = Conv2D(256, (3, 3), activation="linear", padding="same", name="block_7_layer_2")(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(alpha)(x)
    #x = Conv2D(256, (3, 3), activation="linear", padding="same", name="block_7_layer_3")(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(alpha)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name="block_7_pooling")(x)

    x = GlobalAveragePooling2D()(x)

    model = Model(input_image, x, name="vgg16_based")

    model.compile(loss="binary_crossentropy",
                  optimizer="adadelta")

    print("\n ---> Model summary <--- \n")
    model.summary()

    return model



