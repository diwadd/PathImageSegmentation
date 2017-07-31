import keras

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Cropping2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Activation
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
                alpha=0.0):

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


def vgg16_32s_fcn(iw=500, # Input width
                  ih=500, # Input height 
                  ic=3,
                  dropout=0.5,
                  alpha=0.001,
                  classes=2):
    # Based on:
    # Fully Convolutional Models for Semantic Segmentation
    # Evan Shelhamer*, Jonathan Long*, Trevor Darrell
    # PAMI 2016
    # arXiv:1605.06211

    reg_fun = regularizers.l2(alpha)

    input_image = Input((iw, ih, ic))

    # Conv 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block1_conv1')(input_image)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block1_conv2')(x)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Conv 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block2_conv1')(pool1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block2_conv2')(x)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Conv 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block3_conv1')(pool2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block3_conv3')(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Conv 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block4_conv1')(pool3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Conv 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block5_conv1')(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Fully Conv fc6
    fc6 = Conv2D(4096, (7, 7), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='fc6')(pool5)
    drop6 = Dropout(rate=dropout)(fc6)

    # Fully Conv fc7
    fc7 = Conv2D(4096, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='fc7')(drop6)
    drop7 = Dropout(rate=dropout)(fc7)

    score_fr = Conv2D(classes, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='score_fr')(drop7)

    upscore = Conv2DTranspose(classes, kernel_size=(64, 64), strides=(32, 32), kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='upscore')(score_fr)


    _, uw, uh, uc = upscore._keras_shape
    cw = (uw - iw)//2
    ch = (uh - ih)//2
    print("cw: " + str(cw))
    print("ch: " + str(ch))

    score = Cropping2D(cropping=(cw, ch))(upscore)
    output = Activation('softmax')(score)

    model = Model(input_image, output, name="vgg16_based")

    return model


def vgg16_16s_fcn(iw=500, # Input width
                  ih=500, # Input height 
                  ic=3,
                  dropout=0.5,
                  alpha=0.001,
                  classes=2):
    # Based on:
    # Fully Convolutional Models for Semantic Segmentation
    # Evan Shelhamer*, Jonathan Long*, Trevor Darrell
    # PAMI 2016
    # arXiv:1605.06211

    reg_fun = regularizers.l2(alpha)

    input_image = Input((iw, ih, ic))

    # Conv 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block1_conv1')(input_image)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block1_conv2')(x)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Conv 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block2_conv1')(pool1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block2_conv2')(x)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Conv 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block3_conv1')(pool2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block3_conv3')(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Conv 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block4_conv1')(pool3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Conv 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block5_conv1')(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Fully Conv fc6
    fc6 = Conv2D(4096, (7, 7), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='fc6')(pool5)
    drop6 = Dropout(rate=dropout)(fc6)

    # Fully Conv fc7
    fc7 = Conv2D(4096, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='fc7')(drop6)
    drop7 = Dropout(rate=dropout)(fc7)

    score_fr = Conv2D(classes, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='score_fr')(drop7)
    upscore2 = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='upscore2')(score_fr)


    score_pool4 = Conv2D(classes, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='score_pool4')(pool4)

    _, uw, uh, uc = upscore2._keras_shape
    _, sw, sh, sc = score_pool4._keras_shape
    cw1 = (uw - sw)//2
    ch1 = (uh - sh)//2
    #print("cw1: " + str(cw1))
    #print("ch1: " + str(ch1))

    # Technically score_pool4 should have a larger size then upscore2.
    # At least that is what follows from crop(n.score_pool4, n.upscore2).
    # This is, however, not the case and we nned to crop upscore2.

    score_pool4c = Cropping2D(cropping=(cw1, ch1))(upscore2) 
    fuse_pool4 = keras.layers.Add()([score_pool4c, score_pool4])

    upscore16 = Conv2DTranspose(classes, kernel_size=(32, 32), strides=(16, 16), kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha), name='upscore16')(fuse_pool4)

    _, uw, uh, uc = upscore16._keras_shape
    cw2 = (uw - iw)//2
    ch2 = (uh - ih)//2
    #print("cw2: " + str(cw2))
    #print("ch2: " + str(ch2))

    score = Cropping2D(cropping=(cw2, ch2))(upscore16)
    output = Activation('softmax')(score)

    model = Model(input_image, output, name="vgg16_based")

    return model




