import random
import sys
import glob

import cv2
import numpy as np

from keras import backend as K
from keras.models import load_model
from tensorflow.python.ops import math_ops

import deep_models as dm


# model = dm.vg16_fcn()
# model.summary()
# model.compile(loss="categorical_crossentropy",
#                  optimizer="adadelta")


def cross_entropy(t, p):
    return -t*np.log(p)




true_arr = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], 
                     [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                     [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],])


pred_arr = np.array([[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]], 
                     [[0.9, 0.1], [0.4, 0.6], [0.5, 0.5]],
                     [[0.9, 0.1], [0.2, 0.8], [0.9, 0.1]]])

print(true_arr.shape)


ce = cross_entropy(true_arr, pred_arr)
print("ce: " + str(ce))

K.get_session()

true_var = K.variable(true_arr)
pred_var = K.variable(pred_arr)

# output = pred_var / math_ops.reduce_sum(pred_var, reduction_indices=len(pred_var.get_shape()) - 1, keep_dims=True)
# print(K.eval(output))

en = K.categorical_crossentropy(pred_var, true_var, from_logits=False)

en_eval = K.eval(en)
print(en_eval)

K.clear_session()





